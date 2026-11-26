#!/usr/bin/env python
"""
Stage 5 — cross-cohort validation (Reviewer 3, point 4).

The discovery cohort (ROS/MAP) and an independent proteomic cohort (e.g. Mount Sinai Brain Bank, or Mayo) are each processed
with stage 1 (the external cohort with --cohort-table), giving two differential PPI graphs with their own node features.

For every discovery CV fold the GAT is trained ONLY on the fold's training genes of the discovery graph, then applied, without any
re-training, to the external graph. It is scored on that fold's held-out genes that are present in the external network.
Every gene is therefore (i) unseen by the model as a label and (ii) scored on proteomic data from a different cohort,
brain region and TMT batch structure. Pooled out-of-fold scores give the headline AUROC/AUPRC with bootstrap CIs.

Optionally (if stage 2 was also run on the external root) the rankings of unlabelled proteins are compared between cohorts
(Spearman rho, overlap of the top-N).

    python src/stage5_external_validation.py --root data --external-root data_msbb
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import common as C
import stage2_train as S2


def boot_ci(y, p, fn, n=C.N_BOOT, seed=C.SEED_NUMPY):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data", help="discovery cohort root (ROS/MAP)")
    ap.add_argument("--external-root", required=True, help="external cohort root (stage 1 already run)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--monitor", choices=["auroc", "loss"], default="auroc")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=100)
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    pa, pb = C.get_paths(a.root), C.get_paths(a.external_root)
    name = Path(a.external_root).name
    out = pa.res / f"external_{name}"
    out.mkdir(exist_ok=True)
    C.set_seeds()
    data_a, info_a = S2.load_graph(pa.proc)
    data_b, info_b = S2.load_graph(pb.proc)
    ga = {g: i for i, g in enumerate(info_a.gene)}
    gb = {g: i for i, g in enumerate(info_b.gene)}
    common = sorted(set(ga) & set(gb))
    lab_common = info_a.set_index("gene").label.reindex(common)
    print(f"[external] discovery graph {len(ga)} nodes | external graph {len(gb)} nodes | shared {len(common)} | "
          f"labelled & shared: {(lab_common >= 0).sum()} ({int((lab_common == 1).sum())} drivers)")
    Xa = info_a[[f + "_z" for f in C.FEATURES]].values
    Xb = info_b[[f + "_z" for f in C.FEATURES]].values

    rows, pooled = [], {"GAT": [], "LogReg": [], "Degree only (external)": [], "GAT (discovery, same genes)": []}
    for sp in S2.make_splits(info_a, a.repeats):
        tr, va, te, trv = sp["tr"], sp["va"], sp["te"], sp["trv"]
        C_seed = C.SEED_TORCH + sp["k"]
        import torch
        torch.manual_seed(C_seed)
        m, best = S2.train_model(S2.make_gat({}), data_a, tr, va, epochs=a.epochs, patience=a.patience, monitor=a.monitor)
        p_a, p_b = S2.predict(m, data_a), S2.predict(m, data_b)
        lr = LogisticRegression(max_iter=2000).fit(Xa[trv], info_a.label.values[trv].astype(int))
        genes = info_a.gene.values[te]
        keep = np.array([g in gb for g in genes])
        yt = info_a.label.values[te][keep].astype(int)
        ib = np.array([gb[g] for g in genes[keep]])
        ia = te[keep]
        sc = {"GAT": p_b[ib], "LogReg": lr.predict_proba(Xb[ib])[:, 1], "Degree only (external)": info_b.degree.values[ib].astype(float),
              "GAT (discovery, same genes)": p_a[ia]}
        for k, v in sc.items():
            pooled[k].append((sp["rep"], yt, v))
            if 0 < yt.sum() < len(yt):
                rows.append(dict(repeat=sp["rep"], fold=sp["fold"], model=k, n_test=len(yt), n_pos=int(yt.sum()),
                                 AUROC=roc_auc_score(yt, v), AUPRC=average_precision_score(yt, v)))
        print(f"[external] rep{sp['rep']} fold{sp['fold']}: n={len(yt)} ({int(yt.sum())} drivers) | " +
              " | ".join(f"{k.split(' (')[0][:8]} {roc_auc_score(yt, v):.3f}" if 0 < yt.sum() < len(yt) else f"{k[:8]} n/a" for k, v in sc.items()))
    df = pd.DataFrame(rows)
    df.to_csv(out / "transfer_per_fold.csv", index=False)

    summary = {"external_cohort": name, "n_shared_genes": len(common)}
    for k, lst in pooled.items():
        y = np.concatenate([t[1] for t in lst if t[0] == 0])
        p = np.concatenate([t[2] for t in lst if t[0] == 0])
        if 0 < y.sum() < len(y):
            lo, hi = boot_ci(y, p, roc_auc_score)
            lo2, hi2 = boot_ci(y, p, average_precision_score)
            summary[k] = dict(n=int(len(y)), n_pos=int(y.sum()), pooled_AUROC=float(roc_auc_score(y, p)), AUROC_CI95=[lo, hi],
                              pooled_AUPRC=float(average_precision_score(y, p)), AUPRC_CI95=[lo2, hi2], prevalence=float(y.mean()))
    # ---- ranking concordance of unlabelled proteins (needs predictions from both cohorts)
    fa, fb = pa.res / "predictions_all_nodes.csv", pb.res / "predictions_all_nodes.csv"
    if fa.exists() and fb.exists():
        A = pd.read_csv(fa).set_index("gene"); B = pd.read_csv(fb).set_index("gene")
        g = [x for x in A.index if x in B.index and A.label[x] != 1 and B.label[x] != 1]
        rho, p = stats.spearmanr(A.loc[g, "prob_ensemble"], B.loc[g, "prob_ensemble"])
        topA = set(A.loc[g].sort_values("prob_ensemble", ascending=False).head(a.top_n).index)
        topB = set(B.loc[g].sort_values("prob_ensemble", ascending=False).head(a.top_n).index)
        summary["ranking_concordance"] = dict(n_genes=len(g), spearman_rho=float(rho), p=float(p), top_n=a.top_n,
                                              top_n_overlap=len(topA & topB), expected_overlap_by_chance=a.top_n ** 2 / len(g))
    else:
        print("[external] run stage 2 on the external root as well to obtain the cross-cohort ranking concordance")
    (out / "transfer_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Stage 5 finished -> {out}")


if __name__ == "__main__":
    main()
