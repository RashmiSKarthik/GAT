#!/usr/bin/env python
"""
Stage 4 — robustness analyses requested during peer review. Run after stages 1-2 (needs results/cv_fold_metrics.csv for the
paired comparisons with the main GAT).

  --analysis trees           Random forest / gradient boosting (/XGBoost, LightGBM if installed) on the 5 tabular features
  --analysis degree_matched  negatives matched to the positives' degree distribution (+ degree-only baseline on those labels)
  --analysis permutation     label-permutation null for the GAT's AUROC / AUPRC
  --analysis ratios          positive:negative ratios 1:2, 1:5, 1:10 (random negatives, several replicates)
  --analysis pu              non-negative PU learning (nnPU): the 200 'bystanders' are NOT used as training negatives
  --analysis leakfree        inductive ablation: topology recomputed with test nodes/edges masked, scaler fit on training nodes
  --analysis all             everything above

Outputs: <root>/results/reviewer/*.csv  (+ reviewer_summary.json)
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import common as C
import stage1_preprocessing as S1
import stage2_train as S2

DEV = S2.DEV
METRICS = ["AUROC", "AUPRC"]


# --------------------------------------------------------------------------- helpers
def cv_args(a, repeats=None) -> Namespace:
    return Namespace(epochs=a.epochs, patience=a.patience, monitor=a.monitor, repeats=repeats or a.repeats)


def with_labels(data: Data, info: pd.DataFrame, label: np.ndarray):
    d = data.clone()
    d.y = torch.as_tensor(label, dtype=torch.float32, device=DEV)
    i = info.copy()
    i["label"] = label
    return d, i


def summarise(df: pd.DataFrame, by=("model",), cols=METRICS) -> pd.DataFrame:
    g = df.groupby(list(by))[list(cols)]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).reset_index()
    return out


def paired_vs_gat(df: pd.DataFrame, gat: pd.DataFrame, model: str) -> dict:
    """GAT minus `model`, paired on (repeat, fold); corrected resampled t-test."""
    a = gat.set_index(["repeat", "fold"]).sort_index()
    b = df[df.model == model].set_index(["repeat", "fold"]).sort_index()
    idx = a.index.intersection(b.index)
    n_trv, n_te = float(a.n_trv.mean()), float(a.n_te.mean())
    out = {}
    for m in METRICS:
        t = S2.corrected_resampled_ttest(a.loc[idx, m].values - b.loc[idx, m].values, n_trv, n_te)
        out.update({f"d{m}": t["mean_diff"], f"d{m}_ci_low": t["ci_low"], f"d{m}_ci_high": t["ci_high"], f"p_{m}": t["p"]})
    out["k"] = len(idx)
    return out


def load_main_gat(res: Path, repeats: int) -> pd.DataFrame | None:
    f = res / "cv_fold_metrics.csv"
    if not f.exists():
        print("[warn] results/cv_fold_metrics.csv not found: paired comparisons with the main GAT are skipped (run stage 2 first)")
        return None
    cv = pd.read_csv(f)
    g = cv[(cv.model == "GAT") & (cv.repeat < repeats)]
    if g.repeat.nunique() < repeats:
        print(f"[warn] stage 2 was run with fewer than {repeats} repeats; paired comparison uses the available ones")
    return g


# --------------------------------------------------------------------------- A. tree-based baselines
def run_trees(data, info, a, out, gat):
    X = info[[f + "_z" for f in C.FEATURES]].values
    y = info.label.values
    models = {
        "RandomForest": lambda s: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                                         n_jobs=-1, random_state=s),
        "HistGradientBoosting": lambda s: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                                         class_weight="balanced", random_state=s),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = lambda s: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                                    colsample_bytree=0.8, random_state=s, verbosity=0, n_jobs=-1)
    except ImportError:
        print("[trees] xgboost not installed -> skipped (pip install xgboost)")
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = lambda s: LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                                      colsample_bytree=0.8, class_weight="balanced", random_state=s, verbose=-1)
    except ImportError:
        print("[trees] lightgbm not installed -> skipped (pip install lightgbm)")
    rows = []
    for sp in S2.make_splits(info, a.repeats):
        for name, mk in models.items():
            clf = mk(C.SEED_NUMPY + sp["k"]).fit(X[sp["trv"]], y[sp["trv"]].astype(int))
            p = clf.predict_proba(X[sp["te"]])[:, 1]
            yt = y[sp["te"]].astype(int)
            rows.append(dict(repeat=sp["rep"], fold=sp["fold"], model=name, AUROC=roc_auc_score(yt, p),
                             AUPRC=average_precision_score(yt, p)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "trees_per_fold.csv", index=False)
    summ = summarise(df)
    if gat is not None:
        cmp_ = pd.DataFrame([dict(model=m, **paired_vs_gat(df, gat, m)) for m in models])
        summ = summ.merge(cmp_, on="model")
        g = gat[METRICS]
        print(f"[trees] reference GAT: AUROC {g.AUROC.mean():.3f} ± {g.AUROC.std(ddof=1):.3f}, AUPRC {g.AUPRC.mean():.3f} ± {g.AUPRC.std(ddof=1):.3f}")
    summ.to_csv(out / "trees_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    return summ


# --------------------------------------------------------------------------- B. degree-matched negatives + permutation null
def degree_matched_negatives(info, positives, pool, n_neg, rng, window=25):
    """Greedy nearest-neighbour matching on log(1 + degree), without replacement, so that negatives reproduce the positives'
    degree distribution. Each positive receives n_neg/n_pos negatives (the remainder is spread randomly)."""
    deg = np.log1p(info.set_index("gene")["degree"])
    pool = [g for g in pool if g not in set(positives)]
    avail = pd.Series(deg.loc[pool].values, index=pool)
    per = np.full(len(positives), n_neg // len(positives))
    per[rng.choice(len(positives), n_neg - per.sum(), replace=False)] += 1
    chosen = []
    for pg, m in zip(rng.permutation(positives), rng.permutation(per)):
        cand = (avail - deg.loc[pg]).abs().sort_values(kind="stable").index[: max(window, 3 * int(m))]
        pick = list(rng.choice(cand, size=int(m), replace=False))
        chosen += pick
        avail = avail.drop(pick)
    return chosen


def balance_report(info, pos, neg):
    d = info.set_index("gene")["degree"]
    smd = (np.log1p(d.loc[pos]).mean() - np.log1p(d.loc[neg]).mean()) / np.sqrt((np.log1p(d.loc[pos]).var() + np.log1p(d.loc[neg]).var()) / 2)
    return dict(pos_mean_degree=float(d.loc[pos].mean()), neg_mean_degree=float(d.loc[neg].mean()), smd_log_degree=float(smd),
                ks_p=float(stats.ks_2samp(np.log1p(d.loc[pos]), np.log1p(d.loc[neg]))[1]))


def run_degree_matched(data, info, a, out, gat, pool):
    pos = info.gene[info.label == 1].tolist()
    n_neg = int((info.label == 0).sum())
    rows, bal = [], []
    for r in range(a.replicates):
        rng = np.random.default_rng(C.SEED_NUMPY + 100 + r)
        neg = degree_matched_negatives(info, pos, pool, n_neg, rng)
        bal.append(dict(replicate=r, **balance_report(info, pos, neg)))
        lab = np.full(len(info), -1.0); lab[info.gene.isin(pos).values] = 1; lab[info.gene.isin(neg).values] = 0
        d2, i2 = with_labels(data, info, lab)
        torch.manual_seed(C.SEED_TORCH + r)
        df, *_ = S2.run_cv(d2, i2, cv_args(a, 1), baselines=True, verbose=False, gcn=False)
        df["replicate"] = r
        rows.append(df)
        g = df[df.model == "GAT"]
        print(f"[degree-matched] replicate {r}: GAT AUROC {g.AUROC.mean():.3f}  AUPRC {g.AUPRC.mean():.3f} | "
              f"degree-only AUROC {df[df.model=='Degree only'].AUROC.mean():.3f} | SMD(log degree) {bal[-1]['smd_log_degree']:.2f}")
    df = pd.concat(rows)
    df.to_csv(out / "degree_matched_per_fold.csv", index=False)
    pd.DataFrame(bal).to_csv(out / "degree_matched_balance.csv", index=False)
    per_rep = df.groupby(["replicate", "model"])[METRICS].mean().reset_index()
    summ = per_rep.groupby("model")[METRICS].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "degree_matched_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    return summ


def run_permutation(data, info, a, out, gat):
    obs = gat.groupby("repeat")[METRICS].mean().mean() if gat is not None else None
    lab_idx = np.where(info.label.values >= 0)[0]
    rows = []
    for b in range(a.n_perm):
        rng = np.random.default_rng(C.SEED_NUMPY + 1000 + b)
        lab = info.label.values.copy()
        lab[lab_idx] = rng.permutation(lab[lab_idx])           # same class counts, labels detached from the nodes
        d2, i2 = with_labels(data, info, lab)
        torch.manual_seed(C.SEED_TORCH + b)
        df, *_ = S2.run_cv(d2, i2, cv_args(a, 1), baselines=False, verbose=False)
        g = df[df.model == "GAT"]
        rows.append(dict(perm=b, AUROC=g.AUROC.mean(), AUPRC=g.AUPRC.mean()))
        if (b + 1) % 5 == 0:
            print(f"[permutation] {b+1}/{a.n_perm}")
    null = pd.DataFrame(rows)
    null.to_csv(out / "permutation_null.csv", index=False)
    res = {"null_AUROC_mean": float(null.AUROC.mean()), "null_AUROC_sd": float(null.AUROC.std(ddof=1)),
           "null_AUPRC_mean": float(null.AUPRC.mean())}
    if obs is not None:
        for m in METRICS:
            res[f"observed_{m}"] = float(obs[m])
            res[f"p_perm_{m}"] = float((1 + (null[m] >= obs[m]).sum()) / (1 + len(null)))
    print("[permutation]", json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------- C. positive:negative ratios
def run_ratios(data, info, a, out, gat, pool):
    pos = info.gene[info.label == 1].tolist()
    rows = []
    for ratio in a.ratios:
        n_neg = len(pos) * ratio
        for r in range(a.replicates):
            random.seed(C.SEED_PYTHON + 31 * ratio + r)
            neg = random.sample(sorted(set(pool) - set(pos)), n_neg)
            lab = np.full(len(info), -1.0); lab[info.gene.isin(pos).values] = 1; lab[info.gene.isin(neg).values] = 0
            d2, i2 = with_labels(data, info, lab)
            torch.manual_seed(C.SEED_TORCH + r)
            df, *_ = S2.run_cv(d2, i2, cv_args(a, 1), baselines=True, verbose=False, gcn=False)
            prev = len(pos) / (len(pos) + n_neg)
            for m, g in df.groupby("model"):
                rows.append(dict(ratio=f"1:{ratio}", replicate=r, model=m, prevalence=prev, AUROC=g.AUROC.mean(),
                                 AUPRC=g.AUPRC.mean(), AUPRC_lift=g.AUPRC.mean() / prev))
        sub = pd.DataFrame(rows)
        s = sub[(sub.ratio == f"1:{ratio}") & (sub.model == "GAT")]
        print(f"[ratios] 1:{ratio}  GAT AUROC {s.AUROC.mean():.3f} ± {s.AUROC.std(ddof=1):.3f}  AUPRC {s.AUPRC.mean():.3f}  "
              f"(prevalence {s.prevalence.iloc[0]:.3f}, lift {s.AUPRC_lift.mean():.1f}x)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "ratios_per_replicate.csv", index=False)
    summ = df.groupby(["ratio", "model"])[["prevalence", "AUROC", "AUPRC", "AUPRC_lift"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(out / "ratios_summary.csv", index=False)
    return summ


# --------------------------------------------------------------------------- D. PU learning (nnPU)
def nnpu_loss(out_p, out_u, prior):
    """Non-negative PU risk (Kiryo et al., NeurIPS 2017) with the sigmoid loss. Returns (objective, risk)."""
    l = lambda z, y: torch.sigmoid(-y * z)
    rp_pos = l(out_p, 1.0).mean()
    rp_neg = l(out_p, -1.0).mean()
    ru_neg = l(out_u, -1.0).mean()
    neg_risk = ru_neg - prior * rp_neg
    risk = prior * rp_pos + torch.clamp(neg_risk, min=0.0)
    obj = prior * rp_pos + neg_risk if neg_risk >= 0 else -neg_risk          # gradient ascent step when the estimate goes negative
    return obj, risk


def train_nnpu(model, data, pos_tr, unl_tr, val_pos, val_unl, prior, epochs, lr, wd, patience):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pt, ut = torch.as_tensor(pos_tr, device=DEV), torch.as_tensor(unl_tr, device=DEV)
    best, state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        z = model(data.x, data.edge_index, data.edge_attr)
        obj, _ = nnpu_loss(z[pt], z[ut], prior)
        obj.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr)).cpu().numpy()
        yv = np.r_[np.ones(len(val_pos)), np.zeros(len(val_unl))]
        sc = roc_auc_score(yv, np.r_[p[val_pos], p[val_unl]])      # PU-AUC: validation positives vs unlabelled nodes
        if sc > best + 1e-6:
            best, state, bad = sc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if state is not None:
        model.load_state_dict(state)
    return model


def run_pu(data, info, a, out, gat):
    y = info.label.values
    rng = np.random.default_rng(C.SEED_NUMPY)
    rows = []
    for prior in a.pu_priors:
        for sp in S2.make_splits(info, a.repeats):
            tr, va, te = sp["tr"], sp["va"], sp["te"]
            held = set(te) | set(va)
            pos_tr = tr[y[tr] == 1]
            unl_tr = np.array([i for i in range(len(info)) if i not in held and i not in set(pos_tr)])   # incl. unlabelled + train negatives
            val_pos = va[y[va] == 1]
            val_unl = rng.choice(unl_tr, size=min(500, len(unl_tr)), replace=False)
            torch.manual_seed(C.SEED_TORCH + sp["k"])
            m = train_nnpu(S2.make_gat({}), data, pos_tr, unl_tr, val_pos, val_unl, prior, a.epochs, C.LR, C.WEIGHT_DECAY, a.patience)
            p = S2.predict(m, data)
            yt = y[te].astype(int)
            pu_auc = roc_auc_score(np.r_[np.ones((yt == 1).sum()), np.zeros(500)],
                                   np.r_[p[te][yt == 1], p[rng.choice(unl_tr, 500, replace=False)]])
            rows.append(dict(prior=prior, repeat=sp["rep"], fold=sp["fold"], AUROC=roc_auc_score(yt, p[te]),
                             AUPRC=average_precision_score(yt, p[te]), PU_AUROC=pu_auc))
        g = pd.DataFrame(rows)
        g = g[g.prior == prior]
        print(f"[pu] prior={prior}: AUROC (test positives vs held-out bystanders) {g.AUROC.mean():.3f} ± {g.AUROC.std(ddof=1):.3f}  "
              f"AUPRC {g.AUPRC.mean():.3f}  PU-AUROC {g.PU_AUROC.mean():.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "pu_per_fold.csv", index=False)
    summ = df.groupby("prior")[["AUROC", "AUPRC", "PU_AUROC"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(out / "pu_summary.csv", index=False)
    return summ


# --------------------------------------------------------------------------- E. leak-free (inductive) ablation
def _scale(X, fit_idx):
    mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype(np.float32)


def run_leakfree(data, info, a, out, gat, proc):
    """Three settings per fold (same splits / seeds as stage 2):
        transductive               features from the full graph, z-score fitted on all nodes   (the main analysis)
        transductive + train scaler   same features, z-score fitted on the training labelled nodes only
        inductive                  topology of every non-test node recomputed on the graph WITHOUT the test nodes (and their edges),
                                   the GAT is trained on that masked graph; test nodes receive their full-graph features and
                                   the full graph is used only at prediction time. z-score fitted on training nodes.
    Fold-change is a node-level statistic of the expression matrix and does not depend on the graph."""
    if data.edge_attr is not None:
        raise SystemExit("leakfree analysis is implemented for --edge-attr none")
    nodes = info.gene.tolist()
    idx = {g: i for i, g in enumerate(nodes)}
    edges = pd.read_csv(proc / "edges_differential.csv.gz")
    src, tgt = edges.source.map(idx).values, edges.target.map(idx).values
    k = a.betweenness_k
    print(f"[leakfree] recomputing full-graph topology (betweenness pivots: {k or 'exact'}) ...")
    topo = ["degree", "pagerank", "betweenness", "clustering"]
    full = S1.topology_features(nodes, edges, True, k, verbose=False)
    fc = info.fold_change.values[:, None]
    X_full = np.hstack([full[topo].values, fc]).astype(np.float64)
    y_t = data.y
    rows = []
    for sp in S2.make_splits(info, a.repeats):
        tr, va, te = sp["tr"], sp["va"], sp["te"]
        yte = info.label.values[te].astype(int)
        masked = np.isin(src, te) | np.isin(tgt, te)
        topo_tr = S1.topology_features(nodes, edges[~masked], True, k, verbose=False)
        X_tr = np.hstack([topo_tr[topo].values, fc]).astype(np.float64)
        X_ind = X_tr.copy()
        X_ind[te] = X_full[te]                                   # held-out nodes: features from the full graph
        ei_tr = to_undirected(torch.as_tensor(np.vstack([src[~masked], tgt[~masked]]), dtype=torch.long), num_nodes=len(nodes)).to(DEV)
        variants = {
            "transductive": (Data(x=torch.as_tensor(_scale(X_full, np.arange(len(nodes)))), edge_index=data.edge_index, y=y_t).to(DEV), None),
            "transductive + train-fit scaler": (Data(x=torch.as_tensor(_scale(X_full, tr)), edge_index=data.edge_index, y=y_t).to(DEV), None),
            "inductive (test nodes/edges masked)": (
                Data(x=torch.as_tensor(_scale(X_tr, tr)), edge_index=ei_tr, y=y_t).to(DEV),
                Data(x=torch.as_tensor(_scale(X_ind, tr)), edge_index=data.edge_index, y=y_t).to(DEV)),
        }
        for name, (d_train, d_eval) in variants.items():
            torch.manual_seed(C.SEED_TORCH + sp["k"])
            m, _ = S2.train_model(S2.make_gat({}), d_train, tr, va, epochs=a.epochs, patience=a.patience, monitor=a.monitor)
            p = S2.predict(m, d_eval or d_train)
            rows.append(dict(repeat=sp["rep"], fold=sp["fold"], model=name, n_trv=len(sp["trv"]), n_te=len(te),
                             AUROC=roc_auc_score(yte, p[te]), AUPRC=average_precision_score(yte, p[te])))
        print(f"[leakfree] rep{sp['rep']} fold{sp['fold']}: " + " | ".join(f"{r['model'][:12]} {r['AUROC']:.3f}" for r in rows[-3:]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "leakfree_per_fold.csv", index=False)
    summ = summarise(df)
    ref = df[df.model == "transductive"]
    extra = []
    for m in ("transductive + train-fit scaler", "inductive (test nodes/edges masked)"):
        t = paired_vs_gat(df.assign(model=df.model), ref, m)
        extra.append(dict(model=m, **t))
    summ = summ.merge(pd.DataFrame(extra), on="model", how="left")
    summ.to_csv(out / "leakfree_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    return summ


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--analysis", nargs="+", default=["all"],
                    choices=["all", "trees", "degree_matched", "permutation", "ratios", "pu", "leakfree"])
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--monitor", choices=["auroc", "loss"], default="auroc")
    ap.add_argument("--repeats", type=int, default=1, help="CV repeats for trees / pu / leakfree (use the same value as stage 2)")
    ap.add_argument("--replicates", type=int, default=5, help="independent negative-set draws (degree_matched, ratios)")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--ratios", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--pu-priors", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    ap.add_argument("--betweenness-k", type=int, default=500, help="betweenness pivots in the leak-free ablation (None = exact)")
    ap.add_argument("--edge-attr", choices=["none", "string", "abs_r"], default="none")
    a = ap.parse_args()
    todo = ["trees", "degree_matched", "permutation", "ratios", "pu", "leakfree"] if "all" in a.analysis else a.analysis

    paths = C.get_paths(a.root)
    out = paths.res / "reviewer"
    out.mkdir(exist_ok=True)
    C.set_seeds()
    warnings.filterwarnings("ignore", category=FutureWarning)
    data, info = S2.load_graph(paths.proc, a.edge_attr)
    pool = [g for g in (paths.proc / "negative_pool.txt").read_text().split() if g]
    gat = load_main_gat(paths.res, a.repeats)
    summary = {}
    for name in todo:
        print(f"\n===== {name} =====")
        if name == "trees":
            run_trees(data, info, a, out, gat)
        elif name == "degree_matched":
            run_degree_matched(data, info, a, out, gat, pool)
        elif name == "permutation":
            summary["permutation"] = run_permutation(data, info, a, out, gat)
        elif name == "ratios":
            run_ratios(data, info, a, out, gat, pool)
        elif name == "pu":
            run_pu(data, info, a, out, gat)
        elif name == "leakfree":
            run_leakfree(data, info, a, out, gat, paths.proc)
    (out / "reviewer_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nStage 4 finished -> {out}")


if __name__ == "__main__":
    main()
