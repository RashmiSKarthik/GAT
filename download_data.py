#!/usr/bin/env python
"""
Download the public inputs of the pipeline into <root>/raw/ and write a provenance manifest (URL, size, SHA-256, date).

    python scripts/download_data.py --list                       # print every source and its access requirements
    python scripts/download_data.py --only string gwas go        # open resources, no login
    export SYNAPSE_AUTH_TOKEN=...                                # personal access token from synapse.org
    python scripts/download_data.py --only synapse               # ROS/MAP clinical + TMT matrix (controlled access)
    python scripts/download_data.py --only synapse --synapse-extra syn12345678   # any extra Synapse entity (e.g. metadata, MSBB)

ROS/MAP and MSBB are controlled-access AMP-AD resources: you need a Synapse account and an approved data-use request
(https://adknowledgeportal.synapse.org/Data%20Access). This script cannot bypass that step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

STRING_BASE = ["https://stringdb-downloads.org/download", "https://string-db.org/download"]
STRING_FILES = {"string_links": "protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
                "string_info": "protein.info.v12.0/9606.protein.info.v12.0.txt.gz"}
GWAS_INDEX = "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/"
GWAS_PATTERN = r'href="(gwas-catalog-associations_ontology-annotated\.(?:tsv|zip))"'
GO_CURRENT = {"go_obo": "https://purl.obolibrary.org/obo/go/go-basic.obo",
              "go_gaf": "https://current.geneontology.org/annotations/goa_human.gaf.gz"}
GO_ARCHIVE = {"go_obo": "https://release.geneontology.org/{rel}/ontology/go-basic.obo",
              "go_gaf": "https://release.geneontology.org/{rel}/annotations/goa_human.gaf.gz"}
SYNAPSE = {"rosmap_clinical": "syn3191087",          # ROSMAP_clinical.csv
           "rosmap_tmt_matrix": "syn21266454"}        # ROS/MAP DLPFC TMT log2 abundance (median-polish batch-corrected)

LIST = """\
Source                         Where                                                                                 Access
ROS/MAP study page             https://adknowledgeportal.synapse.org/Explore/Studies/DetailsPage/StudyDetails?Study=syn3219045   controlled (Synapse + DUA)
ROS/MAP clinical               https://www.synapse.org/Synapse:syn3191087                                            controlled
ROS/MAP TMT matrix             https://www.synapse.org/Synapse:syn21266454                                           controlled
MSBB study page                https://adknowledgeportal.synapse.org/Explore/Studies/DetailsPage/StudyDetails?Study=syn3159438  controlled
STRING v12.0 (9606)            https://version-12-0.string-db.org/cgi/download                                       open
GWAS Catalog                   https://www.ebi.ac.uk/gwas/docs/file-downloads                                        open
Gene Ontology                  https://geneontology.org/docs/downloads/                                              open
Alzforum Mutations (APP/PSEN1/PSEN2 only, no download needed)   https://www.alzforum.org/mutations                  open
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)
    return {"name": dest.name, "url": url, "path": str(dest), "bytes": dest.stat().st_size, "sha256": sha256(dest),
            "downloaded": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}


def fetch_first(urls: list[str], dest: Path) -> dict:
    err = None
    for u in urls:
        try:
            return fetch(u, dest)
        except Exception as e:  # noqa: BLE001
            err, _ = e, print(f"  failed: {e!r}")
    raise SystemExit(f"all mirrors failed for {dest.name}: {err!r}")


def get_string(raw: Path) -> list[dict]:
    return [fetch_first([f"{b}/{rel}" for b in STRING_BASE], raw / Path(rel).name) for rel in STRING_FILES.values()]


def get_gwas(raw: Path) -> list[dict]:
    html = urllib.request.urlopen(GWAS_INDEX, timeout=60).read().decode("utf-8", "ignore")
    m = re.search(GWAS_PATTERN, html)
    if not m:
        raise SystemExit(f"could not find the ontology-annotated associations file at {GWAS_INDEX}; "
                         "download it manually from https://www.ebi.ac.uk/gwas/docs/file-downloads")
    rec = fetch(GWAS_INDEX + m.group(1), raw / m.group(1))
    if m.group(1).endswith(".zip"):                          # pandas can read a single-file zip directly; unpack for clarity
        with zipfile.ZipFile(raw / m.group(1)) as z:
            z.extractall(raw)
    rec["gwas_catalog_release_dir"] = GWAS_INDEX
    return [rec]


def get_go(raw: Path, release: str | None) -> list[dict]:
    urls = {k: (v.format(rel=release) if release else v) for k, v in (GO_ARCHIVE if release else GO_CURRENT).items()}
    recs = [fetch(urls["go_obo"], raw / "go-basic.obo"), fetch(urls["go_gaf"], raw / "goa_human.gaf.gz")]
    head = (raw / "go-basic.obo").read_text(errors="ignore")[:2000]
    m = re.search(r"data-version:\s*(\S+)", head)
    recs[0]["go_data_version"] = m.group(1) if m else "unknown"
    print(f"  GO ontology data-version: {recs[0]['go_data_version']}  (cite this / pass --go-release to pin it)")
    return recs


def get_synapse(raw: Path, extra: list[str]) -> list[dict]:
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        raise SystemExit("Set SYNAPSE_AUTH_TOKEN (synapse.org -> Account Settings -> Personal Access Tokens, scopes: view, download).")
    import synapseclient
    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)
    recs = []
    for name, sid in list(SYNAPSE.items()) + [(f"extra_{s}", s) for s in extra]:
        print(f"  Synapse {sid} ({name})")
        ent = syn.get(sid, downloadLocation=str(raw), ifcollision="overwrite.local")
        p = Path(ent.path)
        recs.append({"name": p.name, "synapse_id": sid, "version": getattr(ent, "versionNumber", None),
                     "modified": str(getattr(ent, "modifiedOn", "")), "path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p),
                     "downloaded": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--only", nargs="+", choices=["string", "gwas", "go", "synapse"], default=["string", "gwas", "go"])
    ap.add_argument("--go-release", default=None, help="pin a Gene Ontology release, e.g. 2024-01-01")
    ap.add_argument("--synapse-extra", nargs="*", default=[], help="additional Synapse IDs (sample metadata, MSBB, ...)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(LIST)
        return
    raw = Path(a.root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    recs: list[dict] = []
    for part in a.only:
        print(f"[{part}]")
        recs += {"string": lambda: get_string(raw), "gwas": lambda: get_gwas(raw), "go": lambda: get_go(raw, a.go_release),
                 "synapse": lambda: get_synapse(raw, a.synapse_extra)}[part]()
    man = raw / "DATA_MANIFEST.json"
    old = json.loads(man.read_text()) if man.exists() else []
    man.write_text(json.dumps(old + recs, indent=2))
    print(f"manifest -> {man}")
    ref = Path(__file__).resolve().parent.parent / "data" / "reference" / "positive_genes.txt"
    if not (raw / "positive_genes.txt").exists() and ref.exists():
        print(f"note: the 57 positive genes ship with the repository ({ref}); stage 1 finds them automatically")


if __name__ == "__main__":
    main()
