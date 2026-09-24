#!/usr/bin/env python3
"""Independent final QA for required root-scaling artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


OUT = Path("/home/exouser/FORTE/root_scaling_20260831")
REQUIRED = ["ROOT_SCALING_TEST_MANIFEST.json", "ROOT_SCALING_TRAIN_MANIFEST.json",
    "ROOT_SCALING_COLLECTION_QA.md", "ROOT_SCALING_DATA_COUNTS.csv", "ROOT_SCALING_MODEL_HASHES.csv",
    "ROOT_SCALING_LEARNING_CURVE.csv", "ROOT_SCALING_PER_ROOT_METRICS.csv",
    "ROOT_SCALING_SAFETY_METRICS.csv", "ROOT_SCALING_TRAIN_TEST_GAP.csv",
    "ROOT_SCALING_FINAL_REPORT.md", "ROOT_SCALING_FINAL_CLASSIFICATION.json", "SHA256SUMS.txt"]
ALLOWED = {"VISUAL_CONTEXT_IS_INDEPENDENT_ROOT_SAMPLE_LIMITED", "JOINT_VALUE_EMERGES_WITH_CONTEXT_SCALE",
    "VISUAL_IMPROVES_BUT_JOINT_NOT_NEEDED", "JOINT_FRONTIER_SIGNAL_WITHOUT_SAFE_CONTROL",
    "GENERIC_VISUAL_CONTEXT_REMAINS_INEFFICIENT_WITH_MORE_ROOTS",
    "INSUFFICIENT_ROOT_SCALE_TO_DISTINGUISH_SAMPLE_COMPLEXITY_FROM_REPRESENTATION"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def main() -> None:
    failures = []
    for name in REQUIRED:
        if not (OUT / name).exists(): failures.append(f"missing:{name}")
    if failures: raise RuntimeError(failures)
    test = json.loads((OUT / "ROOT_SCALING_TEST_MANIFEST.json").read_text())
    train = json.loads((OUT / "ROOT_SCALING_TRAIN_MANIFEST.json").read_text())
    test_roots = {t: {x["root_id"] for x in rows} for t, rows in test["contexts"].items()}
    for t in ("0", "5"):
        if len(test_roots[t]) != 10: failures.append(f"task{t}:TEST roots !=10")
        for n in (6, 15, 30, 50):
            q = train["nested_sets"][f"S{n}"][t]
            if len(q["root_ids"]) != n or q["independent_root_count"] != n: failures.append(f"task{t}:S{n} root count")
            if set(q["root_ids"]) & test_roots[t]: failures.append(f"task{t}:S{n} TRAIN/TEST overlap")
    models = pd.read_csv(OUT / "ROOT_SCALING_MODEL_HASHES.csv")
    lc = pd.read_csv(OUT / "ROOT_SCALING_LEARNING_CURVE.csv")
    pr = pd.read_csv(OUT / "ROOT_SCALING_PER_ROOT_METRICS.csv")
    sf = pd.read_csv(OUT / "ROOT_SCALING_SAFETY_METRICS.csv")
    gp = pd.read_csv(OUT / "ROOT_SCALING_TRAIN_TEST_GAP.csv")
    if len(models) != 72 or set(models.model) != {"Base","Full Visual","Visual Joint"}: failures.append("model hash table")
    for r in models.itertuples(index=False):
        if sha256(Path(r.checkpoint)) != r.checkpoint_sha256: failures.append(f"checkpoint hash:{r.checkpoint}")
        if sha256(Path(r.PCA_path)) != r.PCA_sha256: failures.append(f"PCA hash:{r.PCA_path}")
        if sha256(Path(r.normalization_path)) != r.normalization_sha256: failures.append(f"norm hash:{r.normalization_path}")
    if len(lc) != 144 or len(sf) != 144 or len(gp) != 144: failures.append("aggregate row counts")
    if len(pr) != 960: failures.append(f"per-root row count={len(pr)}")
    if set(lc.N_root) != {6,15,30,50} or set(lc.task) != {0,5}: failures.append("learning-curve axes")
    if set(lc.aggregation) != {"ENSEMBLE","SEED_0","SEED_1","SEED_2","SEED_MEAN","SEED_STD"}: failures.append("seed aggregations")
    final = json.loads((OUT / "ROOT_SCALING_FINAL_CLASSIFICATION.json").read_text())
    if final.get("classification") not in ALLOWED: failures.append("classification vocabulary")
    for task in (0,5):
        p = OUT / f"ROOT_SCALING_LEARNING_CURVES_TASK{task}.png"
        try:
            im = Image.open(p); im.verify()
            if im.width < 1200 or im.height < 800: failures.append(f"task{task}:figure size")
        except Exception as exc: failures.append(f"task{task}:figure:{exc}")
    expected = {}
    for line in (OUT / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1); expected[name] = h
    for name, h in expected.items():
        if sha256(OUT / name) != h: failures.append(f"delivery hash:{name}")
    report = (OUT / "ROOT_SCALING_FINAL_REPORT.md").read_text()
    for question in ("Was failure at six roots", "Does Full Visual exceed Base", "Does Joint recover",
                     "If Joint improves frontier", "Does the TRAIN→TEST gap", "Next action"):
        if question not in report: failures.append(f"report answer missing:{question}")
    result = {"status": "PASS" if not failures else "FAIL", "failures": failures,
              "counts": {"model_hash_rows": len(models), "learning_curve_rows": len(lc),
                         "per_root_rows": len(pr), "safety_rows": len(sf), "gap_rows": len(gp)},
              "classification": final.get("classification")}
    (OUT / "ROOT_SCALING_FINAL_VALIDATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
