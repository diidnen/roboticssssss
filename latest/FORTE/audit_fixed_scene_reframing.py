#!/usr/bin/env python3
"""Audit the fixed-scene hidden-friction reframing before any new training.

This script is intentionally read-only with respect to the source datasets. It
creates a versioned audit namespace, re-indexes the authoritative 720-row
continuous Joint-common population, and writes a friction-context split that
can be consumed by a later exact-method training wrapper.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FORTE = Path("/home/exouser/FORTE")
SOURCE = FORTE / "gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
TASK_OBJECTS = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    5: "tomato_sauce_1",
    6: "butter_1",
}
TRAJECTORY_FAMILY = "P5S0C_P4B_strict_preprobe_continuous"
TEST_TOKENS = {"5174", "5175", "5176", "5177", "5178", "5179"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(p: Path, obj: object) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def underlying_seed(root_id: str) -> str:
    m = re.search(r"_s(\d+)(?:$|_)", root_id)
    return m.group(1) if m else "UNKNOWN"


def scene_for(task: int) -> tuple[str, str]:
    obj = TASK_OBJECTS[task]
    return f"scene_task{task}_{obj}", f"task{task}:{obj}:{TRAJECTORY_FAMILY}"


def source_rows() -> pd.DataFrame:
    d = pd.read_csv(SOURCE)
    expected = {
        "branch_id", "context_id", "root_id", "task", "friction", "repeat",
        "requested_force_N", "valid", "full_task_success_y",
        "corrected_physical_telemetry_valid", "telemetry_path",
    }
    missing = sorted(expected - set(d.columns))
    if missing:
        raise RuntimeError(f"source missing columns: {missing}")
    if len(d) != 720 or int(d.valid.sum()) != 720 or int(d.corrected_physical_telemetry_valid.sum()) != 720:
        raise RuntimeError("authoritative source is not exactly 720 valid Joint-common rows")
    sealed_re = re.compile(r"(?:^|[^0-9])(?:5174|5175|5176|5177|5178|5179)(?:[^0-9]|$)")
    if any(sealed_re.search(str(x)) for x in d.root_id.astype(str)):
        raise RuntimeError("sealed TEST root token found in source root_id")
    if any(sealed_re.search(str(x)) for x in d.telemetry_path.astype(str)):
        raise RuntimeError("sealed TEST root token found in source telemetry path")
    return d


def reindex(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for r in d.sort_values(["task", "context_id", "requested_force_N", "repeat"]).itertuples(index=False):
        task = int(r.task)
        scene_id, _ = scene_for(task)
        rows.append({
            "scene_id": scene_id,
            "trajectory_family": TRAJECTORY_FAMILY,
            "underlying_seed": underlying_seed(str(r.root_id)),
            "root_id": str(r.root_id),
            "mu": float(r.friction),
            "force_N": float(r.requested_force_N),
            "repeat": int(r.repeat),
            "success": int(r.full_task_success_y),
            "probe_available": 1,
            "direct_usable": 1,
            "imagination_usable": 1,
            "joint_usable": 1,
            "source": str(SOURCE),
            "context_id": str(r.context_id),
            "branch_id": str(r.branch_id),
            "task": task,
            "friction_band": str(r.friction_band),
            "stratum_index": int(r.stratum_index),
            "valid": int(r.valid),
            "corrected_physical_telemetry_valid": int(r.corrected_physical_telemetry_valid),
        })
    out_path = out / "FIXED_SCENE_COMMON_DATASET_AUDIT.csv"
    write_csv(out_path, rows)
    return pd.DataFrame(rows)


def coverage(idx: pd.DataFrame, out: Path) -> list[dict]:
    rows = []
    for scene, g in idx.groupby("scene_id", sort=True):
        contexts = g.groupby("context_id", sort=True)
        mixed = []
        all_success = 0
        all_failure = 0
        for cid, c in contexts:
            ys = set(c.success.astype(int))
            if len(ys) > 1:
                mixed.append(cid)
            elif ys == {1}:
                all_success += 1
            elif ys == {0}:
                all_failure += 1
        force_values = sorted({round(float(x), 8) for x in g.force_N})
        mus = sorted({round(float(x), 8) for x in g.mu})
        rows.append({
            "scene_id": scene,
            "object": TASK_OBJECTS[int(g.task.iloc[0])],
            "trajectory_family": TRAJECTORY_FAMILY,
            "task": int(g.task.iloc[0]),
            "underlying_seed_count": int(g.underlying_seed.nunique()),
            "underlying_seeds": ";".join(sorted(g.underlying_seed.unique())),
            "stored_root_id_count": int(g.root_id.nunique()),
            "stored_root_ids": ";".join(sorted(g.root_id.unique())),
            "unique_mu_count": len(mus),
            "mu_min": min(mus), "mu_max": max(mus),
            "friction_mode": "continuous",
            "force_values_N": ";".join(f"{x:.8f}" for x in force_values),
            "force_min_N": min(force_values), "force_max_N": max(force_values),
            "unique_force_value_count": len(force_values),
            "labels": len(g), "successes": int(g.success.sum()), "failures": int((1-g.success).sum()),
            "context_cells": int(g.context_id.nunique()),
            "mixed_boundary_contexts": len(mixed),
            "mixed_boundary_context_ids": ";".join(mixed),
            "all_success_contexts": all_success, "all_failure_contexts": all_failure,
            "joint_complete_labels": int((g.joint_usable == 1).sum()),
            "note": "scene identity is task/object plus frozen P5-S0-C/P4-B trajectory family; root_id is nested variation, not scene identity",
        })
    write_csv(out / "SCENE_FRICTION_FORCE_COVERAGE.csv", rows)
    return rows


def split_manifests(idx: pd.DataFrame, out: Path) -> tuple[list[dict], list[dict]]:
    """Hold out one context every three sorted contexts within each scene.

    This gives six DEV friction contexts/scene, retains all three friction
    bands in both splits, and never splits rows from one context across TRAIN
    and DEV. It is an offline label split, not a new simulator rollout.
    """
    train, dev = [], []
    split_rows = []
    for scene, g in idx.groupby("scene_id", sort=True):
        context_meta = g.groupby("context_id", sort=True).agg(mu=("mu", "first"), underlying_seed=("underlying_seed", "first"), band=("friction_band", "first")).reset_index()
        context_meta = context_meta.sort_values(["mu", "underlying_seed", "context_id"]).reset_index(drop=True)
        held = set(context_meta.iloc[[2, 5, 8, 11, 14, 17]].context_id)
        for _, c in context_meta.iterrows():
            split = "DEV" if c.context_id in held else "TRAIN"
            split_rows.append({"scene_id": scene, "context_id": c.context_id, "split": split, "mu": float(c.mu), "underlying_seed": c.underlying_seed, "friction_band": c.band})
        for r in g.to_dict("records"):
            q = dict(r); q["split"] = "DEV" if q["context_id"] in held else "TRAIN"
            (dev if q["split"] == "DEV" else train).append(q)
    write_csv(out / "FIXED_SCENE_TRAIN_MANIFEST.csv", train)
    write_csv(out / "FIXED_SCENE_DEV_MANIFEST.csv", dev)
    write_csv(out / "FIXED_SCENE_CONTEXT_SPLIT.csv", split_rows)
    return train, dev


def learning_curve(idx: pd.DataFrame, out: Path) -> dict:
    subsets = {}
    for scene, g in idx.groupby("scene_id", sort=True):
        # Use only the fixed-scene TRAIN contexts. Positions 2,5,8,11,14,17
        # are the six complete friction contexts reserved for the immutable
        # DEV split; they must never enter a learning-curve subset.
        meta = g.groupby("context_id", sort=True).agg(mu=("mu", "first"), underlying_seed=("underlying_seed", "first")).reset_index()
        meta = meta.sort_values(["mu", "underlying_seed", "context_id"]).reset_index(drop=True)
        cs = [x for i, x in enumerate(meta.context_id) if i not in {2, 5, 8, 11, 14, 17}]
        subsets[scene] = {}
        for n in [3, 6, 9, len(cs)]:
            take = cs[:n]
            subsets[scene][f"{n}_contexts"] = take
    obj = {
        "definition": "context-level subsets within each scene family; no row-level context splitting",
        "requested_curve": ["25%", "50%", "75%", "100%"],
        "context_curve": [3, 6, 9, 12],
        "subsets": subsets,
        "note": "3/6/9/12 are 25/50/75/100% of the 12 TRAIN contexts per scene; the six held-out friction contexts per scene are excluded from every subset.",
    }
    write_json(out / "FIXED_SCENE_LEARNING_CURVE_SUBSETS.json", obj)
    return obj


def historical(out: Path) -> str:
    rows = [
        {"method":"Direct", "TRAIN":"not reported as matched full-task SR", "same_scene_heldout":"NOT ESTIMABLE", "cross_scene":"probability_MAE=0.2449; frontier_MAE=0.325N; under_force=0.25", "metric_scope":"old pooled continuous DEV", "source":"FORTE/gnp_style_continuous_20260830_125107/CONTINUOUS_JOINT_VS_FEAS.csv", "comparability":"diagnostic only", "notes":"FEASIBILITY_ONLY backend; pooled 72-context old benchmark; not a same-scene friction split"},
        {"method":"Imagination", "TRAIN":"trajectory/selection diagnostics, not matched full-task SR", "same_scene_heldout":"NOT ESTIMABLE", "cross_scene":"offline selection_accuracy=0.9722; under_force=0.0278", "metric_scope":"historical AFI offline decision audit", "source":"Tabero/analysis/results/active_friction_imagination_20260828_211106/OFFLINE_IMAGINATION_DECISION_AUDIT_SUMMARY.json", "comparability":"not comparable to Direct probability metrics", "notes":"AFI is historical ablation; this is not the proposed Direct selector"},
        {"method":"Joint", "TRAIN":"fit improves sharply; no matched full-task SR reported", "same_scene_heldout":"NOT ESTIMABLE", "cross_scene":"probability_MAE=0.2075; frontier_MAE=0.125N; under_force=0.375; nonmono=0.6667", "metric_scope":"old pooled continuous DEV", "source":"FORTE/gnp_style_continuous_20260830_125107/CONTINUOUS_JOINT_VS_FEAS.csv", "comparability":"diagnostic only", "notes":"Joint-NoVisual/physics auxiliary; frontier signal coexists with worse Brier/NLL/safety"},
    ]
    write_csv(out / "HISTORICAL_METHOD_GENERALIZATION_TABLE.csv", rows)
    report = """# Historical Joint Generalization Audit\n\n## Verdict\n\n`C — NO_CLEAR_SCENE_GENERALIZATION_PATTERN`\n\nThe archived evidence does not contain a clean same-scene held-out-friction benchmark for the historical Direct/Imagination/Joint comparison. Therefore the proposed hypothesis cannot be classified as A or B without relabeling a different split. The old pooled Joint result is a 72-context, four-task diagnostic; the root-heldout result is cross-root/scene-like rather than same-scene friction holding.\n\n## Evidence kept separate\n\n- Old pooled continuous Joint: probability MAE improves from 0.2449 to 0.2075 and frontier MAE from 0.325 N to 0.125 N, but Brier worsens 0.0859 to 0.1352, NLL worsens 0.4627 to 0.9387, under-force worsens 0.25 to 0.375, and nonmonotonic contexts are 0.6667.\n- Retrospective root-heldout Joint-NoVisual: probability MAE improves on all four tasks, while under-force worsens on all four and frontier worsens on tasks 0/1/5. This is a cross-root diagnostic, not same-scene friction generalization.\n- Historical Imagination reports selection/frontier diagnostics, not a matched full-task SR comparison; its metrics are not merged with Direct/Joint probability metrics.\n\n## Interpretation\n\nThe record supports a narrow historical frontier-localization signal and a failure to establish a reliable controller/generalization advantage. It does not establish “strong same-scene, weak cross-scene” or “overfits even within scene” because same-scene held-out friction was not measured under the same protocol.\n\nThe current study therefore proceeds under the explicitly narrowed question: `FIXED-SCENE / LOW-SCENE-DIVERSITY HIDDEN-FRICTION FORCE SELECTION`. The current 720-row common set is audited separately below; historical verdict C does not impose the obsolete 200-independent-root requirement.\n"""
    (out / "HISTORICAL_JOINT_GENERALIZATION_AUDIT.md").write_text(report, encoding="utf-8")
    return "NO_CLEAR_SCENE_GENERALIZATION_PATTERN"


def readiness(idx: pd.DataFrame, cov: list[dict], train: list[dict], dev: list[dict], curve: dict, out: Path, verdict: str) -> None:
    scene_ok = len(cov) >= 2
    context_ok = all(int(r["unique_mu_count"]) >= 10 for r in cov)
    # The frozen collector has task-specific supports: task1 is 4--6 N,
    # while task0/task5 are 3--5 N and task6 is 3--4 N.
    force_support = {0: (3.0, 5.0), 1: (4.0, 6.0), 5: (3.0, 5.0), 6: (3.0, 4.0)}
    force_ok = all(
        float(r["force_min_N"]) <= force_support[int(r["task"])][0] + 0.10
        and float(r["force_max_N"]) >= force_support[int(r["task"])][1] - 0.10
        for r in cov
    )
    transitions_ok = all(int(r["mixed_boundary_contexts"]) >= 1 for r in cov)
    joint_ok = int((idx.joint_usable == 1).sum()) == len(idx)
    split_ok = len(train) > 0 and len(dev) > 0 and not (set(x["context_id"] for x in train) & set(x["context_id"] for x in dev))
    ready = all([scene_ok, context_ok, force_ok, transitions_ok, joint_ok, split_ok])
    decision = "YES — TRAIN NOW" if ready else "NO — TARGETED COLLECTION ONLY"
    payload = {
        "status": "READY_TO_TRAIN_FIXED_SCENE" if ready else "TARGETED_COLLECTION_REQUIRED",
        "decision": decision,
        "historical_verdict": verdict,
        "scientific_scope": "FIXED-SCENE / LOW-SCENE-DIVERSITY HIDDEN-FRICTION FORCE SELECTION",
        "test_status": "TEST NOT OPENED",
        "criteria": {
            "scene_families_at_least_2": scene_ok,
            "at_least_10_friction_contexts_each": context_ok,
            "task_specific_frozen_force_support_coverage": force_ok,
            "success_failure_transition_each": transitions_ok,
            "complete_joint_inputs": joint_ok,
            "same_scene_heldout_friction_split_available": split_ok,
        },
        "counts": {"labels": len(idx), "scenes": len(cov), "train_labels": len(train), "dev_labels": len(dev), "contexts": int(idx.context_id.nunique())},
        "scene_identity": "task/object from p5s0c_paired_boundary_probe_value.py, plus frozen P5-S0-C/P4-B trajectory family; root IDs are nested state/seed variation",
        "split": "hold out six complete friction contexts per scene (sorted context positions 2,5,8,11,14,17), preserving all bands; offline labels only",
        "learning_curve": curve,
        "no_new_collection": True,
        "source_sha256": sha256(SOURCE),
    }
    write_json(out / "FIXED_SCENE_DATA_READINESS.json", payload)
    md = f"""# Fixed-Scene Data Readiness\n\nDecision: **{decision}**\n\nHistorical Joint verdict: `{verdict}`.\n\nThe authoritative common population contains {len(idx)} valid rows, {idx.scene_id.nunique()} scene families, {idx.context_id.nunique()} friction contexts, and {len(train)} TRAIN / {len(dev)} DEV labels under a context-level held-out-friction split. Every row is Joint-complete.\n\n## Scene definition\n\nA scene family is task/object plus the frozen P5-S0-C/P4-B trajectory protocol. The six underlying seeds within each task/object family are nested state/seed variation, not six independent scene identities.\n\n## Readiness criteria\n\n- scene families >=2: `{scene_ok}`\n- >=10 friction contexts per scene: `{context_ok}`\n- 3–5 N coverage per scene: `{force_ok}`\n- success/failure transition contexts: `{transitions_ok}`\n- complete Joint inputs: `{joint_ok}`\n- same-scene held-out friction split: `{split_ok}`\n\nThe result is **{decision}**. No 200-root requirement is applied. If training is started, use the exact frozen Direct/Imagination/Joint implementations and keep full-task success as the primary DEV metric; losses and ranking diagnostics remain secondary.\n\nTEST status: `TEST NOT OPENED`.\n"""
    md = md.replace("3–5 N coverage per scene: `{force_ok}`", "frozen task-specific force-support coverage (task0/5: 3–5 N; task1: 4–6 N; task6: 3–4 N): `{force_ok}`")
    (out / "FIXED_SCENE_DATA_READINESS.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out or FORTE / f"fixed_scene_reframing_{tag}"
    out.mkdir(parents=True, exist_ok=False)
    d = source_rows()
    idx = reindex(d, out)
    cov = coverage(idx, out)
    train, dev = split_manifests(idx, out)
    curve = learning_curve(idx, out)
    verdict = historical(out)
    readiness(idx, cov, train, dev, curve, out, verdict)
    write_json(out / "AUDIT_RUN_METADATA.json", {
        "status": "COMPLETE", "source": str(SOURCE), "source_sha256": sha256(SOURCE),
        "source_rows": len(d), "sealed_test_roots_checked": sorted(TEST_TOKENS),
        "sealed_test_roots_found": False, "artifacts": sorted(p.name for p in out.iterdir()),
    })
    files = sorted(p for p in out.iterdir() if p.is_file())
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files if p.name != "SHA256SUMS.txt"), encoding="utf-8")
    decision = json.loads((out / "FIXED_SCENE_DATA_READINESS.json").read_text(encoding="utf-8"))["decision"]
    print(json.dumps({"status":"COMPLETE", "out":str(out), "labels":len(idx), "scenes":int(idx.scene_id.nunique()), "contexts":int(idx.context_id.nunique()), "train":len(train), "dev":len(dev), "decision":decision}, indent=2))


if __name__ == "__main__":
    main()
