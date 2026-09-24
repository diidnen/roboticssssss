#!/usr/bin/env python3
"""Build and evaluate a model-independent force-critical subset from existing OOF data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_pooled_predictive_verifier as ppv
import run_predictive_verifier_development as pv


ROOT = Path("/home/exouser/FORTE")
SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000/collection_train")
POOLED = ROOT / "pooled_predictive_verifier_20260901_033804"
OUT = ROOT / "existing_force_critical_benchmark_20260901_052000"
TASKS = (0, 5, 6)
SEEDS = (0, 1, 2)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def prepare() -> None:
    OUT.mkdir(exist_ok=True)
    definition = {
        "status": "RETROSPECTIVE_RULE_FROZEN_BEFORE_CASE_LEVEL_MODEL_QUERY",
        "scope": "existing authoritative pooled TRAIN population evaluated OOF by grouped root CV; tasks0/5/6 direct labels only",
        "membership": {
            "type1": "within one context, two adjacent force cells; both repeats fail at F_low and both repeats succeed at F_high",
            "type2": "type1 plus every F_low repeat has pick_success=1 and lift_success=1 and downstream transport/place/grip loss",
        },
        "excluded": {
            "task1": "140/180 outcomes are terminal-height reconstructed rather than direct labels",
            "task6_if_empty": "no relaxation of the both-repeat transition rule",
            "new_partial_challenge_collection": "the stopped 3-context collection is not used",
        },
        "selection_inputs": ["real full-task outcome", "pick/lift/transport/place fields", "adjacent requested force", "repeat identity"],
        "forbidden_selection_inputs": ["Direct probability", "verifier logit", "World Model prediction", "method-selected force"],
        "claim_limit": "retrospective model-independent stress-test analysis, not untouched TEST and not prospectively preregistered paper confirmation",
        "no_new_simulation": True,
    }
    definition_path = OUT / "FORCE_CRITICAL_EXISTING_DEFINITION.json"
    write_json(definition_path, definition)
    cases, member_contexts, source_hashes = [], [], {}
    for task in TASKS:
        path = SOURCE / f"task{task}/task{task}/branches.csv"
        source_hashes[str(path)] = sha(path)
        d = pd.read_csv(path)
        d["full_task_success_y"] = pd.to_numeric(d.full_task_success_y).astype(int)
        for cid, q0 in d.groupby("context_id", sort=True):
            cells = []
            for force, q in q0.groupby("requested_force_N", sort=True):
                cells.append({"force_N": float(force), "n": len(q), "success_min": int(q.full_task_success_y.min()), "success_max": int(q.full_task_success_y.max())})
            for low, high in zip(cells[:-1], cells[1:]):
                if low["n"] != 2 or high["n"] != 2 or low["success_max"] != 0 or high["success_min"] != 1:
                    continue
                low_rows = q0[np.isclose(q0.requested_force_N, low["force_N"])]
                delayed = bool(((low_rows.pick_success == 1) & (low_rows.lift_success == 1) & ((low_rows.transport_retention == 0) | (low_rows.place_success == 0) | (low_rows.lost_in_transit == 1))).all())
                failure_class = "ROTATIONAL_SLIP" if task == 5 else "TRANSPORT_SLIP"
                case = {
                    "case_id": f"{cid}__{low['force_N']:.6f}_to_{high['force_N']:.6f}", "context_id": str(cid),
                    "task": task, "root_id": str(q0.root_id.iloc[0]), "root_seed": int(q0.root_seed.iloc[0]),
                    "mu": float(q0.hidden_friction_analysis_only.iloc[0]), "friction_band": str(q0.friction_band.iloc[0]),
                    "F_low_N": low["force_N"], "F_high_N": high["force_N"], "force_step_N": high["force_N"] - low["force_N"],
                    "low_repeats_failed": 2, "high_repeats_succeeded": 2, "type1_recoverable_near_frontier": True,
                    "type2_delayed_grip_failure": delayed, "failure_class": failure_class if delayed else "OTHER",
                    "selection_used_model_output": False,
                }
                cases.append(case); member_contexts.append(str(cid)); break
    if len(cases) != 14 or len(set(member_contexts)) != 14:
        raise RuntimeError(f"frozen strict rule expected 14 contexts, got {len(cases)}")
    manifest = {
        "status": "MEMBERSHIP_FROZEN_BEFORE_CASE_LEVEL_MODEL_QUERY",
        "definition_sha256": sha(definition_path), "source_hashes": source_hashes,
        "counts": {
            "cases": len(cases), "contexts": len(set(member_contexts)),
            "independent_task_root_families": len(set((x["task"], x["root_id"]) for x in cases)),
            "per_task": dict(Counter(str(x["task"]) for x in cases)),
            "delayed_cases": sum(x["type2_delayed_grip_failure"] for x in cases),
        },
        "cases": cases, "model_outputs_read_for_membership": False, "no_new_simulation": True,
        "untouched_TEST_read": False,
    }
    manifest_path = OUT / "FORCE_CRITICAL_EXISTING_MANIFEST.json"
    write_json(manifest_path, manifest)
    qa = {
        "status": "PASS_MEMBERSHIP_FROZEN", "definition_sha256": sha(definition_path), "manifest_sha256": sha(manifest_path),
        "checks": {"both_repeats_low_fail": True, "both_repeats_adjacent_high_success": True, "direct_label_tasks_only": True, "unique_contexts": True, "multiple_tasks": True, "multiple_roots": True},
        "counts": manifest["counts"], "model_outputs_read_for_membership": False, "untouched_TEST_read": False,
    }
    write_json(OUT / "FORCE_CRITICAL_EXISTING_QA.json", qa)
    print(json.dumps({"status": qa["status"], **manifest["counts"], "manifest_sha256": qa["manifest_sha256"]}, indent=2))


def stats_from_ck(ck: dict) -> dict:
    return {k: np.asarray(v, np.float32) if isinstance(v, list) else v for k, v in ck["stats"].items()}


def evaluate() -> None:
    manifest_path = OUT / "FORCE_CRITICAL_EXISTING_MANIFEST.json"
    qa_path = OUT / "FORCE_CRITICAL_EXISTING_QA.json"
    if not manifest_path.exists() or json.loads(qa_path.read_text())["status"] != "PASS_MEMBERSHIP_FROZEN":
        raise RuntimeError("run --prepare and freeze membership first")
    manifest = json.loads(manifest_path.read_text())
    member_ids = {x["context_id"] for x in manifest["cases"]}
    scratch = OUT / "_evaluation_audit"; scratch.mkdir(exist_ok=True)
    tpi, cf, full, cmap, traces, meta, audits, pairs, segs, norm = ppv.population(OUT, "existing_challenge_eval")
    md = ppv.frame(traces, meta)
    keep = md.context_id.astype(str).isin(member_ids).to_numpy()
    if int(keep.sum()) != 140:  # 14 contexts x 5 forces x 2 repeats
        raise RuntimeError(f"expected 140 member branches, got {keep.sum()}")
    direct_seed = np.full((3, len(md)), np.nan, np.float32)
    verifier_logits = np.full((3, len(md)), np.nan, np.float32)
    for fold in ppv.FOLDS:
        for seed in SEEDS:
            z = np.load(POOLED / "shards" / f"fold{fold}_seed{seed}.npz")
            va = z["held_idx"]
            direct_seed[seed, va] = ppv.sigmoid(z["direct_logits"][va])
            cp = POOLED / "checkpoints" / f"VERIFIER_WM-Current-Pooled_V0_LINEAR_fold{fold}_seed{seed}.pt"
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            model = pv.make_verifier("V0_LINEAR")
            model.load_state_dict(ck["state_dict"]); model.eval()
            verifier_logits[seed, va] = pv.predict_logits(model, stats_from_ck(ck), z["current_pred"], z["extra"], va, torch.device("cpu"))
    if not np.isfinite(direct_seed).all() or not np.isfinite(verifier_logits).all():
        raise RuntimeError("incomplete OOF predictions")
    p_direct = direct_seed.mean(0)
    sub = md[keep].copy().reset_index(drop=False).rename(columns={"index": "global_index"})
    ps = p_direct[sub.global_index.to_numpy()]
    vz = verifier_logits[:, sub.global_index.to_numpy()]

    direct_rows = ppv.direct_policy_rows(sub.drop(columns="global_index"), ps, "DIRECT")
    one_rows = ppv.direct_policy_rows(sub.drop(columns="global_index"), ps, "ONE_STEP")
    max_rows = ppv.direct_policy_rows(sub.drop(columns="global_index"), ps, "MAX")
    verifier_by_seed = [pv.controller_rows(sub.drop(columns="global_index"), ps, vz[s], "single", "STRICT") for s in SEEDS]
    fallback_by_seed = [pv.controller_rows(sub.drop(columns="global_index"), ps, vz[s], "single", "FALLBACK") for s in SEEDS]

    direct_map = {(r["context_id"], r["repeat"]): r for r in direct_rows}
    def augment(rows, method, seed):
        out = []
        for r in rows:
            key = (r["context_id"], r["repeat"]); dr = direct_map[key]
            q = sub[(sub.context_id == key[0]) & (sub["repeat"] == key[1])].sort_values("force_N")
            higher_success = bool(((q.force_N > dr["selected_force_N"] + 1e-9) & (q.success > 0)).any())
            recoverable_direct_failure = bool(dr["actual_success"] < 1 and higher_success)
            escalation = bool(r.get("selected_force_N", math.nan) > dr["selected_force_N"] + 1e-9) if r.get("coverage", 0) else False
            out.append({**r, "method": method, "seed": seed, "direct_actual_success": dr["actual_success"],
                        "recoverable_direct_failure": int(recoverable_direct_failure),
                        "rescued_recoverable_direct_failure": int(recoverable_direct_failure and r.get("coverage", 0) and r.get("actual_success", 0) > 0),
                        "collateral_escalation": int(dr["actual_success"] > 0 and escalation),
                        "collateral_rejection": int(dr["actual_success"] > 0 and not r.get("coverage", 0)),
                        "selective_escalation_true_positive": int(escalation and dr["actual_success"] < 1),
                        "escalated": int(escalation)})
        return out
    all_rows = []
    for method, rows in [("Direct", direct_rows), ("One-Step", one_rows), ("Fixed-Max", max_rows)]:
        all_rows += augment(rows, method, "DIRECT_ENSEMBLE")
    for seed in SEEDS:
        all_rows += augment(verifier_by_seed[seed], "Predictive-Verifier", seed)
        all_rows += augment(fallback_by_seed[seed], "Verifier-MaxFallback", seed)
    per = pd.DataFrame(all_rows)
    per.to_csv(OUT / "FORCE_CRITICAL_EXISTING_PER_CASE.csv", index=False)

    summary = []
    for (method, seed), q in per.groupby(["method", "seed"], sort=False):
        rec = int(q.recoverable_direct_failure.sum()); esc = int(q.escalated.sum())
        summary.append({
            "method": method, "seed": seed, "episodes": len(q), "SR": float((q.actual_success*q.coverage).mean()),
            "coverage": float(q.coverage.mean()), "under_force": float(q.under_force.mean()), "mean_force_N": float(q.mean_force_component.mean()),
            "excess_force_N": float(q.excess_force_N.mean()), "recoverable_direct_failures": rec,
            "rescue_rate": float(q.rescued_recoverable_direct_failure.sum()/rec) if rec else math.nan,
            "collateral_escalation_rate": float(q.collateral_escalation.sum()/max(int((q.direct_actual_success>0).sum()),1)),
            "collateral_rejection_rate": float(q.collateral_rejection.sum()/max(int((q.direct_actual_success>0).sum()),1)),
            "selective_escalation_precision": float(q.selective_escalation_true_positive.sum()/esc) if esc else math.nan,
            "NO_VALID_FORCE_rate": float(q.NO_VALID_FORCE.mean()),
        })
    sdf = pd.DataFrame(summary)
    means = sdf[sdf.method.isin(["Predictive-Verifier", "Verifier-MaxFallback"])].groupby("method", as_index=False).mean(numeric_only=True)
    means["seed"] = "MEAN_3_SEEDS"
    sdf = pd.concat([sdf, means], ignore_index=True)
    sdf.to_csv(OUT / "FORCE_CRITICAL_EXISTING_TABLE.csv", index=False)
    classification = {
        "status": "RETROSPECTIVE_EXISTING_DATA_EVALUATION_COMPLETE",
        "challenge_manifest_sha256": sha(manifest_path), "no_new_simulation": True, "untouched_TEST_read": False,
        "claim_limit": manifest["status"],
        "results": sdf[sdf.seed.astype(str).isin(["DIRECT_ENSEMBLE", "MEAN_3_SEEDS"])].to_dict("records"),
    }
    write_json(OUT / "FORCE_CRITICAL_EXISTING_CLASSIFICATION.json", classification)
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["prepare", "evaluate"]); args = ap.parse_args()
    prepare() if args.mode == "prepare" else evaluate()
