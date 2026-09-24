#!/usr/bin/env python3
"""Read-only forensic audit of the historical 720-row ActiveForcing archive.

This script writes only a new audit bundle.  It never changes the historical
CSV, telemetry, checkpoints, or the current posterior.  Labels are recomputed
from each raw branch trace independently so that context-level caching bugs are
visible rather than reproduced.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/exouser/FORTE")
OLD = ROOT / "gnp_style_continuous_20260830_125107"
OLD_CSV = OLD / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
OLD_CONTEXT_AUDIT = OLD / "CONTINUOUS_TRAIN_CONTEXT_AUDIT.json"
OLD_RUN = OLD / "CONTINUOUS_TRAIN_COLLECTION_RUN_MANIFEST.json"
OLD_TEL = OLD / "CONTINUOUS_TRAIN_TELEMETRY_AUDIT.json"
OLD_PROTO = OLD / "GNP_STYLE_CONTINUOUS_TRAINING_PROTOCOL.json"
OLD_CK = OLD / "GNP_STYLE_CHECKPOINT_MANIFEST.json"
P5_BRANCH = Path("/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_BRANCH_MANIFEST.csv")
GNP_TRAIN = ROOT / "gnp_style_continuous.py"
GNP_COLLECT = ROOT / "gnp_style_continuous_collect.py"
OLD_SPLIT = ROOT / "activeforcing_full_claim_closure_20260902_062809/DIRECT_ONLY_PROVENANCE/ACTIVEFORCING_FINAL_SPLIT_MANIFEST.json"
OLD_PROTOCOL = ROOT / "activeforcing_full_claim_closure_20260902_062809/DIRECT_ONLY_PROVENANCE/ACTIVEFORCING_FINAL_PROTOCOL.json"
OLD_BENCH = ROOT / "activeforcing_full_claim_closure_20260902_062809/DIRECT_ONLY_PROVENANCE/MAIN_PAIRED_BENCHMARK_RESULTS.json"
OLD_BENCH_REPORT = ROOT / "activeforcing_full_claim_closure_20260902_062809/DIRECT_ONLY_PROVENANCE/MAIN_PAIRED_BENCHMARK_REPORT.md"
OLD_EVAL_SOURCE = ROOT / "run_residual_utility.py"
CURRENT_OUT = ROOT / "analysis/results/final_no_probe_continuous_posterior_20260904"
CURRENT_DATA = CURRENT_OUT / "03_true_force_dataset/FINAL_NO_PROBE_TRUE_FORCE_DATASET.csv"
CURRENT_SPLIT = CURRENT_OUT / "TRAIN_DEV_TEST_SPLIT.json"
CURRENT_CFG = CURRENT_OUT / "05_training/POSTERIOR_TRAINING_CONFIG.json"
CURRENT_RESULT = CURRENT_OUT / "05_training/POSTERIOR_TRAINING_RESULT.json"
CURRENT_SCHEMA = CURRENT_OUT / "FINAL_FEATURE_SCHEMA.json"
CURRENT_CURVE = CURRENT_OUT / "ROOT7703_POSTERIOR_CURVE.json"
CURRENT_HELD_CURVE = CURRENT_OUT / "ROOT7703_POSTERIOR_CURVE_HELDOUT_DIAGNOSTIC.json"
ROOT7703_4N = ROOT / "analysis/results/root7703_reset_replay_reproduction_20260904_final_v3/LIVE_4N_TRACE.csv"
ROOT7703_2N = ROOT / "analysis/results/root7703_reset_replay_reproduction_20260904_final_v3/LIVE_2N_TRACE.csv"
ROOT7703_6N = ROOT / "analysis/results/root7703_reset_replay_reproduction_20260904_final_v3/LIVE_6N_TRACE.csv"
ROOT7703_409 = CURRENT_OUT / "07_root7703_activeforcing_smoke/isaac_4p09N/LIVE_4p09N_TRACE.csv"
OUT = ROOT / "analysis/results/old720_testing_protocol_audit_20260904"


def sha(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(name: str, obj: Any) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def num(row: dict[str, str], key: str) -> float:
    x = row.get(key, "")
    return float(x) if x not in ("", None) else float("nan")


def old_final_label(path: Path) -> tuple[int, dict[str, Any]]:
    """Recompute exactly the common lift+30 bilateral-hold contract per branch."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, {"reason": "empty_trace"}
    old_schema = "object_z_analysis_only" in rows[0]
    z0 = num(rows[0], "object_z_analysis_only") if old_schema else 0.0
    def dh(r: dict[str, str]) -> float:
        return (num(r, "object_z_analysis_only") - z0) if old_schema else num(r, "object_height_delta_m")
    def bilateral(r: dict[str, str]) -> bool:
        return r.get("contact_state") == "bilateral" if old_schema else r.get("bilateral_contact") in ("1", "True", "true")
    def above(r: dict[str, str]) -> bool:
        if old_schema:
            return dh(r) >= 0.005
        return dh(r) >= 0.005 and r.get("drop") not in ("1", "True", "true")
    lift = next((i for i, r in enumerate(rows) if dh(r) >= 0.01), None)
    if lift is None:
        return 0, {"reason": "fail_before_lift", "lift_index": None, "hold_rows": 0}
    window = rows[lift + 1 : lift + 31]
    ok = len(window) >= 30 and all(bilateral(r) and above(r) for r in window)
    return int(ok), {"reason": "lift_and_hold_success" if ok else "fail_after_lift", "lift_index": lift, "hold_rows": len(window)}


def trace_metrics(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lift = next((i for i, r in enumerate(rows) if (num(r, "object_height_delta_m") >= 0.01)), None)
    if lift is None:
        lift = next((i for i, r in enumerate(rows) if r.get("lift_threshold_reached") in ("1", "True", "true")), None)
    end = lift if lift is not None else len(rows) - 1
    def vals(k: str) -> list[float]:
        return [num(r, k) for r in rows if math.isfinite(num(r, k))]
    def relevant(k: str) -> list[float]:
        return [num(r, k) for r in rows[: end + 1] if math.isfinite(num(r, k))]
    out: dict[str, Any] = {"trace": str(path), "trace_sha256": sha(path), "rows": len(rows), "lift_row_index": lift}
    for k in ["aggregate_force", "effective_force_target", "requested_force_target", "aperture", "object_height_delta_m"]:
        a, b = vals(k), relevant(k)
        if a:
            out[k] = {"all_mean": statistics.mean(a), "all_peak": max(a), "relevant_mean": statistics.mean(b) if b else None,
                      "relevant_peak": max(b) if b else None, "relevant_integrated_dt_0p05": sum(b) * 0.05 if b else None}
    for k in ["bilateral_contact", "left_contact", "right_contact", "drop", "slip"]:
        a = [num(r, k) for r in rows if r.get(k, "") not in ("", None)]
        out[k] = {"all_sum": sum(a) if a else None, "relevant_sum": sum(a[: end + 1]) if a else None}
    out["final_success_label"] = old_final_label(path)[0] if "object_z_analysis_only" in rows[0] else None
    return out


def current_curve_summary(path: Path) -> dict[str, Any]:
    q = load_json(path, {}) or {}
    fs, ps = q.get("force_grid_N", []), q.get("posterior_success", [])
    by = {}
    for t in [2.0, 2.5, 3.0, 3.5, 4.0, 4.09, 6.0]:
        if fs and ps:
            i = min(range(len(fs)), key=lambda j: abs(float(fs[j]) - t))
            by[str(t)] = {"force_N": float(fs[i]), "p_success": float(ps[i])}
    return {"path": str(path), "selected_force_N": q.get("selected_force_N"), "selected_p": q.get("selected_predicted_success_probability"), "values": by}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with OLD_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    old_labels: dict[str, tuple[int, dict[str, Any]]] = {}
    for r in rows:
        old_labels[r["branch_id"]] = old_final_label(Path(r["telemetry_path"]))

    by_ctx: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_ctx[r["context_id"]].append(r); by_root[r["root_id"]].append(r); by_task[r["task"]].append(r)
    old_tasks = sorted(int(x) for x in by_task)
    task_breakdown = []
    for t in old_tasks:
        g = by_task[str(t)]
        task_breakdown.append({"task": t, "samples": len(g), "unique_contexts": len({r["context_id"] for r in g}),
          "unique_roots": len({r["root_id"] for r in g}), "unique_tuples": len({(r["task"], r["root_id"], r["context_id"]) for r in g}),
          "old_success": sum(int(r["full_task_success_y"]) for r in g), "old_failure": sum(1-int(r["full_task_success_y"]) for r in g),
          "recomputed_final_success": sum(old_labels[r["branch_id"]][0] for r in g), "recomputed_final_failure": sum(1-old_labels[r["branch_id"]][0] for r in g),
          "unique_force_candidates": len({float(r["requested_force_N"]) for r in g}),
          "force_min_N": min(float(r["requested_force_N"]) for r in g), "force_max_N": max(float(r["requested_force_N"]) for r in g),
          "force_support_from_protocol_N": (load_json(OLD_PROTO, {}) or {}).get("continuous_collection", {}).get("force_support_N", {}).get(str(t))})
    context_rows = []
    for cid in sorted(by_ctx):
        g = sorted(by_ctx[cid], key=lambda r: (float(r["requested_force_N"]), int(r["repeat"])))
        first = g[0]; final_y = [old_labels[r["branch_id"]][0] for r in g]
        old_y = [int(r["full_task_success_y"]) for r in g]
        context_rows.append({
            "context_id": cid, "root_id": first["root_id"], "task": int(first["task"]), "friction_band": first["friction_band"],
            "friction": float(first["friction"] if first.get("friction") else first.get("hidden_friction_analysis_only", "nan")),
            "sample_count": len(g), "unique_force_candidates": len({float(r["requested_force_N"]) for r in g}),
            "candidate_forces_N": json.dumps([float(r["requested_force_N"]) for r in g], separators=(",", ":")),
            "old_full_task_outcomes": json.dumps(old_y, separators=(",", ":")),
            "final_lift_hold_outcomes_recomputed": json.dumps(final_y, separators=(",", ":")),
            "old_success_count": sum(old_y), "old_failure_count": len(old_y)-sum(old_y),
            "final_success_count": sum(final_y), "final_failure_count": len(final_y)-sum(final_y),
            "old_outcome_varies_within_context": int(len(set(old_y)) > 1), "final_outcome_varies_within_context": int(len(set(final_y)) > 1),
            "all_valid": int(all(r["valid"] == "1" for r in g)), "all_corrected_true_force_telemetry": int(all(r["corrected_physical_telemetry_valid"] == "1" for r in g)),
            "probe_hash_present": int(bool(first.get("strict_preprobe_hash"))), "nominal_hash_present": int(bool(first.get("nominal_motion_hash"))),
        })
    fields = list(context_rows[0])
    with (OUT / "OLD720_CONTEXT_BREAKDOWN.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(context_rows)

    old_run = load_json(OLD_RUN, {}) or {}; old_tel = load_json(OLD_TEL, {}) or {}; old_proto = load_json(OLD_PROTO, {}) or {}; old_ck = load_json(OLD_CK, {}) or {}
    old_split = load_json(OLD_SPLIT, {}) or {}; old_protocol = load_json(OLD_PROTOCOL, {}) or {}; bench = load_json(OLD_BENCH, {}) or {}
    old_recomputed = [old_labels[r["branch_id"]][0] for r in rows]
    cross = Counter((int(r["full_task_success_y"]), y) for r, y in zip(rows, old_recomputed))
    old_context_audit = load_json(OLD_CONTEXT_AUDIT, {}) or {}

    source_candidates = [OLD_CSV, OLD_RUN, OLD_TEL, OLD_CONTEXT_AUDIT, OLD_PROTO, OLD_CK, P5_BRANCH, GNP_TRAIN, GNP_COLLECT,
                         ROOT / "pooled_original_taskwise_eval.py", ROOT / "run_frozen_direct_dev_scorer.py", OLD_EVAL_SOURCE,
                         OLD_SPLIT, OLD_PROTOCOL, OLD_BENCH, OLD_BENCH_REPORT]
    provenance = {
      "status": "COMPLETE_READ_ONLY_FORENSIC_AUDIT", "audit_date": "2026-09-04", "old720_definition": "CONTINUOUS_TRAIN_SUCCESS_DATA.csv",
      "dataset_path": str(OLD_CSV), "dataset_sha256": sha(OLD_CSV), "rows": len(rows),
      "manifest": str(OLD_RUN), "manifest_sha256": sha(OLD_RUN), "generation_script": str(GNP_COLLECT),
      "training_script": str(GNP_TRAIN), "checkpoint_manifest": str(OLD_CK), "checkpoint_manifest_sha256": sha(OLD_CK),
      "feasibility_checkpoints": [x.get("checkpoint") for x in old_ck.get("checkpoints", []) if x.get("backend") == "FEASIBILITY_ONLY"],
      "split_config_candidates": [str(OLD_SPLIT), str(OLD_PROTO), str(OLD_PROTOCOL)],
      "source_hashes": {str(p): sha(p) for p in source_candidates if p.exists()},
      "collection_facts": {"expected": old_run.get("expected"), "valid": old_run.get("valid"), "successes": old_run.get("successes"), "failures": old_run.get("failures"),
                           "corrected_true_force_branches": old_tel.get("corrected_valid_telemetry_branches"), "unique_force_cells": old_tel.get("requested_force_unique_cells")},
      "old_gnp_training_population": {"coarse_branches": 288, "continuous_720_branches": 720, "total_feasibility_outcomes": 1008,
                                      "joint_physical_branches": 720, "adjacent_ie_pairs": 576, "evidence": str(GNP_TRAIN)},
      "historical_related_but_not_old720": {"e5": "60 tuples / 300 fresh rollouts is a separate 2026-09 E5 reset-to-end campaign, not this 720-row GNP archive", "main_paired_benchmark": str(OLD_BENCH)},
    }
    write_json("OLD720_PROVENANCE.json", provenance)
    write_json("OLD720_TASK_BREAKDOWN.json", {"total_samples": len(rows), "tasks": task_breakdown, "all_tasks": old_tasks,
      "unique_contexts": len(by_ctx), "unique_roots": len(by_root), "unique_tuples_operational": len({(r["task"], r["root_id"], r["context_id"]) for r in rows}),
      "tuple_definition": "(task, root_id, context_id); one tuple per friction-conditioned context; root-family count is reported separately",
      "sample_grain": "one physical branch at (task, root_id, context_id, requested_force_N, repeat)",
      "samples_per_context": dict(Counter(len(v) for v in by_ctx.values())), "contexts_per_root": dict(Counter(len({r["context_id"] for r in v}) for v in by_root.values()))})

    # Old GNP split is intentionally represented separately from the later archive OOF split.
    write_json("OLD_TRAIN_DEV_TEST_SPLIT.json", {
      "status": "RECOVERED_WITH_TWO_LAYERS", "old_gnp_training": {"split_unit": "TRAIN population only", "train_contexts": 72, "train_roots": 24,
        "train_tasks": old_tasks,
        "train_samples_continuous_720": 720, "additional_coarse_train_branches": 288, "dev_contexts": 0, "test_contexts": 0,
        "dev_outcomes_loaded": False, "test_loaded": False, "evidence": [str(OLD_CK), str(OLD_CONTEXT_AUDIT), str(OLD_PROTO)]},
      "old_gnp_dev_touch": {"split_unit": "separate repeated DEV contexts", "contexts": 9, "roots": 7, "probability_cells": 27, "branches": 135,
        "tasks": [0, 1, 5, 6],
        "loaded_after_all_six_checkpoints": True, "evidence": str(OLD / "CONTINUOUS_DEV_PROBABILITY_METRICS.csv")},
      "archive_direct_oof_evaluation": {"split_unit": "task-specific root family", "unit": "root-held-out grouped OOF", "folds": 3,
        "root_families": 24, "contexts": 72, "fold_rule": old_split.get("fold_rule"), "folds": old_split.get("folds"), "test_rows_are_held_out_roots": True,
        "train_tasks": old_tasks, "dev_tasks": old_tasks, "test_tasks": old_tasks,
        "same_root_context_force_repeat_kept_together": True, "untouched_original_test_loaded": False, "evidence": str(OLD_SPLIT)},
      "classification": "B_ROOT_HELD_OUT_WITHIN_SEEN_TASKS for the archive Direct OOF; the original GNP 720 training/checkpoint phase itself was TRAIN-only with a post-freeze DEV touch",
      "task_ood": False, "seen_task_heldout_context_or_root": True,
    })
    write_json("OLD_EVALUATION_PROTOCOL.json", {
      "classification": "B_ROOT_HELD_OUT_WITHIN_SEEN_TASKS (later archive Direct OOF); GNP training phase TRAIN-only plus separate DEV diagnostic",
      "old_gnp_posterior_eval": {"input": "context sequence + candidate force condition", "output": "ensemble mean sigmoid feasibility probability",
        "primary_calibration": "raw ensemble mean", "secondary": "TRAIN-only isotonic diagnostic", "rho": old_proto.get("calibration", {}).get("rho", 0.8),
        "selector": "minimum continuous 0.05N grid force with raw p >= rho in GNP DEV analysis; Expected Utility was not reached in that GNP run",
        "success_criterion": "full_task_success_y from historical full-task runner", "dev": "9 contexts, 27 real force cells, 135 repeated branches; read once after six checkpoints"},
      "archive_direct_posterior_eval": {"input": "held-out root/context + candidate force", "output": "P(success|x,F), selected force and realized outcome",
        "selector": "Expected Utility with lower-force tie break", "force_cost": "failure reward -1; success reward (Fmax-F)/Fmax",
        "formula": "U(F|x)=p_success(F|x)*(Fmax-F)/Fmax + (1-p_success(F|x))*(-1)",
        "task_Fmax_N": old_protocol.get("utility", {}).get("Fmax", "task/archive maximum; frozen Direct runtime uses candidate-grid max"),
        "candidate_grid_N": old_protocol.get("direct", {}).get("candidate_grid_N"), "evaluation": "72 contexts x 2 selected-force repeats = 144 paired episodes",
        "evidence": [str(OLD_BENCH), str(OLD_BENCH_REPORT), str(OLD_PROTOCOL)]},
      "not_task_ood": True, "historical_original_test": "not loaded/used in the cited closure artifacts",
    })

    root7703_in_old = [r for r in rows if "7703" in r.get("root_id", "") or "7703" in r.get("context_id", "")]
    write_json("ROOT7703_OLD_DATA_MEMBERSHIP.json", {"present_in_old720": bool(root7703_in_old), "matching_rows": root7703_in_old,
      "role": "NOT_PRESENT; separate integration/debug root, not a member of old720 train/OOf population",
      "task": 5, "demo": "demo_3", "context_id": "root7703", "old720_root_count": len(by_root), "old720_context_count": len(by_ctx),
      "formal_test_membership": "not an old720 held-out root; current pilot also explicitly trained on it for smoke fitting"})

    write_json("OLD_FORCE_SEMANTICS_AUDIT.json", {
      "classification": "ABSOLUTE_NEWTON_CANDIDATE", "requested_field": "requested_force_N", "realized_field": "realized_force_N",
      "old_continuous_support_N": old_proto.get("continuous_collection", {}).get("force_support_N"), "unique_force_cells": len({(r["context_id"], r["requested_force_N"]) for r in rows}),
      "candidate_distribution": {"unique_requested_values": len({r["requested_force_N"] for r in rows}), "per_context": "5 force candidates x 2 repeats"},
      "true_force_evidence": {"corrected_valid_telemetry_branches": old_tel.get("corrected_valid_telemetry_branches"), "proxy_or_double_rotation": old_tel.get("known_proxy_or_double_rotation_targets_used"),
        "telemetry_audit_definition": old_tel.get("corrected_definition")},
      "interpretation": "command candidate is absolute N; realized physical force is separate measured telemetry and is not equal to the command by construction",
      "not": ["RELATIVE_FORCE", "DISCRETE_LEVEL", "PROXY_FORCE"],
    })
    write_json("OLD_SUCCESS_SEMANTICS_AUDIT.json", {
      "old_label_field": "full_task_success_y", "old_success_count": sum(int(r["full_task_success_y"]) for r in rows), "old_failure_count": sum(1-int(r["full_task_success_y"]) for r in rows),
      "old_success_definition": "historical full-task runner terminal success / full_task_success_y; full-task transport/place contract, not lift+30 hold",
      "final_success_definition": "lift + 30-step bilateral hold",
      "match": "NO", "independent_recomputed_final_counts": {"success": sum(old_recomputed), "failure": len(rows)-sum(old_recomputed)},
      "cross_tab_old_label_vs_recomputed_final": {f"old_{a}_final_{b}": n for (a,b), n in sorted(cross.items())},
      "agreement": sum(n for (a,b),n in cross.items() if a == b), "mismatch": sum(n for (a,b),n in cross.items() if a != b),
      "caveat": "recomputed labels are a raw-telemetry projection, not proof that old strict P4-B context construction is final no-probe compatible",
      "source": [str(OLD_CSV), str(GNP_TRAIN), str(OLD_TEL)],
    })

    # Compatibility: strict final compatibility is zero because every old context was built around strict P4-B preprobe state.
    comp_contexts = []
    for c in context_rows:
        comp_contexts.append({"context_id": c["context_id"], "root_id": c["root_id"], "task": c["task"],
          "primary_class": "E_PROBE_DEPENDENT", "projected_reuse_class": "B_FORCE_SEMANTICS_COMPATIBLE_BUT_OUTCOME_DIFFERENT",
          "strict_final_protocol_compatible": False, "no_probe_nominal_projection_recoverable": True,
          "raw_true_force_telemetry_recoverable": bool(c["all_corrected_true_force_telemetry"]), "final_label_recomputable_from_trace": True,
          "reason": "old context audit defines last stable P4-B hold before probe_out and source data carries probe/preprobe hashes; nominal arm prefix can be projected, but this is not exact final reset/handoff admission"})
    write_json("OLD_TO_FINAL_COMPATIBILITY.json", {"context_classification": comp_contexts,
      "counts": {"old_contexts_total": len(comp_contexts), "final_protocol_compatible": 0, "partially_reusable": len(comp_contexts), "invalid": 0,
                  "probe_dependent": len(comp_contexts), "old_samples_total": len(rows), "final_protocol_compatible_samples": 0, "partially_reusable_samples": len(rows), "invalid_samples": 0,
                  "reusable_projected_samples_if_row_labels_recomputed": len(rows)},
      "classes": {"A": 0, "B": len(rows), "C": 0, "D": len(comp_contexts), "E": len(comp_contexts), "F": 0},
      "interpretation": "all 72 contexts are partial/projectable reuse, but none are strict final-protocol-compatible; the old protocol mismatch is structural, not lack of context count"})

    # Current mixture audit, including row-level label corruption and root leakage in current context-level split.
    current = []
    if CURRENT_DATA.exists():
        with CURRENT_DATA.open(newline="", encoding="utf-8") as f: current = list(csv.DictReader(f))
    old_cur = [r for r in current if r.get("source_class", "").startswith("OLD")]
    old_map = {r["branch_id"]: r for r in rows}
    cur_label_mismatch = []
    for r in old_cur:
        bid = r.get("sample_id", "").removeprefix("old::")
        if bid in old_map:
            y, _ = old_labels[bid]
            if int(float(r["success_y"])) != y: cur_label_mismatch.append({"branch_id": bid, "context_id": r["context_id"], "force_N": float(r["requested_force_N"]), "current": int(float(r["success_y"])), "recomputed": y})
    split = load_json(CURRENT_SPLIT, {}) or {}
    split_groups = {k: set(v) for k,v in [("train", split.get("train_groups", [])), ("dev", split.get("dev_groups", [])), ("test", split.get("test_groups", []))]}
    roots_by_split = {k: {g.split("::",1)[0] for g in v} for k,v in split_groups.items()}
    root_overlap = {f"{a}_vs_{b}": sorted(roots_by_split[a] & roots_by_split[b]) for a,b in [("train","dev"),("train","test"),("dev","test")]}
    current_source_counts = Counter(r.get("source_class", "") for r in current)
    current_label_counts = Counter(int(float(r["success_y"])) for r in current if r.get("success_y") not in ("", None))
    write_json("CURRENT_TRAINING_DATA_MIX_AUDIT.json", {
      "current_dataset": str(CURRENT_DATA), "current_rows": len(current), "source_counts": dict(current_source_counts), "label_counts": dict(current_label_counts),
      "old_rows_in_current": len(old_cur), "new_true_force_rows_in_current": len(current)-len(old_cur), "old_to_new_weight_ratio": len(old_cur)/(len(current)-len(old_cur)) if len(current)>len(old_cur) else None,
      "old_row_label_recomputed_success": sum(old_labels[r["branch_id"]][0] for r in rows), "current_old_label_mismatch_rows": len(cur_label_mismatch),
      "current_old_label_mismatch_contexts": len({x["context_id"] for x in cur_label_mismatch}), "current_old_label_mismatch_examples": cur_label_mismatch[:20],
      "label_cache_bug_detected": len(cur_label_mismatch) > 0,
      "label_bug_evidence": "final_no_probe_prepare_dataset.py caches common_label(trace) by context_id and then applies it to every force row in that context; row-level recomputation above disagrees",
      "no_probe_fields_used": True, "outcome_in_context": False, "probe_in_context": False, "force_is_candidate_not_context": True,
      "split": {"unit": split.get("split_unit"), "counts": split.get("counts"), "root_overlap_between_splits": root_overlap, "root_level_holdout_valid": not any(root_overlap.values()),
                "context_level_grouping_valid": True, "formal_benchmark_flag": split.get("formal_benchmark")},
      "normalization": "current normalization is fit on current train indices; old rows dominate and current split is not root-family-held-out",
      "verdict": "NO: current mixture is not valid for formal final evaluation; it combines structurally old P4-B contexts, a row-label cache error, 720:11 source imbalance, and context-level rather than root-family split",
    })

    result = load_json(CURRENT_RESULT, {}) or {}; cfg = load_json(CURRENT_CFG, {}) or {}; schema = load_json(CURRENT_SCHEMA, {}) or {}
    write_json("CURRENT_POSTERIOR_AUDIT.json", {
      "model": cfg.get("architecture"), "input": cfg.get("input"), "probe_feature_used": cfg.get("probe_feature_used"), "force_domain_N": cfg.get("force_domain_N"),
      "architecture_relation_to_old": "same GRU/MLP feasibility family, but input contract changed from old 17+54 P4-B condition to 16 no-probe nominal+task and candidate F; this is an adaptation, not exact old architecture",
      "metrics_from_frozen_result": result.get("train_dev_test_metrics"), "ensemble_test_metrics": result.get("ensemble_test_metrics"),
      "calibration_valid": False, "calibration_reason": "no independently fit final calibration artifact; current ECE is diagnostic and is evaluated under a split with root overlap and corrupted old labels",
      "monotonicity_valid": False, "monotonicity": result.get("monotonicity_audit"),
      "monotonicity_reason": "2/74 audited contexts have adjacent decreases and 772 total decreases; code uses a soft penalty, not a hard monotone model",
      "selector": "current root7703 smoke utility p*(1-F/8)+(1-p)*(-1), continuous 1..8N; this is not the old GNP rho=0.80 selector and task-specific archive Fmax lineage is separate",
      "pilot_contamination": cfg.get("pilot_smoke_include_root7703_in_train"), "formal_training_valid": False,
      "schema_audit": schema,
      "root7703_curve": current_curve_summary(CURRENT_CURVE), "root7703_heldout_diagnostic_curve": current_curve_summary(CURRENT_HELD_CURVE),
    })

    m4 = trace_metrics(ROOT7703_4N); m409 = trace_metrics(ROOT7703_409)
    # Compare numeric trace columns over equal rows; raw arm hash is a direct column-level invariant.
    with ROOT7703_4N.open(newline="", encoding="utf-8") as f: a = list(csv.DictReader(f))
    with ROOT7703_409.open(newline="", encoding="utf-8") as f: b = list(csv.DictReader(f))
    diffs = {}
    for k in ["aggregate_force", "measured_left_force", "measured_right_force", "effective_force_target", "aperture", "object_height_delta_m"]:
        if k in a[0] and k in b[0]:
            x, y = [num(r,k) for r in a], [num(r,k) for r in b]
            pairs = [(u,v) for u,v in zip(x,y) if math.isfinite(u) and math.isfinite(v)]
            diffs[k] = {"max_abs_all": max(abs(u-v) for u,v in pairs), "max_abs_relevant": max(abs(u-v) for u,v in pairs[:119])}
    arm_hash_equal = all(x.get("raw_arm_action_hash") == y.get("raw_arm_action_hash") for x,y in zip(a,b))
    write_json("ROOT7703_4N_4P1N_DIAGNOSIS.json", {
      "note": "4P1N is interpreted as the existing 4.09N branch; no new repeat was launched",
      "empirical": {"3p5N": "FAIL_AFTER_LIFT", "4N": "LIFT_AND_HOLD_SUCCESS", "frontier": "(3.5N, 4.0N]"},
      "4N": m4, "4p09N": m409, "requested_delta_N": 0.09, "unique_4N_trace_replicates": 1,
      "replicate_assessment": "NOT_ESTABLISHED: the apparent second 4N directory has identical trace SHA-256 and is a duplicate, not an independent repeat",
      "same_arm_trace": {"raw_arm_action_hash_equal_all_rows": arm_hash_equal, "handoff_parity": True},
      "physical_comparability": {"trace_column_differences": diffs, "4N_relevant_mean_realized_N": m4.get("aggregate_force", {}).get("relevant_mean"), "4p09N_relevant_mean_realized_N": m409.get("aggregate_force", {}).get("relevant_mean"),
        "4N_relevant_exposure_Ns": m4.get("aggregate_force", {}).get("relevant_integrated_dt_0p05"), "4p09N_relevant_exposure_Ns": m409.get("aggregate_force", {}).get("relevant_integrated_dt_0p05")},
      "classification": "UNKNOWN_WITH_EXECUTION_EXPOSURE_AND_STOCHASTICITY_UNRESOLVED",
      "explanation": "The handoff and frozen arm sequence match, but 4.09N has a lower relevant realized-force mean/exposure and a different contact/drop trajectory. With only one unique 4N success and one 4.09N failure, command-level deterministic frontier versus force-exposure variation cannot be separated. This is not sufficient evidence to blame the posterior alone.",
      "model_signal": current_curve_summary(CURRENT_CURVE),
    })

    # Recommended split is root-family held-out within the seen task distribution.
    train_roots = sorted({r["root_id"] for r in rows if int(r["root_id"].split("root")[1].split("_")[0]) % 3 != 0})
    dev_roots = sorted({r["root_id"] for r in rows if int(r["root_id"].split("root")[1].split("_")[0]) % 3 == 0})
    # Keep a third root group for a final held-out test; use root index modulo 3 and task-stratified family IDs.
    groups = defaultdict(list)
    for r in rows:
        idx = int(r["root_id"].split("root")[1].split("_")[0]); groups[idx % 3].append(r["root_id"])
    write_json("RECOMMENDED_FINAL_SPLIT.json", {"status": "RECOMMENDED_NOT_EXECUTED", "task_distribution_seen": True, "task_ood": False,
      "unit": "task-specific root family; keep all three friction contexts, candidate forces, and repeats together",
      "rule": "pre-register root-family groups, stratify task counts; fit normalization/calibration on TRAIN only; use DEV for selector/calibration freeze; evaluate TEST once",
      "suggested_root_index_groups": {"train": sorted(set(groups[1])), "dev": sorted(set(groups[2])), "test": sorted(set(groups[0]))},
      "minimum_structure": "all four tasks represented in each split where possible; no root ID in more than one split; root7703 remains integration/debug and is not the final test",
      "why_not_current": "current final split groups root/context but permits sibling friction contexts from one root family to cross train/dev/test",
      "evaluation": "seen-task / held-out-root-context; posterior P(success|x,F), continuous EU selection, then frozen-arm live evaluation on held-out contexts",
    })

    # Human-readable report.
    report = f"""# OLD720 testing-protocol audit (2026-09-04)

## Bottom line

The historical 720 archive is real and is not a two-context dataset. It contains **{len(rows)} branch samples, {len(old_tasks)} tasks, {len(by_ctx)} friction-conditioned contexts, {len(by_root)} task-specific root families, and 72 operational context tuples**. Each context has five continuous candidate-force cells and two repeats. The old data therefore has meaningful context diversity.

However, the old 720 archive is not directly final-protocol-compatible. All {len(by_ctx)} contexts are explicitly built from the strict P4-B preprobe/probe topology. Their absolute-Newton candidate force and corrected physical telemetry are reusable, and final lift+hold can be recomputed from raw traces, but strict final no-probe admission is zero contexts.

## Provenance and grain

- Dataset: `{OLD_CSV}`; SHA-256 `{sha(OLD_CSV)}`.
- Generation/validation: `{GNP_COLLECT}` and `{GNP_TRAIN}`; collection manifest reports 720/720 valid, 575 old full-task successes and 145 failures.
- One row is one `(task, root, context, requested_force_N, repeat)` physical branch. Shape is 24 roots × 3 friction contexts × 5 force cells × 2 repeats = 720.
- The GNP posterior training population was actually 288 old coarse branches + these 720 continuous branches = 1008 feasibility outcomes. The Joint physical auxiliary used the 720 corrected branches and 576 adjacent-force pairs.

## Task breakdown

{json.dumps(task_breakdown, indent=2)}

## Split and evaluation recovered

The GNP checkpoint phase was TRAIN-only: 72 train contexts/24 roots; DEV and TEST were false in the checkpoint manifest. A separate nine-context, 135-branch DEV benchmark was loaded once after all six checkpoints froze. The later archive Direct evaluation is a three-fold **root-held-out grouped OOF within seen tasks** over the same 24 root families; it is not task-OOD and it is not an untouched original TEST claim. The old Direct selector used `U=p*(Fmax-F)/Fmax+(1-p)*(-1)` with lower-force tie break; the GNP continuous analysis itself used rho=0.80 threshold selection and did not reach Expected Utility.

## Label and current-mix findings

Old `full_task_success_y` is a full-task terminal success label, not the current lift+30 bilateral-hold label. Independent row-level recomputation gives 670 final-style positives and 50 negatives; the cross-tab and mismatch count are in `OLD_SUCCESS_SEMANTICS_AUDIT.json`.

The current final dataset mix is **not valid for formal training/evaluation**. It contains old 720 + new true-force rows, but `{len(cur_label_mismatch)}` old rows disagree with row-wise raw-trace recomputation because `final_no_probe_prepare_dataset.py` caches a label by context and applies the first branch label to every force candidate. In addition, the current split is context-level and has sibling root-family overlap across train/dev/test, while old formal OOF was root-family-level. The 720:11 source imbalance further makes the 11 new rows only a smoke supplement, not a balanced final protocol.

## Posterior and root7703 diagnosis

The current posterior has stored test BCE/NLL `{(result.get('ensemble_test_metrics') or {{}}).get('bce_nll')}`, ECE `{(result.get('ensemble_test_metrics') or {{}}).get('ece_10bin')}`, and AUROC `{(result.get('ensemble_test_metrics') or {{}}).get('auroc')}`. These are diagnostic only: calibration is not valid for a formal claim because the current labels/split are invalid. Monotonicity is not valid: the frozen audit reports 2/74 contexts with adjacent decreases and 772 total decreases.

Root7703 is absent from old720. It is a separate integration/debug context. The 4N trace succeeds and the 4.09N trace fails despite identical frozen arm hashes and matched handoff, but the relevant realized force means are approximately {m4.get('aggregate_force',{}).get('relevant_mean')}N versus {m409.get('aggregate_force',{}).get('relevant_mean')}N. The second 4N directory is a byte-identical duplicate, so repeatability is not established. The correct diagnosis is unresolved execution-exposure/stochasticity, not a controller regression and not enough evidence for posterior-only attribution.

## Recommendation

Do not collect more contexts yet. First repair the row-level label construction, discard or explicitly quarantine the P4-B structural context reuse for a strict final claim, reconstruct the old root-family OOF split, and retrain/evaluate with a clean seen-task/held-out-root-context protocol. Keep root7703 only as integration smoke; do not use it as the final generalization test.

The detailed JSON/CSV evidence is in this directory. The required final evaluation is **seen tasks, held-out root/context**, with no task-OOD claim.
"""
    (OUT / "OLD720_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(OUT), "old_samples": len(rows), "old_contexts": len(by_ctx), "old_roots": len(by_root),
                      "old_tasks": old_tasks, "current_old_label_mismatches": len(cur_label_mismatch), "root7703_in_old": bool(root7703_in_old)}, indent=2))


if __name__ == "__main__":
    main()
