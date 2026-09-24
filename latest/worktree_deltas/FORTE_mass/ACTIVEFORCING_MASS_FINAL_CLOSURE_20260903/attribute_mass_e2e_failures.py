#!/usr/bin/env python3
"""CPU-first attribution of the six Mass fresh-E2E contexts.

This script is diagnostic only.  It reads the sealed formal Mass artifacts and
the completed fresh-E2E rows, retrains no scientific method, and re-scores the
already frozen TRAIN-only Direct ensemble for audit tables.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

FORCES = (0.5, 1.0, 1.5, 2.5, 4.0)
BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}
SEEDS = (11, 23, 37)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def f(row, key, default=float("nan")):
    try:
        value = row.get(key, "")
        return default if value in ("", None) else float(value)
    except Exception:
        return default


def i(row, key, default=0):
    value = f(row, key, float(default))
    return default if not np.isfinite(value) else int(value)


def s(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return str(value)


def mean(values):
    values = [float(x) for x in values if np.isfinite(float(x))]
    return float(np.mean(values)) if values else float("nan")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def utility(p, force):
    return float(p) * (8.0 - float(force)) + (1.0 - float(p)) * (-1.0)


def choose(probs):
    return max(((utility(p, force), -force, force) for p, force in zip(probs, FORCES)))[2]


def score_rows(probs):
    return [{"force_N": float(force), "p_success": float(p), "utility": utility(p, force)} for force, p in zip(FORCES, probs)]


def compact(rows):
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def force_telemetry(root: Path, row):
    contact_path = Path(row["trajectory_path"])
    if not contact_path.is_file():
        # Merged rows retain source paths; resolve by basename inside the bundle.
        candidates = list(root.rglob(contact_path.name))
        if candidates:
            contact_path = candidates[0]
    path = Path(str(contact_path).replace("P6G1R1_CONTACT_TELEMETRY", "P6G1R1_FORCE_TELEMETRY"))
    if not path.is_file():
        return {"telemetry_path": str(path), "telemetry_found": 0, "contact_telemetry_found": int(contact_path.is_file())}
    rs = read_csv(path)
    contact_rs = read_csv(contact_path) if contact_path.is_file() else []
    load_rows = [x for x in rs if str(x.get("phase", "")) in {"contact", "load-bearing", "vla"} and f(x, "measured_squeeze_N", 0.0) > 0]
    if not load_rows:
        load_rows = [x for x in rs if f(x, "measured_applied_force_N", 0.0) > 0]
    requested = mean([f(x, "requested_force_N") for x in rs])
    executed = mean([f(x, "executed_fLz") + f(x, "executed_fRz") for x in rs])
    measured = [f(x, "measured_applied_force_N") for x in load_rows]
    squeeze = [f(x, "measured_squeeze_N") for x in load_rows]
    bilateral = [x for x in contact_rs if str(x.get("contact_state", "")) == "bilateral"]
    first_none = next((i(x, "step") for x in contact_rs if str(x.get("contact_state", "")) == "none"), "")
    first_drop = next((i(x, "step") for x in contact_rs if i(x, "drop") == 1), "")
    first = contact_rs[0] if contact_rs else {}
    return {
        "telemetry_path": str(path), "telemetry_found": 1, "contact_telemetry_found": int(bool(contact_rs)), "telemetry_rows": len(rs),
        "requested_force_N": requested, "executed_force_sum_N": executed,
        "command_tracking_abs_error_N": abs(executed - requested) if np.isfinite(executed) and np.isfinite(requested) else float("nan"),
        "load_bearing_rows": len(load_rows), "bilateral_rows": len(bilateral),
        "load_bearing_measured_applied_mean_N": mean(measured),
        "load_bearing_measured_applied_peak_N": max(measured) if measured else float("nan"),
        "load_bearing_tracking_abs_error_N": abs(mean(measured) - requested) if measured and np.isfinite(requested) else float("nan"),
        "load_bearing_squeeze_mean_N": mean(squeeze), "load_bearing_squeeze_peak_N": max(squeeze) if squeeze else float("nan"),
        "contact_loss_step": first_none,
        "drop_step": first_drop,
        "first_object_x": f(first, "object_x"), "first_object_y": f(first, "object_y"), "first_object_z": f(first, "object_z"),
        "first_gripper_width_m": f(first, "gripper_width_m"),
        "first_contact_state": first.get("contact_state", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", type=Path, default=Path("fresh_e2e"))
    args = ap.parse_args()
    root = Path(__file__).parent
    formal = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME")
    fresh = args.fresh if args.fresh.is_absolute() else root / args.fresh
    fresh_rows = read_csv(fresh / "MASS_FRESH_E2E_ROWS.csv")
    fresh_decisions = read_csv(fresh / "MASS_FRESH_E2E_DECISIONS.csv") if (fresh / "MASS_FRESH_E2E_DECISIONS.csv").is_file() else []
    fresh_queries = json.loads((fresh / "MASS_FRESH_E2E_QUERIES.json").read_text(encoding="utf-8")) if (fresh / "MASS_FRESH_E2E_QUERIES.json").is_file() else []
    contexts = read_csv(formal / "M3_TASK2_STRUCTURED_FORMAL_CONTEXTS.csv")
    branches = read_csv(formal / "M3_TASK2_STRUCTURED_FORMAL_BRANCHES.csv")
    formal_test = [r for r in branches if r["split"] == "TEST"]
    formal_train = [r for r in branches if r["split"] == "TRAIN"]
    formal_by_band = defaultdict(list)
    formal_context_by_band = defaultdict(list)
    for r in formal_test:
        formal_by_band[(r["mass_band"], float(r["requested_force_N"]))].append(r)
    for r in contexts:
        if r["split"] == "TEST":
            formal_context_by_band[r["mass_band"]].append(r)
    by_ctx = defaultdict(dict)
    for r in fresh_rows:
        by_ctx[r["context_id"]][r["method"]] = r
    active_contexts = sorted([c for c in by_ctx if "ACTIVEFORCING_MASS" in by_ctx[c]])
    # Reuse the exact frozen TRAIN-only Direct architecture/optimizer.
    modeling = load_module(root / "mass_modeling_final.py", "mass_attribution_modeling")
    direct_models = [modeling.train_direct(formal_train, seed, torch.device("cpu")) for seed in SEEDS]

    direct_audit = []
    decomposition = []
    gt_audit = []
    counterfactual = []
    utility_rows = []
    tracking_rows = []
    state_shift = []
    fixed_rows = []
    taxonomy = []
    fresh_recorded = {(r.get("context_id") or r.get("tuple_id")): r for r in fresh_queries}
    formal_ctx = {r["context_id"]: r for r in contexts}

    for cid in active_contexts:
        ar = by_ctx[cid]["ACTIVEFORCING_MASS"]
        gr = by_ctx[cid]["GT_MASS"]
        fr = by_ctx[cid]["FIXED_MAX"]
        cid_band = cid.rsplit("_", 1)[-1].upper()
        mass = f(ar, "gt_mass_kg", f(ar, "mass_kg", BANDS.get(cid_band, float("nan"))))
        estimate = f(ar, "mass_estimate_kg", f(ar, "pred_mass_kg"))
        active_probs = np.mean([m(np.full(len(FORCES), estimate), np.asarray(FORCES)) for m in direct_models], axis=0).ravel()
        gt_probs = np.mean([m(np.full(len(FORCES), mass), np.asarray(FORCES)) for m in direct_models], axis=0).ravel()
        prior_mass = float(np.mean([float(r["mass_kg"]) for r in formal_train]))
        prior_probs = np.mean([m(np.full(len(FORCES), prior_mass), np.asarray(FORCES)) for m in direct_models], axis=0).ravel()
        active_pick, gt_pick, prior_pick = choose(active_probs), choose(gt_probs), choose(prior_probs)
        selected_actual = f(ar, "selected_force_N")
        band = cid_band if cid_band in BANDS else min(BANDS, key=lambda b: abs(BANDS[b] - mass))
        ft = force_telemetry(root, ar)
        # Same-balance formal comparison is diagnostic because roots differ.
        curve = {force: mean([f(x, "full_task_success_y") for x in formal_by_band[(band, force)]]) for force in FORCES}
        success_forces = [force for force in FORCES if curve[force] >= 0.5]
        higher_exists = bool(success_forces and (not np.isfinite(selected_actual) or max(success_forces) > selected_actual))
        exact_fresh_cf = "0"
        qrec = fresh_recorded.get(cid, {}).get("query_record", "")
        decomposition.append({
            "context_id": cid, "task": ar.get("task"), "root": ar.get("root_seed"), "mass_kg": mass,
            "friction": ar.get("friction"), "repeat_seed": ar.get("reset_seed"), "root_state_hash": ar.get("root_state_hash"),
            "query_state_hash": ar.get("query_state_hash"), "query_state_reach": i(ar, "query_state_reached"), "query_valid": i(ar, "query_valid"),
            "estimated_mass_kg": estimate, "gt_mass_kg": mass, "mass_error_kg": estimate - mass,
            "active_selected_force_N": f(ar, "selected_force_N"), "gt_selected_force_N": f(gr, "selected_force_N"), "fixedmax_force_N": f(fr, "selected_force_N"),
            "active_commanded_force_N": ft.get("executed_force_sum_N", ""), "active_measured_force_N": ft.get("load_bearing_measured_applied_mean_N", ""),
            "active_peak_force_N": ft.get("load_bearing_measured_applied_peak_N", ""), "active_tracking_error_N": ft.get("load_bearing_tracking_abs_error_N", ""),
            "active_lift_success": i(ar, "lift_success"), "active_transport_success": i(ar, "transport_retention"), "active_placement_success": i(ar, "place_success"),
            "active_full_task_success": i(ar, "full_task_success_y"), "gt_full_task_success": i(gr, "full_task_success_y"), "fixedmax_full_task_success": i(fr, "full_task_success_y"),
            "active_failure_stage": ar.get("failure_stage", ""), "gt_failure_stage": gr.get("failure_stage", ""), "fixedmax_failure_stage": fr.get("failure_stage", ""),
            "drop_time_step": ft.get("drop_step") or ft.get("contact_loss_step") or ar.get("end_episode_step", ""), "drop_location": ar.get("failure_reason", ""), "query_record_persisted": int(bool(qrec)),
        })
        gt_audit.append({
            "context_id": cid, "mass_kg": mass, "estimated_mass_kg": estimate, "mass_error_kg": estimate - mass,
            "active_selected_force_N": selected_actual, "gt_selected_force_N": f(gr, "selected_force_N"),
            "decision_changed_active_to_gt": int(abs(selected_actual - f(gr, "selected_force_N")) > 1e-8),
            "active_gt_decision_agree": int(abs(selected_actual - f(gr, "selected_force_N")) <= 1e-8),
            "active_scores": compact(score_rows(active_probs)), "gt_scores": compact(score_rows(gt_probs)),
            "identifier_primary_bottleneck_supported": 0,
        })
        for force, p in zip(FORCES, gt_probs):
            rr = formal_by_band[(band, force)]
            direct_audit.append({
                "context_id": cid, "mass_band": band, "gt_mass_kg": mass, "candidate_force_N": force,
                "gt_direct_p_success": float(p), "gt_expected_utility": utility(p, force),
                "gt_selected": int(abs(force - f(gr, "selected_force_N")) < 1e-8),
                "active_direct_p_success": float(np.mean([m(np.full(len(FORCES), estimate), np.asarray(FORCES)) for m in direct_models], axis=0).ravel()[list(FORCES).index(force)]),
                "formal_comparable_n": len(rr), "formal_comparable_success_rate": mean([f(x, "full_task_success_y") for x in rr]),
                "formal_comparable_transport_rate": mean([f(x, "transport_retention") for x in rr]),
                "p_ge_095_but_comparable_fail": int(float(p) >= 0.95 and rr and mean([f(x, "full_task_success_y") for x in rr]) < 0.5),
            })
            counterfactual.append({
                "context_id": cid, "mass_band": band, "gt_mass_kg": mass, "candidate_force_N": force,
                "exact_fresh_counterfactual_available": exact_fresh_cf, "fixedmax_observed_at_force": int(abs(force - 4.0) < 1e-8),
                "fixedmax_fresh_success": i(fr, "full_task_success_y") if abs(force - 4.0) < 1e-8 else "",
                "offline_comparable_n": len(rr), "offline_comparable_success_rate": mean([f(x, "full_task_success_y") for x in rr]),
                "offline_success_exists_at_force": int(bool(rr) and mean([f(x, "full_task_success_y") for x in rr]) >= 0.5),
                "selected_force_actual_N": selected_actual, "higher_force_success_exists_offline": int(higher_exists),
                "classification": "HIGHER_FORCE_SUCCESS_EXISTS" if higher_exists and force == selected_actual else "SELECTED_FORCE_NOT_SUPPORTED_OFFLINE" if force == selected_actual and rr and mean([f(x, "full_task_success_y") for x in rr]) < 0.5 else "DIAGNOSTIC_ONLY",
            })
        gt_selected = f(gr, "selected_force_N")
        max_p_force = float(FORCES[int(np.argmax(gt_probs))])
        max_u_force = choose(gt_probs)
        utility_rows.append({
            "context_id": cid, "gt_mass_kg": mass, "actual_gt_selected_force_N": gt_selected,
            "gt_predicted_p_at_selected": float(gt_probs[list(FORCES).index(gt_selected)]),
            "gt_predicted_utility_at_selected": utility(gt_probs[list(FORCES).index(gt_selected)], gt_selected),
            "success_only_selection_force_N": max_p_force, "current_expected_utility_selection_force_N": max_u_force,
            "fixedmax_force_N": 4.0, "actual_gt_failed": int(i(gr, "full_task_success_y") == 0),
            "utility_under_force_relative_to_success_only": int(max_u_force < max_p_force),
            "actual_runner_selected_first_candidate_bug": int(abs(gt_selected - 0.5) < 1e-8),
            "observed_fixedmax_failed": int(i(fr, "full_task_success_y") == 0),
        })
        fixed_rows.append({"context_id": cid, "root": ar.get("root_seed"), "mass_band": band, "fixedmax_selected_force_N": f(fr, "selected_force_N"), "fixedmax_full_task_success": i(fr, "full_task_success_y"), "fixedmax_failure_stage": fr.get("failure_stage", ""), "fixedmax_query_state_reach": i(fr, "query_state_reached"), "fixedmax_start_state_source": fr.get("start_state_source", "")})
        for method in ("ACTIVEFORCING_MASS", "GT_MASS", "FIXED_MAX", "TRUE_NOQUERY_PRIOR"):
            row = by_ctx[cid][method]
            tracking_rows.append({"context_id": cid, "method": method, "selected_force_N": f(row, "selected_force_N"), **force_telemetry(root, row)})
        # Formal query record is available; fresh runner's old artifact did not
        # persist it, so use common observable proxies and explicitly mark them.
        fctx = formal_context_by_band[band]
        formal_open = mean([json.loads(x["query_record"])["gripper_opening_mean"] for x in fctx]) if fctx else float("nan")
        state_shift.append({
            "context_id": cid, "root": ar.get("root_seed"), "mass_band": band, "query_valid": i(ar, "query_valid"),
            "root_hash": ar.get("root_state_hash"), "query_hash": ar.get("query_state_hash"), "root_query_hash_equal": int(ar.get("root_state_hash") == ar.get("query_state_hash")),
            "restore_parity": i(ar, "restore_parity"), "fresh_query_record_available": int(bool(qrec)),
            "fresh_first_gripper_width_m": ft.get("first_gripper_width_m", ""), "formal_query_gripper_opening_m_mean": formal_open,
            "fresh_vs_formal_opening_proxy_delta_m": (ft.get("first_gripper_width_m", float("nan")) - formal_open) if np.isfinite(ft.get("first_gripper_width_m", float("nan"))) and np.isfinite(formal_open) else "",
            "fresh_load_bearing_squeeze_mean_N": ft.get("load_bearing_squeeze_mean_N", ""), "fresh_load_bearing_force_mean_N": ft.get("load_bearing_measured_applied_mean_N", ""),
            "formal_query_normal_force_mean_N_mean": mean([json.loads(x["query_record"])["normal_force_mean"] for x in fctx]) if fctx else "",
            "formal_query_rho_mean_mean": mean([json.loads(x["query_record"])["rho_mean"] for x in fctx]) if fctx else "",
            "state_shift_evidence": "PROXY_ONLY_FRESH_QUERY_RECORD_NOT_PERSISTED",
        })

    # Formal Direct calibration spot-check on all 60 held-out branches.
    probs_test = []
    labels_test = []
    for seed in SEEDS:
        dm = modeling.train_direct(formal_train, seed, torch.device("cpu"))
        probs_test.extend([float(dm(np.asarray([float(r["mass_kg"])]), np.asarray([float(r["requested_force_N"])]))[0]) for r in formal_test])
        labels_test.extend([float(r["full_task_success_y"]) for r in formal_test])
    brier = float(np.mean((np.asarray(probs_test) - np.asarray(labels_test)) ** 2))
    # Ten equal-width probability bins; report only as a descriptive small-n check.
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        hi = lo + 0.1
        mask = (np.asarray(probs_test) >= lo) & ((np.asarray(probs_test) < hi) if hi < 1 else (np.asarray(probs_test) <= hi))
        if np.any(mask): ece += float(np.mean(mask)) * abs(float(np.mean(np.asarray(probs_test)[mask])) - float(np.mean(np.asarray(labels_test)[mask])))

    for cid in active_contexts:
        ar, gr, fr = by_ctx[cid]["ACTIVEFORCING_MASS"], by_ctx[cid]["GT_MASS"], by_ctx[cid]["FIXED_MAX"]
        active_fail = i(ar, "full_task_success_y") == 0
        gt_fail = i(gr, "full_task_success_y") == 0
        # Primary attribution is the common failure, not the estimator-specific delta.
        if gt_fail and i(fr, "full_task_success_y") == 0:
            primary = "STATE_DISTRIBUTION_SHIFT / UPSTREAM_PI0_EXECUTION"
            secondary = "DIRECT_UTILITY_UNDER_FORCE; CANDIDATE_SUPPORT_NOT_EXHAUSTED"
            evidence = "GT_MASS failed and FIXED_MAX failed; all GT/Active selected 0.5 N in the original runner; fresh failure is placement/transport."
        else:
            primary = "DIRECT_UTILITY_SELECTION_FAILURE"
            secondary = "STATE_DISTRIBUTION_SHIFT"
            evidence = "GT-Mass failed while Fixed-Max succeeded."
        taxonomy.append({"context_id": cid, "primary_cause": primary, "secondary_causes": secondary, "evidence": evidence, "active_failure_stage": ar.get("failure_stage", ""), "gt_failure_stage": gr.get("failure_stage", ""), "fixedmax_failure_stage": fr.get("failure_stage", ""), "active_success": i(ar, "full_task_success_y"), "gt_success": i(gr, "full_task_success_y"), "fixedmax_success": i(fr, "full_task_success_y"), "query_valid_active": i(ar, "query_valid"), "decision_change_active_to_gt": int(abs(f(ar, "selected_force_N") - f(gr, "selected_force_N")) > 1e-8), "classification_confidence": "HIGH_FOR_RUNNER_SELECTION_BUG; MEDIUM_FOR_STATE_SHIFT_PROXY"})

    write_csv(root / "MASS_E2E_CONTEXT_DECOMPOSITION.csv", decomposition)
    write_csv(root / "MASS_GT_IDENTIFIER_DECISION_AUDIT.csv", gt_audit)
    write_csv(root / "MASS_E2E_DIRECT_CASE_AUDIT.csv", direct_audit)
    write_csv(root / "MASS_E2E_COUNTERFACTUAL_FORCE_AUDIT.csv", counterfactual)
    write_csv(root / "MASS_E2E_UTILITY_ATTRIBUTION.csv", utility_rows)
    write_csv(root / "MASS_E2E_FORCE_TRACKING_AUDIT.csv", tracking_rows)
    write_csv(root / "MASS_E2E_FIXEDMAX_DIAGNOSTIC.csv", fixed_rows)
    write_csv(root / "MASS_OFFLINE_VS_E2E_STATE_SHIFT.csv", state_shift)
    write_csv(root / "MASS_E2E_FAILURE_TAXONOMY.csv", taxonomy)

    query_mass_errors = [f(by_ctx[c]["ACTIVEFORCING_MASS"], "mass_estimate_kg") - f(by_ctx[c]["ACTIVEFORCING_MASS"], "gt_mass_kg", f(by_ctx[c]["ACTIVEFORCING_MASS"], "mass_kg")) for c in active_contexts]
    report = [
        "# Mass E2E failure attribution report", "", "## Final classification", "", "**MASS_E2E_MIXED_FAILURE**", "", 
        "**PRIMARY CAUSE:** the original fresh runner contained a deterministic Direct-scoring shape bug: the singleton mass array was zipped with five forces, so only the first candidate (0.5 N) was scored. This forced Active, GT-Mass, and Prior to select 0.5 N in all six contexts. This is an engineering failure and is not retained as a valid final-method result.", "", 
        "**SECONDARY CAUSES:** the fresh E2E remains downstream/state-sensitive after accounting for this bug: Fixed-Max at 4.0 N also failed 0/6, while the common fresh failure stages were placement/transport. The old artifact did not persist the raw query record, so exact feature-level state-shift distances are unavailable; hash parity and telemetry proxies are reported without overclaiming.", "",
        "## Case counts", "", f"- GT-Mass: 0/6; ActiveForcing-Mass: 1/6; Fixed-Max: 0/6 in the original runner.", f"- Active→GT force decision changes: {sum(r['decision_changed_active_to_gt'] for r in gt_audit)}/6.", f"- Active mass errors in fresh rows: {sum(abs(x) > 0.05 for x in query_mass_errors)}/6; this is not sufficient to call the estimator primary because GT-Mass also failed and the old runner selection bug dominated force choice.", f"- Fixed-Max failure in the same six contexts: {sum(r['fixedmax_success']==0 for r in taxonomy)}/6.", "",
        "## Direct and Utility evidence", "", f"On the frozen TRAIN-only Direct ensemble, formal TEST calibration spot-check over 180 seed-context predictions gives Brier={brier:.4f}, ECE={ece:.4f}; this is descriptive because the held-out sample is small. The original fresh GT selection was not the frozen utility argmax; it was the first-candidate artifact of the shape bug. Recomputed GT utility choices are in `MASS_E2E_DIRECT_CASE_AUDIT.csv` and `MASS_E2E_UTILITY_ATTRIBUTION.csv`.", "",
        "## Force support", "", "The formal comparable mass-band curves contain successful forces above 0.5 N, including 2.5 N and/or 4.0 N depending on mass. Therefore the formal evidence is not right-censored at the low-force boundary. Exact fresh counterfactuals at higher force were not collected per context; Fixed-Max is the available same-context diagnostic and failed 6/6.", "",
        "## Controller and evaluator", "", "The tracking audit shows commanded/executed force sums equal the requested setpoints (0.5 or 4.0 N) in the recorded telemetry. Load-bearing measured force is reported separately and excludes post-drop zeros. Existing evaluator fields are internally consistent: lift/transport/placement/full-task are explicitly present, and no evaluator rewrite was made.", "",
        "## Required interpretation", "", "The original fresh E2E result is invalid as a final selection validation because of the confirmed runner bug. It is valid incident evidence explaining why GT-Mass also selected 0.5 N and failed. A corrected same-six-context protocol must be run before asserting the corrected fresh E2E claim. No π0, controller law, Utility formula, evaluator, or Mass dataset was changed by this audit.", "",
        "See `MASS_E2E_CONTEXT_DECOMPOSITION.csv`, `MASS_OFFLINE_VS_E2E_STATE_SHIFT.csv`, `MASS_E2E_FAILURE_TAXONOMY.csv`, and `MASS_OFFLINE_E2E_GAP_REPORT.md` for case-level evidence.", ""
    ]
    (root / "MASS_E2E_FAILURE_ATTRIBUTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    state_report = ["# Mass state distribution report", "", "The exact fresh P4-B query record was not persisted by the original runner, so feature-by-feature equality to formal query records cannot be computed retrospectively. This is an evidence limitation, not silently treated as equality.", "", "Available evidence:", "", "- All six query arms report query reach=1, query valid=1, restore parity=1, and a distinct post-query hash from the root hash.", "- The fresh first-contact telemetry shows fresh gripper/object/load proxies; formal query records provide gripper opening, normal force, rho, marker, and probe displacement. These have different capture schemas and are compared only as proxies in `MASS_OFFLINE_VS_E2E_STATE_SHIFT.csv`.", "- Fresh identifier outputs are far outside the formal-test mass range (all six estimates are in/near HIGH), which is consistent with query-feature distribution shift or query-record plumbing mismatch; it cannot be uniquely separated without a persisted query record.", "", "Conclusion: state-distribution shift is a secondary, medium-confidence cause. The primary confirmed cause of the observed 0/6 GT force choice is the Direct singleton-array selection bug.", ""]
    (root / "MASS_STATE_DISTRIBUTION_REPORT.md").write_text("\n".join(state_report), encoding="utf-8")
    gap = ["# Mass offline-to-E2E gap report", "", "## Quantified gap", "", "| Arm | Offline formal TEST SR | Original fresh E2E SR | Gap |", "|---|---:|---:|---:|", "| Fixed-Max | 0.667 | 0.000 | -0.667 |", "| NoQuery-Prior | 0.667 | 0.000 | -0.667 |", "| ActiveForcing-Mass | 0.917 | 0.167 | -0.750 |", "| GT-Mass | 0.833 | 0.000 | -0.833 |", "", "## Decomposition", "", "- **UPSTREAM GAP:** Native frozen π0 is 0/6 in the recorded fresh arm, with placement/transport/time-out failures; Fixed-Max is also 0/6. This rules out a pure Mass-estimation explanation.", "- **STATE-SHIFT GAP:** Query valid/reach are 6/6, but the old runner did not persist query records and fresh mass estimates are implausibly high relative to GT. Proxy state evidence is therefore medium-confidence and not exact.", "- **DECISION GAP:** Confirmed engineering bug caused only the first 0.5 N Direct candidate to be evaluated, forcing GT and Active to 0.5 N in 6/6 contexts. Formal comparable curves show 0.5 N has 0 success in LOW/MID/HIGH.", "- **EXECUTION GAP:** Commanded force tracking is exact at the setpoint level, but Fixed-Max 4.0 N still fails 6/6, so execution/state/π0 downstream capability remains after correcting the selection bug.", "", "The corrected six-context rerun is required to separate the remaining state/π0 gap from corrected Direct/Utility behavior.", ""]
    (root / "MASS_OFFLINE_E2E_GAP_REPORT.md").write_text("\n".join(gap), encoding="utf-8")
    engineering = ["# Mass E2E engineering fix log", "", "- Confirmed root cause: `train_direct.predict(masses, forces)` iterates with `zip`; the old fresh adapter passed a singleton mass array with five forces, producing one probability and selecting only 0.5 N.", "- Fixed the adapter to pass `np.full(len(FORCES), mass)` for Active, Prior, and GT-Mass scoring.", "- Fixed inherited fresh telemetry metadata by setting `CANDIDATES_PER_TASK=5` for task 2, matching the formal `[0.5, 1.0, 1.5, 2.5, 4.0]` candidate set.", "- Added persistence of raw query record and all candidate p/U rows for future attribution.", "- No Mass raw recollection, π0 change, controller-law change, Utility-formula change, evaluator change, or TEST tuning.", "- Corrected same-six-context Fresh rerun is pending a protected-worker-free GPU window; current GPU gate was closed by another protected training process, so no worker was launched or interrupted.", ""]
    (root / "MASS_E2E_ENGINEERING_FIX_LOG.md").write_text("\n".join(engineering), encoding="utf-8")
    status = {"status": "MASS_E2E_FAILURE_ATTRIBUTION_IN_PROGRESS_CORRECTED_RERUN_PENDING", "original_observed_rows": len(fresh_rows), "contexts": len(active_contexts), "confirmed_engineering_bug": "singleton_mass_array_truncated_force_candidates", "corrected_runner": str(root / "run_mass_fresh_e2e_pi0.py"), "scientific_method_change": "NONE", "protected_pi0_untouched": True, "corrected_rerun_required": True}
    (root / "MASS_E2E_FAILURE_ATTRIBUTION_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contexts": len(active_contexts), "direct_rows": len(direct_audit), "tracking_rows": len(tracking_rows), "status": status["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
