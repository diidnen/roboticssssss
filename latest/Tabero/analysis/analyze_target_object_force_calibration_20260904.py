#!/usr/bin/env python3
"""Read-only analysis for the isolated target-object force calibration runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/exouser/Tabero/E3_E6_E7_LANES")
R6 = ROOT / "TARGET_OBJECT_FORCE_CALIBRATION_20260904_r6"
STATIC = ROOT / "TARGET_OBJECT_FORCE_CALIBRATION_20260904_static_r2"
OUT = ROOT / "TARGET_OBJECT_FORCE_CALIBRATION_FINAL_20260904_r2"
E3 = Path("/home/exouser/E3_E6_E7_LANES/E3_FULLTASK_VS_LOCALLIFT")


def read_csv(p: Path) -> list[dict[str, str]]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict]) -> None:
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def x(r: dict, k: str) -> float:
    try:
        v = float(r.get(k, "nan"))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    out = np.empty(len(a), dtype=float); out[order] = np.arange(len(a), dtype=float)
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return corr(rank(a), rank(b))


def physical(r: dict) -> float:
    l, rr = x(r, "left_object_normal_N"), x(r, "right_object_normal_N")
    return 2.0 * min(l, rr) if np.isfinite(l) and np.isfinite(rr) and int(float(r.get("bilateral_object_contact", 0))) else np.nan


def summarise(rows: list[dict], phase: str, force: float, step_lo: int = -1) -> dict:
    q = [r for r in rows if abs(x(r, "requested_force_N") - force) < 1e-6 and r.get("phase") == phase and (step_lo < 0 or x(r, "step") >= step_lo) and np.isfinite(physical(r))]
    v = np.asarray([physical(r) for r in q], float)
    return {
        "requested_force_N": force, "phase": phase, "n_bilateral_samples": len(v),
        "bilateral_fraction": float(np.mean([int(float(r.get("bilateral_object_contact", 0))) for r in rows if abs(x(r, "requested_force_N") - force) < 1e-6 and r.get("phase") == phase])) if any(abs(x(r, "requested_force_N") - force) < 1e-6 and r.get("phase") == phase for r in rows) else np.nan,
        "physical_squeeze_mean_N": float(np.mean(v)) if len(v) else np.nan,
        "physical_squeeze_std_N": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
        "physical_squeeze_median_N": float(np.median(v)) if len(v) else np.nan,
        "physical_squeeze_p10_N": float(np.percentile(v, 10)) if len(v) else np.nan,
        "physical_squeeze_p90_N": float(np.percentile(v, 90)) if len(v) else np.nan,
        "physical_squeeze_peak_N": float(np.max(v)) if len(v) else np.nan,
    }


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    dyn = read_csv(R6 / "TARGET_OBJECT_FORCE_CALIBRATION_TIMESERIES.csv")
    sta = read_csv(STATIC / "TARGET_OBJECT_FORCE_CALIBRATION_TIMESERIES.csv")
    sta = [r for r in sta if r.get("source_action_index", "")]
    dyn_branches = [r for r in dyn if r.get("source_action_index", "")]
    forces = [2., 3., 4., 5., 6.]

    mapping = []
    for f in forces:
        s = summarise(sta, "STABLE_GRASP", f, step_lo=20)
        s["dataset"] = "same_state_static_hold"
        mapping.append(s)
    for f in forces:
        for ph in ("STABLE_GRASP", "LIFT", "TRANSPORT"):
            s = summarise(dyn_branches, ph, f, step_lo=5 if ph == "STABLE_GRASP" else -1)
            s["dataset"] = "same_state_policy_continuation"
            mapping.append(s)
    write_csv(OUT / "TABLE_B_SAME_STATE_CALIBRATION.csv", mapping)

    m = [r for r in mapping if r["dataset"] == "same_state_static_hold"]
    cmd = np.asarray([float(r["requested_force_N"]) for r in m], float)
    phys = np.asarray([float(r["physical_squeeze_mean_N"]) for r in m], float)
    slope, intercept = np.polyfit(cmd, phys, 1)
    fit = intercept + slope * cmd
    force_order = int(np.all(np.diff(phys) > 0))
    monotonic = {"within_state_spearman": spearman(cmd, phys), "strict_monotonicity": force_order, "linear_slope_N_per_command_N": float(slope), "linear_intercept_N": float(intercept), "mapping_mae_N": float(np.mean(abs(phys - cmd)),), "mapping_bias_N": float(np.mean(phys - cmd)), "mapping_rmse_N": float(np.sqrt(np.mean((phys - cmd) ** 2)))}
    (OUT / "TABLE_C_PHYSICAL_FORCE_MONOTONICITY.json").write_text(json.dumps(monotonic, indent=2) + "\n", encoding="utf-8")

    # Old formal squeeze vs target-object squeeze on the same isolated run.
    compare = []
    for r in dyn_branches:
        obj = physical(r); old = x(r, "controller_f_sq_meas_raw_N")
        compare.append({"requested_force_N": x(r, "requested_force_N"), "phase": r.get("phase", ""), "step": x(r, "step"), "old_canonical_squeeze_N": old, "target_object_bilateral_squeeze_N": obj, "difference_old_minus_object_N": old - obj if np.isfinite(old) and np.isfinite(obj) else np.nan, "left_object_normal_N": x(r, "left_object_normal_N"), "right_object_normal_N": x(r, "right_object_normal_N"), "bilateral_object_contact": r.get("bilateral_object_contact", "")})
    write_csv(OUT / "TABLE_A_OLD_NET_VS_OBJECT_SPECIFIC.csv", compare)
    valid = [(x(r, "old_canonical_squeeze_N"), x(r, "target_object_bilateral_squeeze_N")) for r in compare if np.isfinite(x(r, "old_canonical_squeeze_N")) and np.isfinite(x(r, "target_object_bilateral_squeeze_N"))]
    oa, ob = (np.asarray(valid).T if valid else (np.array([]), np.array([])))
    old_obj = {"n": len(valid), "pearson": corr(oa, ob) if valid else np.nan, "spearman": spearman(oa, ob) if valid else np.nan, "mae_old_minus_object_N": float(np.mean(abs(oa-ob))) if valid else np.nan, "bias_old_minus_object_N": float(np.mean(oa-ob)) if valid else np.nan}
    (OUT / "TABLE_A_OLD_NET_VS_OBJECT_SPECIFIC.json").write_text(json.dumps(old_obj, indent=2) + "\n", encoding="utf-8")

    # Measurement sanity counts and provenance.
    sanity = []
    for label, rs in (("replay_and_branches", dyn), ("static_hold_branches", sta)):
        sanity.append({"dataset": label, "rows": len(rs), "no_object_contact": sum(int(not int(float(r.get("left_object_contact", 0))) and not int(float(r.get("right_object_contact", 0)))) for r in rs), "left_only": sum(int(int(float(r.get("left_object_contact", 0))) and not int(float(r.get("right_object_contact", 0)))) for r in rs), "right_only": sum(int(int(float(r.get("right_object_contact", 0))) and not int(float(r.get("left_object_contact", 0)))) for r in rs), "bilateral": sum(int(float(r.get("bilateral_object_contact", 0))) for r in rs), "matrix_shape_observed": r.get("object_force_matrix_w", "")[:80] if rs else ""})
    write_csv(OUT / "MEASUREMENT_SANITY_COUNTS.csv", sanity)

    # Formal E3 selected-command evaluation is deliberately proxy-only: E3
    # logs do not contain target-object force matrix telemetry.
    utility_rows = []
    curves = E3 / "E3_ALL_FORCE_CURVES.csv"
    if curves.exists():
        cr = read_csv(curves)
        for method in sorted(set(r.get("method", "") for r in cr)):
            q = [r for r in cr if r.get("method") == method and r.get("utility_selected") == "1"]
            if not q:
                continue
            cmds = np.asarray([x(r, "force") for r in q], float)
            # calibrated 2--6 N map; outside range is explicitly extrapolated.
            proxy = intercept + slope * cmds
            utility_rows.append({"method": method, "n_selected": len(q), "selected_command_mean_N": float(np.mean(cmds)), "selected_command_std_N": float(np.std(cmds)), "physical_mean_force_proxy_N": float(np.mean(proxy)), "physical_mean_force_proxy_std_N": float(np.std(proxy)), "proxy_is_extrapolated": int(np.any((cmds < 2) | (cmds > 6))), "formal_target_object_measurement_available": 0, "interpretation": "offline proxy only; not direct physical evaluation of formal E3 branches"})
    utility_rows.extend([
        {"method": "Fixed-Max_8N", "n_selected": "not_run", "selected_command_mean_N": 8.0, "selected_command_std_N": 0.0, "physical_mean_force_proxy_N": float(intercept + slope*8), "physical_mean_force_proxy_std_N": 0.0, "proxy_is_extrapolated": 1, "formal_target_object_measurement_available": 0, "interpretation": "proxy only"},
        {"method": "ActiveForcing_existing_E3_utility", "n_selected": "see method rows", "selected_command_mean_N": "see source", "selected_command_std_N": "see source", "physical_mean_force_proxy_N": "not directly validated", "physical_mean_force_proxy_std_N": "", "proxy_is_extrapolated": "", "formal_target_object_measurement_available": 0, "interpretation": "requires target-object telemetry on selected formal rollouts; intentionally not collected this turn"},
    ])
    write_csv(OUT / "TABLE_D_UTILITY_PHYSICAL_PROXY_ONLY.csv", utility_rows)

    report = f"""# Target-object physical force calibration and Utility validation\n\nStatus: measurement-only; no formal benchmark, training, old720 data, evaluator, selector, or controller source was modified.\n\n## Measurement\n\nThe authoritative target-object measurement is `contact_grasp_black_book_1.data.force_matrix_w`, shape `(N, 2 finger bodies, 1 filtered target object, 3 world-force components)`, summed over the filter axis separately for `panda_leftfinger` and `panda_rightfinger`. Finger closing normals are the explicit `left_gripper_frame`/`right_gripper_frame` +Z axes; physical bilateral squeeze is `2*min(abs(F_left·n_left), abs(F_right·n_right))` only when both object contacts exceed 0.15 N.\n\nThe first verified bilateral contact is replay step 85. Same-state restore parity is 5/5. Direct reset is zero object contact.\n\n## Calibration\n\nStatic-hold mapping (steps 20--100) has strict command ordering `{force_order}` and within-state Spearman `{monotonic['within_state_spearman']:.3f}`. Fitted mapping is physical squeeze = `{slope:.3f} * command + {intercept:.3f}` N over 2--6 N. Absolute command-vs-physical MAE is `{monotonic['mapping_mae_N']:.3f}` N and bias is `{monotonic['mapping_bias_N']:.3f}` N.\n\nThis is a one-context, no-repeat calibration, so it validates the direction/range of the current controller at this grasp state, not a universal object/controller calibration.\n\n## Utility\n\n`TABLE_D_UTILITY_PHYSICAL_PROXY_ONLY.csv` is explicitly proxy-only. Existing formal E3 rollout logs have no target-object contact matrix, so this turn does not claim that the formal ActiveForcing selector has reduced actual object force, and Fixed-Max was not run. A direct success-versus-physical-force claim requires a measurement-instrumented selected-force rollout, which is outside the requested no-formal-benchmark scope.\n\n## Causal classification\n\n- FORCE_COMMAND_CAUSALLY_EFFECTIVE_BUT_ABSOLUTE_N_UNCALIBRATED\n- FORCE_RELATIVE_SIGNAL_VALID\n- TARGET_OBJECT_MEASUREMENT_VALIDATED_FOR_THIS ENV/SENSOR CONFIGURATION\n- UTILITY_PHYSICAL_FORCE_SAVING: NOT YET VALIDATED (formal logs lack object-specific force)\n\n## Provenance\n\n- Current action source: `source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py`\n- Current force action config: 13D action; left/right z force slots are indices 9 and 12; `d_cmd` is a gripper joint position target adjusted by force error.\n- Existing replay: `{R6 / 'TARGET_OBJECT_FORCE_CALIBRATION_TIMESERIES.csv'}`\n- Static calibration: `{STATIC / 'TARGET_OBJECT_FORCE_CALIBRATION_TIMESERIES.csv'}`\n"""
    (OUT / "TARGET_OBJECT_FORCE_CALIBRATION_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "CALIBRATION_ANALYSIS_MANIFEST.json").write_text(json.dumps({"status": "COMPLETE", "read_only_analysis": True, "source_r6": str(R6), "source_static": str(STATIC), "formal_e3_modified": False, "old720_modified": False, "controller_modified": False, "formal_benchmark_rerun": False, "static_mapping": monotonic, "utility_status": "PROXY_ONLY_NOT_PHYSICAL_FORMAL_E3_VALIDATION"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "monotonic": monotonic, "old_vs_object": old_obj, "utility_rows": utility_rows}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
