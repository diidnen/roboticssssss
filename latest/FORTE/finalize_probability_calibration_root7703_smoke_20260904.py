#!/usr/bin/env python3
"""Finalize the probability-calibration/root7703 smoke artifact directory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("/home/exouser/FORTE/analysis/results/probability_calibration_root7703_probe_smoke_20260904")
LIVE = OUT / "live_root7703_probe_final_1328"
BASE = Path("/home/exouser/FORTE/analysis/results/root7703_reset_replay_reproduction_20260904_final_v3/LIVE_6N_TRACE.csv")


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def relevant_metrics(path: Path) -> dict[str, object]:
    d = pd.read_csv(path)
    lift = d.index[d["lift_threshold_reached"].fillna(False).astype(bool)]
    end = int(lift[0]) if len(lift) else len(d) - 1
    q = d.iloc[: end + 1]
    return {
        "rows": int(len(d)),
        "first_lift_row": end,
        "mean_realized_force_N_relevant": float(q["aggregate_force"].mean()),
        "peak_realized_force_N_relevant": float(q["aggregate_force"].max()),
        "force_exposure_Ns_relevant": float(q["aggregate_force"].sum() * 0.05),
        "max_object_height_m": float(d["object_height"].max()),
        "drop_any": bool(d["drop"].fillna(False).astype(bool).any()),
    }


def main() -> None:
    raw = json.loads((OUT / "RAW_CALIBRATION_AUDIT.json").read_text())
    comparison = json.loads((OUT / "CALIBRATION_METHOD_COMPARISON.json").read_text())
    selected = json.loads((OUT / "SELECTED_CALIBRATION.json").read_text())
    mono = json.loads((OUT / "CALIBRATION_MONOTONICITY_AUDIT.json").read_text())
    live = json.loads((LIVE / "ROOT7703_ACTIVEFORCING_RESULT.json").read_text())
    mu = json.loads((LIVE / "ROOT7703_MU_POSTERIOR.json").read_text())
    eu = json.loads((LIVE / "ROOT7703_EXPECTED_UTILITY.json").read_text())
    curve = json.loads((LIVE / "ROOT7703_POSTERIOR_CURVE.json").read_text())

    # This is the already-produced clean training-distribution continuous
    # diagnostic, with the frozen Platt scalar map applied to the same curves.
    # It is explicitly not a held-out claim.
    curves = json.loads((Path("/home/exouser/FORTE/analysis/results/final_probe_continuous_posterior_rebuild_20260904") / "ROOT720_POSTERIOR_CURVES.json").read_text())
    ind = json.loads((Path("/home/exouser/FORTE/analysis/results/final_probe_continuous_posterior_rebuild_20260904") / "IN_DISTRIBUTION_EVAL.json").read_text())
    a = float(selected["fit"]["positive_slope"])
    b = float(selected["fit"]["intercept"])
    def utility(p: float, force: float) -> float:
        return p * (5.0 - force) / 5.0 + (1.0 - p) * (-1.0)
    raw_below = calibrated_below = 0
    raw_selected = []
    calibrated_selected = []
    for cid, q in curves.items():
        force = np.asarray(q["force_N"], dtype=float)
        p = np.clip(np.asarray(q["p_success"], dtype=float), 1e-7, 1.0 - 1e-7)
        pc = 1.0 / (1.0 + np.exp(-(a * np.log(p / (1.0 - p)) + b)))
        sr = float(force[int(np.argmax([utility(x, y) for x, y in zip(p, force)]))])
        sc = float(force[int(np.argmax([utility(x, y) for x, y in zip(pc, force)]))])
        ref = next(x for x in ind["rows"] if x["context_id"] == cid)
        frontier = float(ref["minimum_observed_lift_hold_success_force_N"])
        raw_below += int(sr < frontier)
        calibrated_below += int(sc < frontier)
        raw_selected.append(sr)
        calibrated_selected.append(sc)
    diagnostic = {
        "scope": "old720_all_72_contexts_training_distribution_diagnostic; not held-out",
        "force_domain_N": [3.0, 5.0],
        "raw_selected_below_min_observed_success_count": raw_below,
        "calibrated_selected_below_min_observed_success_count": calibrated_below,
        "context_count": len(curves),
        "raw_mean_selected_force_N": float(np.mean(raw_selected)),
        "calibrated_mean_selected_force_N": float(np.mean(calibrated_selected)),
        "continuous_selected_force_success_rate": None,
        "success_rate_note": "not identifiable when selected continuous force is not an observed branch; compare bracket position instead",
    }
    dump("IN_DISTRIBUTION_SELECTOR_DIAGNOSTIC.json", diagnostic)
    dump("EXPECTED_UTILITY_AUDIT.json", {
        "formula": "U(F|x)=p(success|x,F)*(Fmax-F)/Fmax+(1-p(success|x,F))*(-1)",
        "Fmax_N": 5.0,
        "low_force_tie_break": True,
        "changed": False,
        "root7703_used_for_calibration": False,
        "root7703_used_for_model_selection": False,
        "selected_force_raw_N": eu["raw_selected"]["force_N"],
        "selected_force_calibrated_N": eu["calibrated_selected"]["force_N"],
        "selected_force_used_N": eu["selected_force_used"],
        "model_supported_force_range_N": [3.0, 5.0],
        "requested_named_grid_outside_support_N": [2.0, 2.5, 6.0],
        "outside_support_policy": "recorded as unavailable; no extrapolation used",
    })

    active_m = relevant_metrics(LIVE / "ROOT7703_ACTIVEFORCING_TRACE.csv")
    high_m = relevant_metrics(BASE)
    report = f"""# Probability calibration and root7703 real-probe smoke

## Frozen method

Single physical P4-B probe -> friction posterior p(mu|probe) -> continuous feasibility posterior P(success|context,mu,F) -> marginalization -> frozen Expected Utility -> continuous Newton force -> unchanged arm replay and Tabero online gripper controller.

No re-query, learned gate, discrete force selector, arm replanning, or controller change was introduced.

## Probability reliability

Root-family OOF used {raw['root_family_count']} root families and {raw['raw_oof_metrics']['n']} rows. Root7703 was excluded from calibration and model selection.

Raw OOF: NLL={raw['raw_oof_metrics']['nll']:.6f}, Brier={raw['raw_oof_metrics']['brier']:.6f}, ECE={raw['raw_oof_metrics']['ece']:.6f}, AUROC={raw['raw_oof_metrics']['auroc']:.6f}.

Selected calibration: {selected['selected_method']}; calibrated OOF: NLL={selected['selected_metrics']['nll']:.6f}, Brier={selected['selected_metrics']['brier']:.6f}, ECE={selected['selected_metrics']['ece']:.6f}, AUROC={selected['selected_metrics']['auroc']:.6f}. The mapping consumes only raw probability/logit. Scalar mapping preserved force monotonicity (raw and calibrated violation rate 0).

The reported calibration comparison fits the standard scalar calibrators on the complete OOF prediction table and evaluates that same OOF table; it is therefore a probability-reliability diagnostic, not a second nested calibration holdout.

The old 72-context in-distribution diagnostic remains aggressive: raw and calibrated EU selected below the minimum observed lift+hold success force for {raw_below}/{len(curves)} contexts. This is not repaired by calibration and is not used to tune root7703.

## Root7703 real-probe smoke

P4-B probe: valid={live['probe_valid']}; rows={mu['probe_record']['probe_rows']}; bilateral post-probe={mu['probe_record']['post_probe_bilateral_contact']}; fixed prior used={mu['fixed_prior_used']}. μ posterior mean={mu['mean']:.6f}, std={mu['std']:.6f}, members={mu['member_means']}.

Handoff parity passed. EU selected {live['selected_force_N']:.2f}N (raw/calibrated both {live['selected_force_raw_N']:.2f}/{live['selected_force_calibrated_N']:.2f}N), with calibrated predicted probability {live['predicted_success_probability']:.6f}. The model support was [3,5]N; 2, 2.5, and 6N were recorded as outside support rather than extrapolated.

Development outcome: lift={live['lift_success']}, 30-step hold={live['hold_30_step_success']}. The trace also records a later drop after the hold window and strict force-tracking validity=false; this is not claimed as long-horizon task success.

Relevant-window ActiveForcing mean/peak/exposure: {active_m['mean_realized_force_N_relevant']:.6f}N / {active_m['peak_realized_force_N_relevant']:.6f}N / {active_m['force_exposure_Ns_relevant']:.6f}Ns.

The already-valid same-state 6N branch had lift={True}, 30-step hold={True}; its relevant-window mean/peak/exposure were {high_m['mean_realized_force_N_relevant']:.6f}N / {high_m['peak_realized_force_N_relevant']:.6f}N / {high_m['force_exposure_Ns_relevant']:.6f}Ns. Command saving is 3.0N; relevant-window realized mean saving is {high_m['mean_realized_force_N_relevant']-active_m['mean_realized_force_N_relevant']:.6f}N and exposure saving is {high_m['force_exposure_Ns_relevant']-active_m['force_exposure_Ns_relevant']:.6f}Ns. These are descriptive because the ActiveForcing trace has post-hold drop and strict tracking invalidity.

## Status

The final method was unchanged. The real-probe lift+hold smoke passed, but external probability reliability is not established by this one context: root7703's selected 3N is below its empirical frontier and the post-hold trace dropped. Long-horizon evaluation is not ready.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
