#!/usr/bin/env python3
"""Finalize E7 only after both real DEV execution stages have passed."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import activeforcing_e7_continuous_repair as repair


DEFAULT_OUT = repair.OUT


def md_table(frame: pd.DataFrame) -> str:
    return repair.markdown_table(frame, list(frame.columns))


def require_pass(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"required execution audit missing: {path}")
    value = json.loads(path.read_text())
    if value.get("status") != "PASS":
        raise SystemExit(f"execution audit is not PASS: {path}")
    return value


def paired_interval(values: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, float)
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(samples, 0.05)), float(np.quantile(samples, 0.95))


def finalize(out: Path) -> None:
    small_audit = require_pass(out / "E7_OFFGRID_DEV_EXECUTION_AUDIT.json")
    full_audit = require_pass(out / "E7_MATCHED_FULL_DEV_EXECUTION_AUDIT.json")
    gate = json.loads((out / "CONTINUOUS_DIRECT_VALID_GATE.json").read_text())
    if gate.get("status") != "PASS":
        raise SystemExit("Direct gate is not PASS")
    rule = json.loads((out / "E7_FINAL_DECISION_RULE.json").read_text())
    if not rule.get("frozen_before_real_matched_DEV_outcomes"):
        raise SystemExit("final decision rule was not frozen")

    offline = pd.read_csv(out / "TABLE_E7_CONTINUOUS_PLANNER_OFFLINE_FROZEN.csv")
    realized = pd.read_csv(out / "E7_MATCHED_FULL_DEV_RESULTS.csv")
    keys = ["context_id", "planning", "planner", "K"]
    if len(offline) != 144 or len(realized) != 144 or realized[keys].duplicated().any():
        raise SystemExit(f"matched matrix incomplete: offline={len(offline)} realized={len(realized)}")
    keep = keys + ["requested_force_N", "controller_setpoint_N", "controller_exact_float_match",
                   "measured_loadbearing_force_N", "tracking_error_N", "final_success", "failure_stage",
                   "state_parity", "telemetry_path", "telemetry_sha256"]
    merged = offline.drop(columns=["measured_loadbearing_force_N", "realized_SR", "realized_utility", "utility_regret"])
    merged = merged.merge(realized[keep], on=keys, how="left", validate="one_to_one", suffixes=("", "_executed"))
    if merged.final_success.isna().any() or not merged.controller_exact_float_match.all() or not merged.state_parity.eq(1).all():
        raise SystemExit("matched result identity/parity validation failed")
    if not np.allclose(merged.requested_force_N, merged.requested_force_N_executed, rtol=0.0, atol=0.0):
        raise SystemExit("planned/executed force mismatch")
    merged["realized_SR"] = merged.final_success.astype(float)
    merged["realized_utility"] = [
        y * (1.0 - f / repair.TASK_FMAX[int(t)]) + (1.0 - y) * -1.0
        for y, f, t in zip(merged.realized_SR, merged.requested_force_N, merged.task)
    ]
    best = merged.groupby(["context_id", "planning", "K"]).realized_utility.transform("max")
    merged["utility_regret"] = best - merged.realized_utility
    merged.to_csv(out / "TABLE_E7_CONTINUOUS_PLANNER.csv", index=False)

    summary = merged.groupby(["planning", "planner", "K"], as_index=False).agg(
        n_contexts=("context_id", "nunique"), mean_selected_force_N=("selected_force_N", "mean"),
        mean_measured_force_N=("measured_loadbearing_force_N", "mean"),
        mean_tracking_MAE_N=("tracking_error_N", "mean"), realized_SR=("realized_SR", "mean"),
        mean_realized_utility=("realized_utility", "mean"), mean_utility_regret=("utility_regret", "mean"),
        mean_predicted_utility=("predicted_expected_utility", "mean"),
        strict_under_force_rate=("under_force", "mean"), mean_latency_ms=("latency_ms_total_cpu", "mean"),
    )

    decision_rows = []
    for k in repair.K_VALUES:
        q = merged[(merged.planning == "POSTERIOR") & (merged.K == k)]
        utility = q.pivot(index="context_id", columns="planner", values="realized_utility")
        success = q.pivot(index="context_id", columns="planner", values="realized_SR")
        du = (utility.PROPOSAL_GUIDED - utility.FIXED_GRID).to_numpy(float)
        ds = (success.PROPOSAL_GUIDED - success.FIXED_GRID).to_numpy(float)
        ulo, uhi = paired_interval(du, int(rule["bootstrap"]["seed"]) + int(k))
        slo, shi = paired_interval(ds, int(rule["bootstrap"]["seed"]) + 100 + int(k))
        decision_rows.append({"K": int(k), "n_paired_contexts": len(du),
                              "proposal_minus_fixed_mean_utility": float(np.mean(du)),
                              "utility_delta_bootstrap90_lo": ulo, "utility_delta_bootstrap90_hi": uhi,
                              "proposal_minus_fixed_mean_SR": float(np.mean(ds)),
                              "SR_delta_bootstrap90_lo": slo, "SR_delta_bootstrap90_hi": shi})
    decision = pd.DataFrame(decision_rows)
    support = bool(((decision.proposal_minus_fixed_mean_utility >= 0.02) &
                    (decision.utility_delta_bootstrap90_lo >= 0.0) &
                    (decision.proposal_minus_fixed_mean_SR >= -0.10)).all())
    negative = bool((((decision.proposal_minus_fixed_mean_utility <= -0.02) &
                      (decision.utility_delta_bootstrap90_hi <= 0.0)).all()) or
                    (decision.proposal_minus_fixed_mean_SR < -0.10).any())
    classification = "E7_SUPPORTED" if support else ("E7_NEGATIVE" if negative else "E7_NEUTRAL")
    final_status = ("E7_COMPLETE_NEGATIVE" if classification == "E7_NEGATIVE"
                    else "E7_CONTINUOUS_PLANNING_SCIENTIFICALLY_VALIDATED")

    (out / "TABLE_E7_CONTINUOUS_PLANNER.md").write_text(
        "# E7 Continuous Planner — Real Matched DEV\n\n" + md_table(summary) +
        "\n\n## Frozen primary decision\n\n" + md_table(decision) +
        f"\n\nClassification: **{classification}**. Final status: **{final_status}**.\n")
    (out / "E7_FINAL_DECISION.json").write_text(json.dumps({
        "classification": classification, "final_status": final_status,
        "decision_rows": decision_rows, "rule": rule, "sealed_TEST_used": False,
    }, indent=2, sort_keys=True) + "\n")
    (out / "E7_COMPLETION_AUDIT.md").write_text(f"""# E7 Completion Audit

- Direct gate: PASS.
- Small exact-float execution: PASS ({small_audit['observed_rollouts']}/{small_audit['expected_rollouts']}).
- Full matched execution: PASS ({full_audit['observed_rollouts']}/{full_audit['expected_rollouts']}).
- Planned/executed bit-exact float identity: PASS.
- Branch state parity: PASS.
- Full matrix: 144/144, 9 DEV contexts, POINT/POSTERIOR, four planners, K=5/10.
- Expected Utility used; hard-rho selector absent.
- Sealed TEST outcomes read: no.
- Classification: {classification}.
- Final status: {final_status}.
""")
    (out / "ACTIVEFORCING_E7_FINAL_REPORT.md").write_text(f"""# ACTIVEFORCING E7 Continuous Planner Repair and Validation

## Final status: {final_status}

Scientific classification: **{classification}**.

The invalid legacy E7 result was not reused as final evidence. This closure uses 720 native-float TRAIN branches, contact/load-bearing-only actuator calibration, genuine root/force holdouts, a monotone calibrated Direct gate, distinct matched-budget generators, and real exact-float DEV execution. The authoritative Expected Utility was used throughout; no hard-rho selector was used and sealed TEST was not read.

## Direct gate

- Alternating-anchor probability MAE: {gate['checks']['alternating_anchor_probability_mae']:.4f}.
- Held-out-root Brier: {gate['checks']['heldout_root_brier']:.4f}.
- DEV point probability MAE: {gate['checks']['dev_point_probability_mae']:.4f}.
- Strict posterior-planner DEV under-force: {gate['checks']['dev_runtime_under_force_rate']:.4f}.
- Monotonic context rate: {gate['checks']['dev_monotonic_context_rate']:.4f}.

## Real planner evidence

{md_table(decision)}

The primary rule and bootstrap seed were frozen before matched rollout outcomes. With only nine paired DEV contexts, uncertainty is reported explicitly; the result is not promoted beyond the frozen support/neutral/negative rule. Full per-action evidence is in `TABLE_E7_CONTINUOUS_PLANNER.csv`, and command/measured-force parity remains inspectable through every telemetry SHA-256.
""")
    print(json.dumps({"classification": classification, "final_status": final_status}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    finalize(args.output)


if __name__ == "__main__":
    main()
