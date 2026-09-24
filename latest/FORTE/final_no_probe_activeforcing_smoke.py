#!/usr/bin/env python3
"""Offline selector adapter and final root7703 live-smoke result writer."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "analysis/results/final_no_probe_continuous_posterior_20260904"
SMOKE = OUT / "07_root7703_activeforcing_smoke"
FRONTIER = OUT / "ROOT7703_REFINED_FRONTIER.json"
CURVE = OUT / "ROOT7703_POSTERIOR_CURVE.json"
BASELINE_TRACE = ROOT / "analysis/results/root7703_reset_replay_reproduction_20260904_final_v3/LIVE_6N_TRACE.csv"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def force_label(force: float) -> str:
    value = f"{force:.2f}".rstrip("0").rstrip(".")
    return value.replace(".", "p") + "N"


def make_target() -> dict[str, Any]:
    curve = json.loads(CURVE.read_text())
    frontier = json.loads(FRONTIER.read_text())
    target = float(curve["selected_force_N"])
    idx = int(np.argmin(np.abs(np.asarray(curve["force_grid_N"]) - target)))
    model_path = (OUT / "FINAL_CHECKPOINT_PATH.txt").read_text().strip()
    target_artifact = {
        "schema": "FINAL_NO_PROBE_ACTIVEFORCING_TARGET_V1",
        "context_id": "root7703",
        "handoff_step": 95,
        "feature_schema": str(OUT / "FINAL_FEATURE_SCHEMA.json"),
        "model_checkpoint": model_path,
        "probe_feature_used": False,
        "no_requery": True,
        "selected_force_N": target,
        "requested_force_target_N": target,
        "controller_target_N": target,
        "predicted_success_probability": float(curve["posterior_success"][idx]),
        "expected_utility": float(curve["utility"][idx]),
        "utility_rule": curve["utility_rule"],
        "force_adapter": "identity absolute-Newton candidate into validated Tabero online controller",
        "empirical_frontier": frontier["frontier_interval"],
        "arm_trajectory_unchanged": True,
    }
    dump(SMOKE / "ACTIVEFORCING_TARGET.json", target_artifact)
    return target_artifact


def integrate(rows: list[dict[str, str]], first_lift: int | None) -> float | None:
    if first_lift is None:
        relevant = rows
    else:
        relevant = [r for r in rows if int(float(r["env_step"])) <= first_lift]
    if not relevant:
        return None
    return float(sum(float(r["aggregate_force"]) * 0.05 for r in relevant))


def finalize(isaac_dir: Path, target: dict[str, Any]) -> dict[str, Any]:
    result_path = isaac_dir / "ONLINE_FORCE_TRACKING_RESULT.json"
    result = json.loads(result_path.read_text())
    branches = result.get("branches", [])
    if isinstance(branches, dict):
        branches = list(branches.values())
    branch = next((b for b in branches if abs(float(b.get("requested_force", math.nan)) - target["selected_force_N"]) < 1e-5), None)
    if branch is None:
        raise RuntimeError(f"selected target {target['selected_force_N']} not found in {result_path}")
    label = force_label(target["selected_force_N"])
    trace_path = isaac_dir / f"LIVE_{label}_TRACE.csv"
    trace = read_rows(trace_path)
    baseline = read_rows(BASELINE_TRACE)
    arm_hash = [r.get("raw_arm_action_hash") for r in trace]
    baseline_hash = [r.get("raw_arm_action_hash") for r in baseline]
    arm_unchanged = arm_hash == baseline_hash
    handoff = json.loads((isaac_dir / "HANDOFF_PARITY.json").read_text())
    handoffs = handoff.get("handoffs", [])
    handoff_ok = bool(handoffs and all(
        h.get("physical_state_parity") and h.get("runtime_state_parity")
        and h.get("observation_parity") and h.get("bilateral_contact")
        for h in handoffs
    ))
    first_lift = branch.get("first_lift_step")
    realized = branch.get("mean_realized_force_during_relevant_window")
    fixed_frontier = json.loads(FRONTIER.read_text())
    fixed = {str(x["requested_force_N"]): x for x in fixed_frontier["candidates"]}
    fixed_low = fixed["2.0"]
    fixed_high = fixed["6.0"]
    exposure = integrate(trace, int(first_lift) if first_lift is not None else None)
    fixed_high_trace = read_rows(BASELINE_TRACE)
    high_exposure = integrate(fixed_high_trace, 118)
    active_result = {
        "schema": "ROOT7703_ACTIVEFORCING_RESULT_V1",
        "model_checkpoint": target["model_checkpoint"],
        "context_features": {
            "schema": str(OUT / "FINAL_FEATURE_SCHEMA.json"),
            "probe_feature_used": False,
            "query_count": 0,
        },
        "selected_force_N": target["selected_force_N"],
        "requested_force_N": target["requested_force_target_N"],
        "predicted_success_probability": target["predicted_success_probability"],
        "utility": target["expected_utility"],
        "realized_force_N": realized,
        "realized_force_trace": str(trace_path),
        "integrated_force_exposure_Ns_relevant_window": exposure,
        "hand_off_parity": handoff_ok,
        "raw_arm_hash_equal_to_fixed_high": arm_unchanged,
        "raw_arm_hash": hashlib.sha256("".join(str(x) for x in arm_hash).encode()).hexdigest(),
        "lift_success": bool(branch.get("lift_success")),
        "hold_30_step_success": bool(branch.get("hold_after_lift_success")),
        "contact_loss": bool(branch.get("contact_loss")),
        "slip": bool(branch.get("slip")),
        "drop": bool(branch.get("drop")),
        "first_lift_step": first_lift,
        "first_contact_loss_step": branch.get("first_contact_loss_step"),
        "max_object_height": branch.get("max_object_height"),
        "final_object_height": branch.get("final_object_height"),
        "outcome": branch.get("outcome"),
        "activeforcing_integration_valid": bool(handoff_ok and arm_unchanged),
        "final_no_probe_activeforcing_smoke": "PASS" if branch.get("hold_after_lift_success") and handoff_ok else "FAIL",
        "failure_class": None if branch.get("hold_after_lift_success") else "MODEL_OR_SELECTOR_ERROR",
    }
    dump(OUT / "ROOT7703_ACTIVEFORCING_RESULT.json", active_result)
    dump(OUT / "BASELINE_COMPARISON.json", {
        "protocol": "same root7703 HDF5 reset/replay/handoff; frozen arm; force only differs",
        "fixed_low_2N": {"requested_force_N": 2.0, "outcome": fixed_low["outcome"], "lift_success": fixed_low["lift_success"], "hold_30_step_success": fixed_low["hold_30_step_success"], "realized_force_N": fixed_low["mean_realized_force_during_relevant_window_N"]},
        "activeforcing": {"requested_force_N": target["selected_force_N"], "outcome": active_result["outcome"], "lift_success": active_result["lift_success"], "hold_30_step_success": active_result["hold_30_step_success"], "realized_force_N": realized, "integrated_force_exposure_Ns": exposure},
        "fixed_high_6N": {"requested_force_N": 6.0, "outcome": fixed_high["outcome"], "lift_success": fixed_high["lift_success"], "hold_30_step_success": fixed_high["hold_30_step_success"], "realized_force_N": fixed_high["mean_realized_force_during_relevant_window_N"], "integrated_force_exposure_Ns": high_exposure},
        "saving_vs_6N": {
            "commanded_abs_N": 6.0 - target["selected_force_N"],
            "commanded_percent": (6.0 - target["selected_force_N"]) / 6.0 * 100.0,
            "realized_abs_N": None if realized is None else fixed_high["mean_realized_force_during_relevant_window_N"] - realized,
            "realized_percent": None if realized is None else (fixed_high["mean_realized_force_during_relevant_window_N"] - realized) / fixed_high["mean_realized_force_during_relevant_window_N"] * 100.0,
            "integrated_exposure_abs_Ns": None if exposure is None or high_exposure is None else high_exposure - exposure,
            "integrated_exposure_percent": None if exposure is None or high_exposure is None else (high_exposure - exposure) / high_exposure * 100.0,
        },
    })
    report = f"""# Final no-probe ActiveForcing root7703 smoke

FINAL_NO_PROBE_ACTIVEFORCING_SMOKE = {active_result["final_no_probe_activeforcing_smoke"]}

Selected continuous target: {target["selected_force_N"]:.2f} N; predicted P(success)={target["predicted_success_probability"]:.4f}; utility={target["expected_utility"]:.4f}.

The context contains only the frozen VLA nominal arm prefix and task identity. Probe features, friction, outcomes, minimum force, future height, and future slip are excluded. Candidate force is an independent absolute-Newton input.

Handoff parity={handoff_ok}; raw arm trajectory unchanged={arm_unchanged}; lift={active_result["lift_success"]}; 30-step hold={active_result["hold_30_step_success"]}; outcome={active_result["outcome"]}.

The fixed 2N and fixed 6N baselines use the previously validated root7703 true-force traces. Commanded and realized-force savings are recorded in BASELINE_COMPARISON.json.
"""
    (OUT / "FINAL_ACTIVEFORCING_REPORT.md").write_text(report, encoding="utf-8")
    return active_result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-isaac", type=Path)
    args = parser.parse_args()
    target = make_target()
    if args.finalize_isaac:
        print(json.dumps(finalize(args.finalize_isaac, target), indent=2))
    else:
        print(json.dumps(target, indent=2))


if __name__ == "__main__":
    main()
