#!/usr/bin/env python3
"""Write the auditable pilot-status handoff for the current final-method run."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "analysis/results/final_no_probe_continuous_posterior_20260904"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    context = load(OUT / "VALID_CONTEXT_MANIFEST.json")
    frontier = load(OUT / "ROOT7703_REFINED_FRONTIER.json")
    training = load(OUT / "05_training/POSTERIOR_TRAINING_RESULT.json")
    target = load(OUT / "07_root7703_activeforcing_smoke/ACTIVEFORCING_TARGET.json")
    live = load(OUT / "ROOT7703_ACTIVEFORCING_RESULT.json")
    baseline = load(OUT / "BASELINE_COMPARISON.json")
    pilot = {
        "schema": "FINAL_NO_PROBE_PILOT_EVAL_V1",
        "status": "BLOCKED_INSUFFICIENT_HELDOUT_CONTEXTS",
        "heldout_pilot_contexts_requested": [3, 5],
        "heldout_pilot_contexts_available": 0,
        "valid_context_count": context["valid_context_count"],
        "reason": "Only root7703 and root7704 passed the current admission rule; a 3-5 held-out pilot would not be scientifically valid.",
        "fixed_low_2N_success_rate": None,
        "fixed_high_6N_success_rate": None,
        "activeforcing_success_rate": None,
        "next_action": "admit additional fresh contexts only after exact handoff and native/force continuation parity passes",
    }
    dump(OUT / "08_pilot_eval/PILOT_EVAL_RESULT.json", pilot)
    dump(OUT / "FINAL_METHOD_STATUS.json", {
        "FINAL_STATUS": "PARTIAL_FINAL_METHOD_SMOKE_FAIL",
        "ROOT7703_REFINED_FRONTIER": frontier["frontier_interval"],
        "VALID_CONTEXT_COUNT": context["valid_context_count"],
        "TRUE_FORCE_SAMPLE_COUNT": load(OUT / "TRUE_FORCE_DATASET_MANIFEST.json")["sample_count"],
        "OLD_DATA_REUSED_COUNT": load(OUT / "TRUE_FORCE_DATASET_MANIFEST.json")["old_reused_sample_count"],
        "NEW_TRUE_FORCE_SAMPLE_COUNT": load(OUT / "TRUE_FORCE_DATASET_MANIFEST.json")["new_true_force_sample_count"],
        "FINAL_FEATURE_SCHEMA": str(OUT / "FINAL_FEATURE_SCHEMA.json"),
        "PROBE_FEATURE_USED": False,
        "POSTERIOR_MODEL": "3-seed GRU(16,64)+force MLP(1,64)+feasibility head with pilot monotonic regularizer",
        "POSTERIOR_CHECKPOINT": training["checkpoint"],
        "POSTERIOR_TRAIN_VALID": False,
        "POSTERIOR_TRAIN_VALID_REASON": "pilot-only; monotonicity still has 2/74 contexts with adjacent decreases and held-out root7703 is not formal",
        "ROOT7703_SELECTED_FORCE": target["selected_force_N"],
        "ROOT7703_PREDICTED_SUCCESS_PROB": target["predicted_success_probability"],
        "ROOT7703_REALIZED_FORCE": live["realized_force_N"],
        "ROOT7703_LIFT_SUCCESS": live["lift_success"],
        "ROOT7703_HOLD_SUCCESS": live["hold_30_step_success"],
        "FIXED_LOW_2N_OUTCOME": baseline["fixed_low_2N"]["outcome"],
        "FIXED_HIGH_6N_OUTCOME": baseline["fixed_high_6N"]["outcome"],
        "ACTIVEFORCING_COMMAND_FORCE_SAVING_VS_6N": baseline["saving_vs_6N"]["commanded_abs_N"],
        "ACTIVEFORCING_REALIZED_FORCE_SAVING_VS_6N": baseline["saving_vs_6N"]["realized_abs_N"],
        "FINAL_NO_PROBE_ACTIVEFORCING_SMOKE": live["final_no_probe_activeforcing_smoke"],
        "PILOT_CONTEXTS": pilot["heldout_pilot_contexts_available"],
        "ACTIVEFORCING_PILOT_SUCCESS_RATE": pilot["activeforcing_success_rate"],
        "FIXED_HIGH_PILOT_SUCCESS_RATE": pilot["fixed_high_6N_success_rate"],
        "ACTIVEFORCING_MEAN_FORCE": None,
        "FIXED_HIGH_MEAN_FORCE": None,
        "CURRENT_BLOCKER": "INSUFFICIENT_TRAIN_DATA plus posterior calibration/force-response instability; only two admitted contexts and no valid held-out pilot",
        "NEXT_ACTION": pilot["next_action"],
    })
    print(json.dumps({"status": "written", "pilot": pilot["status"]}, indent=2))


if __name__ == "__main__":
    main()
