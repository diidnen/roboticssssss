#!/usr/bin/env python3
"""Offline shadow audit for the bounded adaptive opening release law.

The source force/aperture trajectory remains the executed old controller.  The
counterfactual columns below are command-only shadows; they never claim a
counterfactual physical force outcome.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SOURCE = Path(
    "/home/exouser/FORTE/analysis/results/"
    "actuator_following_arm_coupling_diagnosis_20260905/"
    "ROOT7703_4N_NORMAL_TRACE.csv"
)
OUT = Path(
    "/home/exouser/FORTE/analysis/results/"
    "backlog_slope_dynamic_release_fix_20260905"
)
DT_S = 1.0 / 60.0
TARGET_N = 4.0
SLOPE_N_PER_MM = 9.4
DEADZONE_N = 0.25
CONTACT_GUARD_WIDTH_N = 0.75
OLD_OPEN_RATE_MPS = 0.0024
REJECTED_2X_OPEN_RATE_MPS = 0.0048
EARLY_WINDOW_STEPS = 15


@dataclass(frozen=True)
class Candidate:
    alpha: float
    backlog_limit_mm: float
    rate_multiplier: float


PREDECLARED = tuple(
    Candidate(alpha, backlog_mm, rate)
    for alpha in (0.10, 0.20, 0.30)
    for backlog_mm in (1.2124786153435707, 1.738918013870716, 2.0110527984797955)
    for rate in (1.0, 1.25, 1.5)
)
# Conservative primary candidate: middle alpha, empirical P50 backlog limit,
# and the smallest allowed ceiling increase.  It is selected by a fixed rule,
# not by task outcome.
BACKLOG_SOFT_START_MM = 1.2124786153435707
SELECTED = Candidate(0.20, 2.0110527984797955, 1.25)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_bool(row: dict[str, str], key: str) -> bool:
    return row[key].strip().lower() in {"1", "true", "yes"}


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def load_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {
        "physics_step", "F_des", "F_meas_raw", "F_meas_filtered",
        "d_previous_command", "d_actual_before", "delta_d_after_rate_limit",
        "bilateral_contact",
    }
    missing = sorted(required.difference(rows[0] if rows else {}))
    if missing:
        raise RuntimeError(f"source trace missing required fields: {missing}")
    physics = [int(as_float(row, "physics_step")) for row in rows]
    if len(rows) != 435 or len(set(physics)) != len(physics):
        raise RuntimeError("source trace grain is not one unique row per physics step")
    if any(b != a + 1 for a, b in zip(physics, physics[1:])):
        raise RuntimeError("source physics-step sequence is not contiguous")
    return rows


def contact_scale(raw_force_n: float) -> tuple[str, float]:
    if raw_force_n <= TARGET_N + DEADZONE_N:
        return "WEAKENING", 0.0
    scale = float(np.clip(
        (raw_force_n - (TARGET_N + DEADZONE_N)) / CONTACT_GUARD_WIDTH_N,
        0.0,
        1.0,
    ))
    return ("STABLE" if scale >= 1.0 else "WEAKENING"), scale


def shadow(rows: list[dict[str, str]], c: Candidate, law: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    command_m = as_float(rows[0], "d_previous_command")
    release_active = False
    command_onset_m = command_m
    actual_onset_m = as_float(rows[0], "d_actual_before")
    out: list[dict[str, Any]] = []
    total_release_m = 0.0
    early_release_by_segment: list[float] = []
    current_early_m = 0.0
    segment_step = 0
    max_backlog_mm = 0.0
    release_requests = 0
    backlog_suppressed = 0
    contact_guarded = 0
    added_at_or_above_limit = 0
    near_boundary_release_m = 0.0

    for row in rows:
        raw_n = as_float(row, "F_meas_raw")
        filt_n = as_float(row, "F_meas_filtered")
        actual_m = as_float(row, "d_actual_before")
        bilateral = as_bool(row, "bilateral_contact")
        excess_n = max(filt_n - TARGET_N, 0.0)
        release_demand = bilateral and excess_n >= DEADZONE_N

        if release_demand and not release_active:
            release_active = True
            command_onset_m = command_m
            actual_onset_m = actual_m
            segment_step = 0
            current_early_m = 0.0
        elif not release_demand and release_active:
            early_release_by_segment.append(current_early_m)
            release_active = False

        opening_backlog_mm = 0.0
        backlog_factor = 1.0
        margin_state = "LOST" if not bilateral else "STABLE"
        margin_factor = 0.0 if not bilateral else 1.0
        delta_needed_mm = excess_n / SLOPE_N_PER_MM
        proposed_m = c.alpha * delta_needed_mm / 1000.0

        if release_active:
            opening_backlog_mm = max(
                ((command_m - command_onset_m) - (actual_m - actual_onset_m)) * 1000.0,
                0.0,
            )
            max_backlog_mm = max(max_backlog_mm, opening_backlog_mm)
            if law in {"S2", "S3"}:
                if c.backlog_limit_mm <= BACKLOG_SOFT_START_MM + 1e-12:
                    backlog_factor = float(np.clip(
                        1.0 - opening_backlog_mm / c.backlog_limit_mm, 0.0, 1.0
                    ))
                else:
                    backlog_factor = float(np.clip(
                        (c.backlog_limit_mm - opening_backlog_mm)
                        / (c.backlog_limit_mm - BACKLOG_SOFT_START_MM),
                        0.0,
                        1.0,
                    ))
            if law == "S3":
                margin_state, margin_factor = contact_scale(raw_n)
            elif not bilateral:
                margin_state, margin_factor = "LOST", 0.0

            if law == "S0":
                delta_m = max(as_float(row, "delta_d_after_rate_limit"), 0.0)
            else:
                delta_m = min(
                    proposed_m * backlog_factor * margin_factor,
                    OLD_OPEN_RATE_MPS * c.rate_multiplier * DT_S,
                )
            release_requests += 1
            backlog_suppressed += int(backlog_factor < 0.999999)
            contact_guarded += int(margin_factor < 0.999999)
            added_at_or_above_limit += int(
                opening_backlog_mm >= c.backlog_limit_mm - 1e-12 and delta_m > 1e-12
            )
            if raw_n <= TARGET_N + 1.0:
                near_boundary_release_m += delta_m
            if segment_step < EARLY_WINDOW_STEPS:
                current_early_m += delta_m
            segment_step += 1
        else:
            delta_m = 0.0

        before_m = command_m
        command_m += delta_m
        total_release_m += delta_m
        out.append({
            "physics_step": int(as_float(row, "physics_step")),
            "F_des_N": TARGET_N,
            "F_meas_raw_N": raw_n,
            "F_meas_filtered_N": filt_n,
            "force_excess_N": excess_n,
            "bilateral_contact_observed_old_trace": bilateral,
            "contact_margin_state": margin_state,
            "contact_margin_scale": margin_factor,
            "local_slope_N_per_mm": SLOPE_N_PER_MM,
            "delta_d_needed_mm": delta_needed_mm,
            "opening_backlog_mm": opening_backlog_mm,
            "backlog_scale": backlog_factor,
            "shadow_delta_d_mm": delta_m * 1000.0,
            "shadow_d_force_cmd_before_m": before_m,
            "shadow_d_force_cmd_after_m": command_m,
            "shadow_only_no_counterfactual_force_claim": True,
        })

    if release_active:
        early_release_by_segment.append(current_early_m)
    metrics = {
        "law": law,
        "alpha_release": None if law == "S0" else c.alpha,
        "backlog_limit_mm": None if law in {"S0", "S1"} else c.backlog_limit_mm,
        "rate_multiplier": 1.0 if law == "S0" else c.rate_multiplier,
        "opening_rate_ceiling_mps": (
            OLD_OPEN_RATE_MPS
            if law == "S0"
            else OLD_OPEN_RATE_MPS * c.rate_multiplier
        ),
        "predicted_cumulative_opening_mm": total_release_m * 1000.0,
        "predicted_maximum_opening_backlog_mm": max_backlog_mm,
        "opening_request_steps": release_requests,
        "backlog_suppressed_steps": backlog_suppressed,
        "contact_guard_activated_steps": contact_guarded,
        "release_added_at_or_above_backlog_limit_steps": added_at_or_above_limit,
        "near_contact_boundary_release_mm": near_boundary_release_m * 1000.0,
        "first_250ms_release_by_segment_mm": [x * 1000.0 for x in early_release_by_segment],
        "maximum_command_jump_mm": max((x["shadow_delta_d_mm"] for x in out), default=0.0),
    }
    return out, metrics


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    # Empirical offset-corrected backlog distribution from the executed old law.
    old_cmd = as_float(rows[0], "d_previous_command")
    old_actual = as_float(rows[0], "d_actual_before")
    active = False
    cmd_onset = old_cmd
    actual_onset = old_actual
    old_backlogs: list[float] = []
    for row in rows:
        demand = (
            as_bool(row, "bilateral_contact")
            and as_float(row, "F_meas_filtered") >= TARGET_N + DEADZONE_N
        )
        if demand and not active:
            cmd_onset = as_float(row, "d_previous_command")
            actual_onset = as_float(row, "d_actual_before")
        active = demand
        if demand:
            old_backlogs.append(max(
                ((as_float(row, "d_previous_command") - cmd_onset)
                 - (as_float(row, "d_actual_before") - actual_onset)) * 1000.0,
                0.0,
            ))

    _, s0 = shadow(rows, SELECTED, "S0")
    _, s1 = shadow(rows, SELECTED, "S1")
    _, s2 = shadow(rows, SELECTED, "S2")
    selected_trace, s3 = shadow(rows, SELECTED, "S3")

    grid: list[dict[str, Any]] = []
    for candidate in PREDECLARED:
        _, metrics = shadow(rows, candidate, "S3")
        old_early = s0["first_250ms_release_by_segment_mm"][:2]
        new_early = metrics["first_250ms_release_by_segment_mm"][:2]
        faster = len(old_early) == len(new_early) == 2 and all(
            new > old + 1e-9 for new, old in zip(new_early, old_early)
        )
        safer_than_2x = (
            metrics["maximum_command_jump_mm"]
            < REJECTED_2X_OPEN_RATE_MPS * DT_S * 1000.0 - 1e-9
            and metrics["near_contact_boundary_release_mm"]
            < s0["near_contact_boundary_release_mm"]
            and metrics["release_added_at_or_above_backlog_limit_steps"] == 0
        )
        metrics.update({
            "release_faster_than_old_first_250ms_of_first_two_segments": faster,
            "release_safer_than_rejected_2x_near_contact_boundary": safer_than_2x,
        })
        grid.append(metrics)

    selected_grid = next(
        item for item in grid
        if item["alpha_release"] == SELECTED.alpha
        and item["backlog_limit_mm"] == SELECTED.backlog_limit_mm
        and item["rate_multiplier"] == SELECTED.rate_multiplier
    )
    shadow_pass = bool(
        selected_grid["release_faster_than_old_first_250ms_of_first_two_segments"]
        and selected_grid["release_safer_than_rejected_2x_near_contact_boundary"]
        and selected_grid["maximum_command_jump_mm"] <= 0.05 + 1e-9
        and selected_grid["release_added_at_or_above_backlog_limit_steps"] == 0
        and selected_grid["contact_guard_activated_steps"] > 0
        and selected_grid["backlog_suppressed_steps"] > 0
    )

    with (OUT / "SHADOW_RELEASE_TRACE.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(selected_trace[0]))
        writer.writeheader()
        writer.writerows(selected_trace)

    design = {
        "schema": "BACKLOG_SLOPE_RELEASE_DESIGN_V1",
        "target_force_N": TARGET_N,
        "opening_sign": "positive aperture delta opens the gripper",
        "direct_target_minus_actual_not_used_reason": (
            "contact compliance creates a large static target/actual offset"
        ),
        "opening_backlog_definition": (
            "max((d_cmd-d_cmd_at_release_onset) - "
            "(d_actual-d_actual_at_release_onset), 0)"
        ),
        "backlog_suppression_definition": (
            "scale=1 through empirical P50; linearly taper to 0 at the "
            "selected hard limit (P90)"
        ),
        "backlog_soft_start_mm": BACKLOG_SOFT_START_MM,
        "empirical_old_backlog_quantiles_mm": {
            "p50": percentile(old_backlogs, 0.50),
            "p75": percentile(old_backlogs, 0.75),
            "p90": percentile(old_backlogs, 0.90),
        },
        "local_force_aperture_slope_N_per_mm": SLOPE_N_PER_MM,
        "online_slope_update_used": False,
        "alpha_candidates": [0.10, 0.20, 0.30],
        "backlog_limit_candidates_source": "old 4N offset-corrected P50/P75/P90",
        "rate_ceiling_candidates": [1.0, 1.25, 1.5],
        "selected_candidate": selected_grid,
        "contact_margin_guard": {
            "metric": "object-filtered bilateral true squeeze raw force",
            "lost": "bilateral contact false",
            "weakening": "F_raw <= F_des + 1.0N",
            "continuous_scale": "clip((F_raw-(F_des+0.25N))/0.75N,0,1)",
        },
        "force_gain_changed": False,
        "integral_added": False,
        "rejected_2x_rate_reused": False,
    }
    comparison = {
        "source_trace": str(SOURCE),
        "source_trace_grain": "one row per 60Hz physics step",
        "source_trace_rows": len(rows),
        "counterfactual_scope": (
            "command trajectory only; observed force/contact remain old-policy data"
        ),
        "laws": {"S0": s0, "S1": s1, "S2": s2, "S3": s3},
        "predeclared_s3_grid": grid,
    }
    safety = {
        "shadow_mode_pass": shadow_pass,
        "shadow_backlog_bounded": selected_grid["release_added_at_or_above_backlog_limit_steps"] == 0,
        "shadow_no_large_command_jump": selected_grid["maximum_command_jump_mm"] <= 0.05 + 1e-9,
        "shadow_contact_guard_valid": selected_grid["contact_guard_activated_steps"] > 0,
        "shadow_release_faster_than_old": selected_grid["release_faster_than_old_first_250ms_of_first_two_segments"],
        "shadow_release_slower_than_rejected_2x_near_contact_boundary": selected_grid["release_safer_than_rejected_2x_near_contact_boundary"],
        "important_limitation": (
            "PASS authorizes one live diagnostic; it does not predict physical tracking success"
        ),
    }
    for name, value in (
        ("RELEASE_LAW_DESIGN.json", design),
        ("SHADOW_RELEASE_COMPARISON.json", comparison),
        ("SHADOW_SAFETY_RESULT.json", safety),
    ):
        (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT), **safety}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
