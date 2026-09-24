#!/usr/bin/env python3
"""Freeze a task-agnostic adaptive boundary acquisition protocol.

No simulator is launched here.  Candidate rows are recommendations from the
frozen state machine and remain non-launchable until the force-interface and
resource gates pass.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "BOUNDARY_AWARE_FULL_TASK_FEASIBILITY_DATA_CLOSURE_20260902"
LABELS = OUT / "AUTHORITATIVE_720_BOUNDARY_LABELS.csv"
BOUNDARIES = OUT / "EXISTING_BOUNDARY_STATUS.csv"
CONTROLLER = Path("/home/exouser/Tabero/analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py")
CONTROLLER_AUDIT = Path("/home/exouser/Tabero/analysis/results/dev_fine_force_forensic_20260829_150000/CONTINUOUS_FORCE_CONTROLLER_AUDIT.json")
LOW_FORCE_EVIDENCE = Path("/home/exouser/Tabero/analysis/results/f1r2_tabero_reactive_force_20260818_205953/FINAL_VERDICT.json")
TASKS = (0, 1, 5, 6)
ROOT_INDICES = (0, 1)
RESOLUTION_N = 0.25
COARSE_STEP_N = 1.0
EMPIRICAL_LOWER_GUARD_N = 1.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bracket(q: pd.DataFrame, label: str):
    fail = q.loc[q[label] == 0, "force_N"]
    succ = q.loc[q[label] == 1, "force_N"]
    return (float(fail.max()) if len(fail) else np.nan, float(succ.min()) if len(succ) else np.nan)


def round_resolution(x: float) -> float:
    return round(x / RESOLUTION_N) * RESOLUTION_N


def propose(q: pd.DataFrame) -> dict:
    local_fail, local_success = bracket(q, "local_lift_success")
    full_fail, full_success = bracket(q, "full_task_success")
    min_force = float(q.force_N.min())
    if np.isnan(local_fail):
        candidate = max(EMPIRICAL_LOWER_GUARD_N, round_resolution(min_force - COARSE_STEP_N))
        return {
            "search_target": "LOCAL_LIFT_BOUNDARY", "acquisition_state": "DOWNWARD_SEARCH_LOCAL",
            "proposed_force_N": candidate, "lower_bracket_N": "", "upper_bracket_N": min_force,
            "reason": "lowest observed force still lifts; move downward by the preregistered coarse step",
        }
    if local_fail >= local_success:
        # Repeat noise/nonmonotonicity: replicate the nearest contradictory
        # existing cells before interpolating a fictitious boundary.
        candidate = round_resolution((local_fail + local_success) / 2.0)
        return {
            "search_target": "LOCAL_LIFT_BOUNDARY", "acquisition_state": "REPLICATE_LOCAL_OVERLAP",
            "proposed_force_N": candidate, "lower_bracket_N": local_fail, "upper_bracket_N": local_success,
            "reason": "local outcomes overlap in force; acquire repeats before midpoint refinement",
        }
    if local_success - local_fail > RESOLUTION_N + 1e-12:
        candidate = round_resolution((local_fail + local_success) / 2.0)
        return {
            "search_target": "LOCAL_LIFT_BOUNDARY", "acquisition_state": "BISECT_LOCAL",
            "proposed_force_N": candidate, "lower_bracket_N": local_fail, "upper_bracket_N": local_success,
            "reason": "clean local failure/success bracket; bisect at calibrated resolution",
        }
    if np.isnan(full_fail):
        candidate = local_success
        return {
            "search_target": "FULL_TASK_BOUNDARY", "acquisition_state": "SEEK_DELAYED_FAILURE",
            "proposed_force_N": candidate, "lower_bracket_N": "", "upper_bracket_N": full_success,
            "reason": "local boundary resolved but no downstream failure; replicate just above local transition",
        }
    if np.isnan(full_success):
        candidate = round_resolution(full_fail + COARSE_STEP_N)
        return {
            "search_target": "FULL_TASK_BOUNDARY", "acquisition_state": "UPWARD_SEARCH_FULL",
            "proposed_force_N": candidate, "lower_bracket_N": full_fail, "upper_bracket_N": "",
            "reason": "full task fails throughout observed support; move upward within unchanged task support/safety ceiling",
        }
    if full_fail >= full_success:
        candidate = round_resolution((full_fail + full_success) / 2.0)
        return {
            "search_target": "FULL_TASK_BOUNDARY", "acquisition_state": "REPLICATE_FULL_OVERLAP",
            "proposed_force_N": candidate, "lower_bracket_N": full_fail, "upper_bracket_N": full_success,
            "reason": "full-task outcomes overlap in force; replicate before interpolation",
        }
    candidate = round_resolution((full_fail + full_success) / 2.0)
    return {
        "search_target": "FULL_TASK_BOUNDARY", "acquisition_state": "BISECT_FULL",
        "proposed_force_N": candidate, "lower_bracket_N": full_fail, "upper_bracket_N": full_success,
        "reason": "clean full-task failure/success bracket; bisect at calibrated resolution",
    }


def main() -> None:
    d = pd.read_csv(LABELS)
    selected = d[d.root_index.isin(ROOT_INDICES)].copy()
    assert selected.task.isin(TASKS).all()
    assert selected.context_id.nunique() == 24
    targets = []
    next_rows = []
    for (task, root, ridx, band, mu, cid), q in selected.groupby(["task", "root_id", "root_index", "friction_band", "mu_GT", "context_id"], sort=True):
        targets.append({
            "task": int(task), "task_priority": TASKS.index(int(task)) + 1, "root_id": root, "root_index": int(ridx),
            "friction_band": band, "friction": mu, "context_id": cid, "split": "TRAIN_DEV_OBSERVED_ROOT",
            "selection_rule": "ROOT_INDEX_0_OR_1_FOR_EVERY_TASK;ALL_THREE_FRICTION_BANDS",
        })
        p = propose(q)
        p.update({
            "task": int(task), "task_priority": TASKS.index(int(task)) + 1, "root_id": root, "root_index": int(ridx),
            "friction_band": band, "friction": mu, "context_id": cid, "repeat_count_required": 2,
            "candidate_is_launchable": 0,
            "blocking_gate": "MINIMAL_FORCE_INTERFACE_CALIBRATION_PASS_AND_RESOURCE_GATE",
            "utility_used": 0, "task_specific_outcome_tuning_used": 0,
        })
        next_rows.append(p)
    pd.DataFrame(targets).to_csv(OUT / "QUALIFICATION_TARGET_CONTEXTS.csv", index=False)
    pd.DataFrame(next_rows).to_csv(OUT / "BOUNDARY_NEXT_QUERY_STATE.csv", index=False)

    protocol = {
        "name": "BOUNDARY_AWARE_FULL_TASK_FEASIBILITY_DATA_CLOSURE",
        "created_utc": now(), "status": "FROZEN_PRE_OUTCOME_CPU_PREPARATION",
        "scope": {"tasks": list(TASKS), "root_indices_each_task": list(ROOT_INDICES), "friction_bands": ["LOW", "MID", "HIGH"], "contexts": 24},
        "priority_order_is_scheduling_only": list(TASKS),
        "task_agnostic_rule": True,
        "outcome_classes": ["LOCAL_FAILURE", "LOCAL_SUCCESS_FULL_FAILURE", "FULL_SUCCESS"],
        "boundaries": ["LOCAL_LIFT", "FULL_TASK"],
        "initial_rule": "If lowest tested force lifts, query one calibrated coarse step lower. If failure/success are bracketed, bisect. If outcomes overlap, replicate contradictory cells before interpolation.",
        "coarse_step_N": COARSE_STEP_N,
        "refinement_resolution_N": RESOLUTION_N,
        "resolution_status": "COMMAND_SUPPORTED_BUT_ADJACENT_MEASURED_SEPARATION_REQUIRES_MINIMAL_PREFLIGHT",
        "empirical_lower_guard_N": EMPIRICAL_LOWER_GUARD_N,
        "lower_guard_status": "PRIOR_EMPIRICAL_STABLE_SETPOINT_ON_DIFFERENT_TASK;NOT_CONTROLLER_SAFETY_CERTIFICATION",
        "minimum_repeats_near_boundary": 2,
        "stopping_rules": [
            "local or full bracket width <= 0.25 N after repeat consistency check",
            "controller lower guard reached with all local successes => NO_LOCAL_BOUNDARY_IN_SAFE_RANGE",
            "task support upper bound reached with all full failures => RIGHT_CENSORED_FULL_BOUNDARY",
            "candidate not measurably distinct in preflight",
            "resource gate closed or telemetry/state-parity failure",
        ],
        "unchanged_components": ["pi0", "P4-B query", "friction estimator", "Direct architecture", "Utility", "candidate semantics", "labels", "controller semantics"],
        "new_candidate_effect_on_runtime_utility": "NONE; acquisition branches train/evaluate Direct only and do not expand the frozen runtime candidate set",
        "launch_gates": {
            "force_interface": "MINIMAL_FORCE_INTERFACE_CALIBRATION_PASS.json must exist and pass 1.0 N lower setpoint plus 0.25 N adjacent distinguishability around planned region",
            "resources": "No protected simulator conflict; GPU utilization <= 90%, with stricter capacity/headroom check before launch",
            "telemetry": "all required telemetry and state hashes configured",
        },
        "lineage": {"old": "UNIFORM_OLD", "new": "BOUNDARY_SEEKING"},
        "source_hashes": {str(p): sha(p) for p in [LABELS, BOUNDARIES, CONTROLLER, CONTROLLER_AUDIT, LOW_FORCE_EVIDENCE]},
        "prohibited": ["task6-specific force rule", "Utility retuning", "global Fmax promotion", "pi0 retraining", "identifier retraining", "P4-B change", "friction estimator change", "TEST claim"],
    }
    (OUT / "BOUNDARY_ACQUISITION_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")

    safety_rows = [
        {"evidence": "controller source", "claim": "arbitrary float force command accepted; force not clipped in _make_action", "value": "a[0,9]=a[0,12]=0.5*f_star", "scope": "implementation", "qualification": "PASS"},
        {"evidence": "controller source", "claim": "servo displacement deadband", "value": "0.4 N", "scope": "implementation", "qualification": "PASS"},
        {"evidence": "continuous-force audit", "claim": "0.25 N commands supported", "value": "4.25/4.50/4.75 N", "scope": "command interface", "qualification": "PASS_COMMAND_ONLY"},
        {"evidence": "legacy low-force validation", "claim": "lowest stable requested setpoint observed", "value": "1.0 N measured 1.196 N", "scope": "different task/protocol", "qualification": "EMPIRICAL_NOT_CERTIFIED"},
        {"evidence": "project search", "claim": "controller-certified global safe minimum", "value": "NOT_FOUND", "scope": "all inspected sources", "qualification": "MISSING"},
        {"evidence": "P7B preflight", "claim": "6/7/8 N formal tracking rows", "value": "PENDING_PHYSICAL_RUNTIME", "scope": "not usable", "qualification": "NO_EVIDENCE"},
    ]
    pd.DataFrame(safety_rows).to_csv(OUT / "FORCE_INTERFACE_SAFETY_AUDIT.csv", index=False)
    safety_md = f"""# Force-interface safety and resolution audit

Status: `MINIMAL_PREFLIGHT_REQUIRED_BEFORE_BOUNDARY_COLLECTION`

The controller accepts arbitrary floating-point targets and does not clip the force command; each finger receives half the requested total force. The force servo uses a 0.4 N deadband and clips gripper displacement, not force. Prior project evidence executed 0.25 N commands at 4.25/4.50/4.75 N and a separate legacy task tracked a 1.0 N request at 1.196 N.

No inspected source certifies a project-wide safe minimum. Therefore 1.0 N is only an empirical lower guard, not a safety certificate, and the unlaunched 0.5 N assumption in `current4task_low_force_e3.py` is rejected.

Before any object rollout, a minimal interface preflight must verify:

1. Stable, nonsaturated tracking at 1.0 N in the frozen current controller.
2. Measured separation for adjacent 0.25 N commands around each planned bracket region; the separation criterion is preregistered as non-overlapping 95% intervals or a mean difference >= 0.20 N with tracking MAE <= 0.40 N.
3. No force spike above the existing project-wide 8 N action-support ceiling during preflight.

Until that JSON gate exists and passes, every row in `BOUNDARY_NEXT_QUERY_STATE.csv` has `candidate_is_launchable=0`.

Sources:

- `{CONTROLLER}` (SHA-256 `{sha(CONTROLLER)}`)
- `{CONTROLLER_AUDIT}` (SHA-256 `{sha(CONTROLLER_AUDIT)}`)
- `{LOW_FORCE_EVIDENCE}` (SHA-256 `{sha(LOW_FORCE_EVIDENCE)}`)
"""
    (OUT / "FORCE_INTERFACE_SAFETY_AUDIT.md").write_text(safety_md)


if __name__ == "__main__":
    main()
