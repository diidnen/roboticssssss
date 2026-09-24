#!/usr/bin/env python3
"""Build truthful phase-1 manifests and cross-campaign handoffs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "BOUNDARY_AWARE_FULL_TASK_FEASIBILITY_DATA_CLOSURE_20260902"
LABELS = OUT / "AUTHORITATIVE_720_BOUNDARY_LABELS.csv"
PROTOCOL = OUT / "BOUNDARY_ACQUISITION_PROTOCOL.json"
TRAIN_PROTOCOL = OUT / "BOUNDARY_DIRECT_MATCHED_TRAINING_PROTOCOL.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    d = pd.read_csv(LABELS)
    task_balance = []
    ready = []
    for task, q in d.groupby("task", sort=True):
        rec = {
            "task": int(task), "branches": len(q),
            "local_failure_n": int((q.local_lift_success == 0).sum()),
            "local_success_full_failure_n": int(((q.local_lift_success == 1) & (q.full_task_success == 0)).sum()),
            "full_success_n": int((q.full_task_success == 1).sum()),
            "telemetry_present_n": int(q.telemetry_exists.sum()),
            "full_label_reconstructed_n": int((q.label_source != "DIRECT_CUMULATIVE_BRANCH_LABEL").sum()),
        }
        rec["three_regime_mechanism_ready"] = bool(rec["local_failure_n"] and rec["local_success_full_failure_n"] and rec["full_success_n"])
        task_balance.append(rec)
        if rec["three_regime_mechanism_ready"]:
            ready.append(int(task))
    write_json("CURRENT_TASK_E3_MECHANISM_READY.json", {
        "status": "CURRENT_TASK_E3_MECHANISM_READY",
        "evidence_scope": "OBSERVED_TRAIN_DEV_MECHANISM_EVIDENCE_ONLY",
        "ready_tasks": ready,
        "task_balance": task_balance,
        "criterion": "at least one local failure, one local-success/full-failure branch, and one full-task success",
        "task1_caveat": "140/180 full-task labels are frozen reconstruction",
        "task6_note": "all 180 full-task labels direct; 2 local failures, 3 delayed failures, 175 full successes",
        "long_horizon_E3_cancelled": False,
        "source": str(LABELS), "source_sha256": sha(LABELS),
    })

    write_json("BOUNDARY_AUGMENTED_DATASET_MANIFEST.json", {
        "status": "PENDING_BOUNDARY_COLLECTION",
        "old_data": {"acquisition_type": "UNIFORM_OLD", "branches": 720, "manifest": str(LABELS), "sha256": sha(LABELS)},
        "new_data": {"acquisition_type": "BOUNDARY_SEEKING", "branches": 0, "manifest": None},
        "old_branches": 720, "new_boundary_branches": 0, "tasks": [0, 1, 5, 6],
        "telemetry_complete": False, "state_hash_parity_complete": False,
        "old_archive_overwritten": False,
        "blocked_by": ["MINIMAL_FORCE_INTERFACE_CALIBRATION_PASS", "PROTECTED_SIMULATOR_CAPACITY"],
        "training_allowed": False,
        "test_claim_allowed": False,
    })

    common = """## Unified boundary-seeking rule

For every task/root/physics context, classify branches as local failure, local-success/full-failure, or full success. If the current minimum still lifts, search downward by the calibrated coarse step. After a failure/success bracket exists, bisect to the empirically distinguishable force resolution. Replicate stochastic overlaps before interpolation. Stop at the safe lower guard and report censoring rather than inventing a frontier.

Required branch telemetry: task/root/physics, command/mean/peak force, repeat, initial/query state hashes, grasp/lift height/lift hold, retention/reorientation/placement/full success, drop time/position, failure stage/reason, and tracking error.

Do not change π0, P4-B, friction estimator, Direct architecture, Utility, or controller semantics. Do not spend rollouts in an obviously high-force all-success region.
"""
    e3 = f"""# Boundary collection handoff to E3

Status: `MECHANISM_READY_ON_OBSERVED_CURRENT_TASKS; LONG_HORIZON_STILL_REQUIRED`

Tasks {ready} already contain all three physical regimes in observed TRAIN/DEV data. This is mechanism evidence only; it does not replace genuinely long-horizon task-level validation. Task1 carries a reconstructed-label caveat, while task6 is the clean direct-label anchor.

{common}

E3-specific: locate and report both the 3 cm local-lift transition and the later retention/reorientation/placement transition. Reuse already scheduled task5 cells when identical in root, friction, force, repeat, and state hash; never duplicate them merely to satisfy a table shape.
"""
    (OUT / "BOUNDARY_COLLECTION_HANDOFF_TO_E3.md").write_text(e3)
    mass = f"""# Boundary collection handoff to Mass

Status: `PROTOCOL_READY; NO_RUNNING_CAMPAIGN_MUTATION`

{common}

Mass-specific: treat mass as the physics-context coordinate and seek the lift and full-task transitions separately inside each task/root/mass cell. Preserve the current Mass result directory and config. Apply this rule only to a new, coordinator-approved TRAIN/DEV collection; do not alter the active formal campaign.
"""
    (OUT / "BOUNDARY_COLLECTION_HANDOFF_TO_MASS.md").write_text(mass)
    joint = f"""# Boundary collection handoff to Joint friction×mass

Status: `PROTOCOL_READY; NO_RUNNING_CAMPAIGN_MUTATION`

{common}

Joint-specific: the context key is task × root × friction × mass. Allocate matched budget to cells whose present outcomes do not bracket either transition; avoid multiplying high-force successes across the full Cartesian grid. Keep physics estimation and acquisition selection distinct in lineage.
"""
    (OUT / "BOUNDARY_COLLECTION_HANDOFF_TO_JOINT.md").write_text(joint)

    fresh = """# Boundary-trained Direct fresh TEST plan

Status: `PLAN_ONLY_NOT_AUTHORIZED`

This plan activates only if the matched TRAIN/DEV comparison passes the boundary-data promotion gate. The observed 720 branches, the diagnosed task6 roots, and all new qualification roots are development evidence and cannot be relabeled as sealed TEST.

Use genuinely fresh physical root families that have not been viewed for selection, calibration, stopping, or reporting. Freeze before opening outcomes:

- unchanged π0 checkpoint, identifier, P4-B query, friction estimator, Direct architecture, candidate support, Utility, controller, labels, and episode semantics;
- OLD Direct versus BOUNDARY-trained Direct, plus Success-Only and Fixed-Max;
- task6 as primary and task0/task1/task5 as mandatory cross-task guardrails;
- root-level paired allocation, at least two repeats per selected candidate, complete telemetry and state-hash parity;
- primary metrics: full-task SR, mean force, under-force, excess force, realized Utility, rescue, collateral, calibration, ranking, and boundary MAE.

The coordinator must attest that roots are genuinely fresh. Any E5 roots already viewed are ineligible even if previously called sealed. No collection is launched from this plan automatically.
"""
    (OUT / "BOUNDARY_DIRECT_FRESH_TEST_PLAN.md").write_text(fresh)

    write_json("PHASE1_MANIFEST_QA.json", {
        "status": "PASS_CPU_PHASE1",
        "files": {name: sha(OUT / name) for name in [
            "CURRENT_TASK_E3_MECHANISM_READY.json", "BOUNDARY_AUGMENTED_DATASET_MANIFEST.json",
            "BOUNDARY_COLLECTION_HANDOFF_TO_E3.md", "BOUNDARY_COLLECTION_HANDOFF_TO_MASS.md",
            "BOUNDARY_COLLECTION_HANDOFF_TO_JOINT.md", "BOUNDARY_DIRECT_FRESH_TEST_PLAN.md",
        ]},
        "boundary_protocol_sha256": sha(PROTOCOL), "training_protocol_sha256": sha(TRAIN_PROTOCOL),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
