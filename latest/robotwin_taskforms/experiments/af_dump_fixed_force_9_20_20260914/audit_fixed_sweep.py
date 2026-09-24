"""Independent raw audit and contact-realization metrics for the fixed sweep."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "af_dump_original_restore_20260912"
sys.path.insert(0, str(OLD))
from audit_original_collected_group import audit_branch, audit_query


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_metrics(branch: Path, command: float) -> dict:
    active = []
    contact = []
    filtered = []
    apertures = []
    raw_all = []
    longest_loss = 0
    current_loss = 0
    contact_started = False
    with gzip.open(branch / "physics_trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            raw = float(row["contact"]["measured_squeeze_n"])
            inner = row["original_squeeze_inner"]
            reference = float(inner["force_reference_N"])
            filt = float(inner["measured_filtered_N"])
            aperture = float(inner["inner_aperture_m"])
            values = np.asarray([raw, reference, filt, aperture], float)
            if not np.isfinite(values).all() or not 0 <= aperture <= 0.04:
                raise ValueError("Nonfinite or out-of-range controller telemetry")
            raw_all.append(raw)
            if reference > 0:
                active.append(raw)
                apertures.append(aperture)
                if raw >= 1.0:
                    contact_started = True
                    contact.append(raw)
                    filtered.append(filt)
                    current_loss = 0
                elif contact_started:
                    current_loss += 1
                    longest_loss = max(longest_loss, current_loss)
    if not raw_all or not active:
        raise ValueError("Empty active-force telemetry")
    return {
        "physics_steps": len(raw_all),
        "active_reference_steps": len(active),
        "target_contact_steps": len(contact),
        "target_contact_fraction": len(contact) / len(active),
        "active_mean_squeeze_N": float(np.mean(active)),
        "target_contact_mean_squeeze_N": float(np.mean(contact)) if contact else None,
        "target_contact_median_squeeze_N": float(np.median(contact)) if contact else None,
        "target_contact_mean_filtered_N": float(np.mean(filtered)) if filtered else None,
        "target_contact_tracking_ratio": float(np.mean(contact) / command) if contact else None,
        "active_zero_aperture_fraction": float(np.mean(np.asarray(apertures) <= 1e-7)),
        "active_min_aperture_m": float(np.min(apertures)),
        "max_squeeze_N": float(np.max(raw_all)),
        "longest_postcontact_loss_steps": longest_loss,
        "contact_threshold_N": 1.0,
    }


def audit_context(case: Path) -> dict:
    job = case / "job"
    result = read(job / "FIXED_SWEEP_CONTEXT_RESULT.json")
    context = result["context"]
    query = audit_query(job / "query")
    if not query["passed"]:
        raise ValueError("Query audit failed")
    lock = read(job / "PREACTION_SELECTION_LOCK.json")
    if result["selection_lock_sha256"] != sha(job / "PREACTION_SELECTION_LOCK.json"):
        raise ValueError("Selection lock changed")
    online = read(job / "online_qualification.json")
    forces = [float(value) for value in context["forces_N"]]
    if online["first_chunk_hashes"] != [lock["first_chunk_sha256"]] * len(forces):
        raise ValueError("Unpaired first policy chunks")
    if online["common_handoff_state_sha256"] != lock["state_sha256"]:
        raise ValueError("Common handoff hash mismatch")
    branches = []
    for index, force in enumerate(forces):
        matches = list(job.glob(f"branch_{index}_{force:g}N"))
        if len(matches) != 1:
            raise ValueError("Missing fixed-force branch")
        branch = matches[0]
        raw_audit = audit_branch(branch)
        raw_result = read(branch / "result.json")
        if float(raw_result["force_setpoint_bilateral_n"]) != force:
            raise ValueError("Audited branch force mismatch")
        binding = raw_result["arbitration_binding"]
        if binding["force_support_bilateral_N"] != [0.5, 15.0]:
            raise ValueError("15 N arbitration support binding absent")
        if binding["utility_changed"] or binding["release_and_feedback_rules_changed"]:
            raise ValueError("Controller semantics changed")
        branches.append(
            {
                "method": f"Fixed-{force:g}N",
                "force_N": force,
                "success": int(raw_result["success"]),
                "raw_audit": raw_audit,
                "trace_metrics": trace_metrics(branch, force),
                "result": raw_result,
            }
        )
    return {
        "passed": True,
        "context": context,
        "query_audit": query,
        "branches": branches,
        "first_chunk_pairing_exact": True,
        "common_handoff_state_sha256": lock["state_sha256"],
        "context_result_sha256": sha(job / "FIXED_SWEEP_CONTEXT_RESULT.json"),
        "audit_source_sha256": sha(Path(__file__)),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_context(args.case.resolve()), indent=2))
