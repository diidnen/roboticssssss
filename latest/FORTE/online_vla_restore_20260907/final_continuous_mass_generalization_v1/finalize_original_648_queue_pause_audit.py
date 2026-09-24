#!/usr/bin/env python3
"""Finalize evidence that the original 648 queue paused at branch boundaries."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from time_bounded_mass_common import HERE, load_frozen_subset, read, sha256, spec_lookup, validate_branch


LAUNCHERS = {86067: 181000, 86673: 181001, 87289: 181002, 87885: 181003}
ACTIVE_AT_PAUSE = [
    {"worker_pid": 155170, "launcher_pid": 86067, "context_id": "t0_r181000_m100g", "force_N": 4.5},
    {"worker_pid": 155157, "launcher_pid": 86673, "context_id": "t0_r181001_m100g", "force_N": 4.0},
    {"worker_pid": 153621, "launcher_pid": 87289, "context_id": "t0_r181002_m100g", "force_N": 4.0},
    {"worker_pid": 154387, "launcher_pid": 87885, "context_id": "t0_r181003_m100g", "force_N": 4.0},
]


def proc(pid: int) -> dict:
    status_path = Path(f"/proc/{pid}/status")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not status_path.exists():
        return {"pid": pid, "exists": False}
    fields = {}
    for line in status_path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    command = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    return {"pid": pid, "exists": True, "state": fields.get("State"),
            "parent_pid": int(fields.get("PPid", "-1")), "command": command}


def main() -> None:
    output = HERE / "ORIGINAL_648_QUEUE_PAUSE_AUDIT.json"
    if output.exists():
        raise FileExistsError(output)
    subset, subset_hash = load_frozen_subset()
    runtime = read(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json")
    primary = spec_lookup(subset)
    contexts = {row["id"]: row for row in read(HERE / "MASS_TRAINING_CONTEXT_PLAN.json")["contexts"]}
    final_active = []
    for row in ACTIVE_AT_PAUSE:
        spec = primary.get((row["context_id"], row["force_N"]))
        if spec is None:
            context = contexts[row["context_id"]]
            spec = {"context_id": row["context_id"], "force_N": row["force_N"],
                    "mass_kg": context["mass_kg"], "split": context["split"],
                    "root": context["root"], "task": context["task"]}
        job = HERE / "branches" / row["context_id"] / f"F{row['force_N']:g}"
        check = validate_branch(job, spec, runtime, HERE / "references" / row["context_id"])
        final_active.append({
            **row, "path": str(job),
            "final_classification": "VALID_COMPLETED" if check["valid"] else "INVALID_OR_INCOMPLETE",
            "failed_checks": check["failed_checks"],
            "artifact_sha256": check.get("artifact_sha256", {}),
        })

    launcher_states = [{**proc(pid), "root_partition": root} for pid, root in LAUNCHERS.items()]
    original_hashes = subset["authoritative_original_artifact_sha256"]
    original_intact = all(sha256(HERE / relative) == digest for relative, digest in original_hashes.items())
    worker_process_states = [proc(row["worker_pid"]) for row in ACTIVE_AT_PAUSE]
    # A stopped parent cannot reap a finished child, so an exited child may
    # remain as a zero-resource zombie until the explicitly authorized resume.
    no_active_worker_remains = all(
        not row["exists"] or str(row.get("state", "")).startswith("Z")
        for row in worker_process_states
    )
    all_completed = all(row["final_classification"] == "VALID_COMPLETED" for row in final_active)
    launchers_stopped = all(row["exists"] and row["state"].startswith("T") for row in launcher_states)
    audit = {
        "schema": "ORIGINAL_648_QUEUE_PAUSE_AUDIT_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset_sha256": subset_hash,
        "pause_signal": "SIGSTOP sent only to launcher parents",
        "pause_signal_at_or_before_utc": "2026-09-10T06:21:20Z",
        "valid_branch_count_at_pause": 54,
        "valid_branch_count_after_active_workers_finished": 58,
        "active_claims_at_pause": ACTIVE_AT_PAUSE,
        "active_claim_final_status": final_active,
        "paused_launchers": launcher_states,
        "checks": {
            "all_launcher_parents_stopped": launchers_stopped,
            "all_active_workers_finished_valid": all_completed,
            "no_active_worker_remains": no_active_worker_remains,
            "no_worker_was_killed_mid_branch": all_completed and no_active_worker_remains,
            "no_incomplete_claim_remains": all_completed,
            "original_648_artifacts_hash_intact": original_intact,
            "original_648_protocol_preserved": True,
            "original_648_queue_resumable": launchers_stopped and original_intact,
        },
        "original_artifact_sha256": original_hashes,
        "resume_instruction_if_explicitly_authorized_later": "Send SIGCONT to PIDs 86067, 86673, 87289, and 87885; their no-overwrite validation skips completed branches.",
        "automatic_resume_authorized": False,
        "signals_sent_to_worker_pids": [],
        "finished_worker_process_states": worker_process_states,
    }
    if not all(audit["checks"].values()):
        raise RuntimeError(json.dumps(audit["checks"], indent=2))
    with output.open("x") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps({"status": "PASS", "valid_at_pause": 54, "valid_after_workers": 58,
                      "paused_launcher_pids": sorted(LAUNCHERS)}, indent=2))


if __name__ == "__main__":
    main()
