#!/usr/bin/env python3
"""CPU-only synthetic unit test for nominal DEV analysis and immutable promotion."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from validate_and_lock_task5_onboarded_candidate import sha256, tree_manifest


ART = Path(__file__).resolve().parent
CONFIG = "pi0_lora_tacfield_e3_task5_5demo_7dpf"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="e3_post_onboarding_test_") as temp:
        root = Path(temp)
        checkpoint = root / "experiment" / "999"
        (checkpoint / "params").mkdir(parents=True)
        (checkpoint / "_CHECKPOINT_METADATA").write_text("unit\n")
        (checkpoint / "params" / "_METADATA").write_text("unit params\n")
        (checkpoint / "params" / "manifest.ocdbt").write_text("unit manifest\n")
        fingerprint, files = tree_manifest(checkpoint)
        norm = root / "norm_stats.json"
        dims = {"state": 7, "actions": 13, "tactile_prefix": 396}
        norm.write_text(json.dumps({"norm_stats": {
            name: {"mean": [0.0] * dim, "std": [1.0] * dim, "q01": [0.0] * dim, "q99": [1.0] * dim}
            for name, dim in dims.items()
        }}))
        lock = root / "CANDIDATE_LOCK.json"
        lock.write_text(json.dumps({
            "status": "CANDIDATE_LOCKED_FOR_NOMINAL_DEV_NOT_FINAL_FREEZE",
            "task": "libero_10/task5",
            "config": CONFIG,
            "selected_step": 999,
            "checkpoint_dir": str(checkpoint),
            "checkpoint_tree_sha256": fingerprint,
            "checkpoint_files": files,
            "norm_stats": str(norm),
            "norm_stats_sha256": sha256(norm),
            "openpi_commit": "31049447d685cb36ddaeddda4f1d62fec0bc6392",
        }, sort_keys=True))
        lock_sha = sha256(lock)

        evaluation = root / "nominal_dev"
        e5_root = root / "e5"
        e5_root.mkdir()
        (e5_root / "CORE_GPU_SCHEDULER.jsonl").write_text(json.dumps({
            "event": "scheduler_start", "pid": 424242, "utc": "2026-01-01T00:00:00Z"
        }) + "\n")
        coverage = e5_root / "E5_COVERAGE_STATUS.json"
        coverage.write_text(json.dumps({"status": "PASS", "accepted_unique_rollouts": 0}))
        exit_evidence = root / "E5_NATURAL_EXIT.json"
        exit_evidence.write_text(json.dumps({
            "schema": "E5_SCHEDULER_NATURAL_EXIT_EVIDENCE_V1",
            "scheduler_pid": 424242, "exit_kind": "NATURAL", "exit_code": 0,
        }, sort_keys=True))
        issued = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for index, seed in enumerate((7600, 7601, 7602, 7603, 7604)):
            logs = evaluation / f"NOMINAL_DEV_root{seed}_mu0.6_F8N" / "logs"
            logs.mkdir(parents=True)
            (logs.parent / "CANDIDATE_LOCK.json").write_bytes(lock.read_bytes())
            gate_copy = logs.parent / "COORDINATOR_FINAL_GATE_DYNAMIC.json"
            gate_copy.write_text(json.dumps({
                "schema": "E3_COORDINATOR_FINAL_GATE_DYNAMIC_V1",
                "status": "FINAL_GATE_PASS",
                "authorized": True,
                "operation": "E3_TASK5_NOMINAL_DEV_CELL",
                "root_seed": seed,
                "issued_at_utc": issued,
                "e5_scheduler_pid": 424242,
                "e5_scheduler_exit": "NATURAL",
                "e5_scheduler_exit_code": 0,
                "e5_scheduler_exit_evidence": {"path": str(exit_evidence), "sha256": sha256(exit_evidence)},
                "e5_silence_seconds": 10,
                "active_e5_pids": [],
                "active_e5_processes": [],
                "e5_last_activity_utc": "2026-01-01T00:00:00Z",
                "protected_processes_audited": True,
                "protected_process_snapshot": [],
                "duplicate_scan_empty": True,
                "resource_scan_pass": True,
                "freshness": {"fresh_at_issue": True, "max_age_seconds": 60},
                "e5_coverage_artifact": {"path": str(coverage), "sha256": sha256(coverage), "status": "PASS"},
            }, sort_keys=True))
            episode = {
                "method": "E3_POST_ONBOARDING_NOMINAL_DEV",
                "e3_policy_fingerprint": fingerprint,
                "e3_candidate_lock_sha256": lock_sha,
                "e3_policy_config": CONFIG,
                "e3_checkpoint_step": 999,
                "e3_coordinator_dynamic_gate_sha256": sha256(gate_copy),
                "task_id": 5,
                "friction": 0.6,
                "peak_predicted_force_slot_N": 8.0,
                "root_state_hash": hashlib.sha256(str(seed).encode()).hexdigest(),
                "official_success": int(index < 3),
                "pick_success": 1,
                "lift_success": 1,
                "transport_success": int(index < 3),
                "place_success": int(index < 3),
                "steps": 200,
                "mean_measured_force_N": 4.0,
                "peak_measured_force_N": 8.0,
            }
            with (logs / "unit_episodes.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(episode))
                writer.writeheader(); writer.writerow(episode)
            with (logs / "unit_steps.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["contact"])
                writer.writeheader(); writer.writerow({"contact": 1})

        subprocess.run([
            sys.executable, str(ART / "analyze_task5_post_onboarding_nominal_dev.py"),
            str(evaluation), str(lock), "--e5-root", str(e5_root),
        ], check=True, capture_output=True, text=True)
        gate = evaluation / "E3_TASK5_POST_ONBOARDING_NOMINAL_DEV_GATE.json"
        assert json.loads(gate.read_text())["gate_pass"] is True
        frozen = root / "FROZEN_CHECKPOINT.json"
        subprocess.run([
            sys.executable, str(ART / "promote_task5_onboarded_candidate_after_nominal_dev.py"),
            str(lock), str(gate), str(frozen),
        ], check=True, capture_output=True, text=True)
        result = json.loads(frozen.read_text())
        assert result["status"] == "TASK5_ONBOARDED_PI0_FROZEN_AFTER_NOMINAL_DEV_PASS"
        assert result["checkpoint_tree_sha256"] == fingerprint
    print("E3_POST_ONBOARDING_ANALYZER_PROMOTION_UNIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
