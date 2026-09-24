"""Freeze and execute the bounded four-rollout cross-force intervention."""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
OLD = HERE.parent / "af_dump_original_restore_20260912"
HISTORY = HERE.parent / "af_dump_maxf8_confirmation_storage_20260913/inference_v4"
sys.path.insert(0, str(OLD))
from rootlocal_collection_contract import now, read, sha, verify_runtime, write

PYTHON = BASE / "venv_robotwin/bin/python3"
REPOSITORY = BASE / "RoboTwin"


def freeze():
    path = HERE / "CROSSFORCE_PROTOCOL.json"
    if path.exists():
        return read(path)
    cases = []
    definitions = [
        ("test_mu0.675_root200002_ps50200002", [3.8, 8.0], ["CROSS_LOW", "NATIVE_HIGH"]),
        ("test_mu0.675_root200002_ps60200002", [3.8, 8.0], ["NATIVE_LOW", "CROSS_HIGH"]),
    ]
    for name, forces, methods in definitions:
        historical = HISTORY / name
        cases.append(
            {
                "id": name,
                "forces_N": forces,
                "methods": methods,
                "historical_spec": str(historical / "SPEC.json"),
                "historical_spec_sha256": sha(historical / "SPEC.json"),
                "historical_lock": str(historical / "job/PREACTION_SELECTION_LOCK.json"),
                "historical_lock_sha256": sha(historical / "job/PREACTION_SELECTION_LOCK.json"),
                "historical_result": str(historical / "job/AF_INFERENCE_RESULT.json"),
                "historical_result_sha256": sha(historical / "job/AF_INFERENCE_RESULT.json"),
            }
        )
    protocol = {
        "stage": "POSTHOC_DEVELOPMENT_CROSS_FORCE",
        "created_utc": now(),
        "cases": cases,
        "planned_rollouts": 4,
        "model_training": False,
        "outcome_informed_case_selection": True,
        "source_hashes": {
            str(HERE / name): sha(HERE / name)
            for name in ["run_crossforce.py", "crossforce_worker.py", "CROSSFORCE_EXPERIMENT_CARD.md"]
        },
        "minimum_start_free_bytes": int(1.5 * 1024**3),
        "disk_reserve_bytes": int(0.75 * 1024**3),
    }
    write(path, protocol)
    return protocol


def execute():
    lock = (HERE / "CROSSFORCE_QUEUE.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = freeze()
    audits = []
    for case in protocol["cases"]:
        for path, expected in protocol["source_hashes"].items():
            if sha(path) != expected:
                raise ValueError("Queue source drift")
        for key in ["historical_spec", "historical_lock", "historical_result"]:
            if sha(case[key]) != case[key + "_sha256"]:
                raise ValueError("Historical artifact drift")
        output = HERE / ("cross_" + case["id"])
        if output.exists():
            if not (output / "CROSSFORCE_AUDIT.json").exists():
                raise RuntimeError("Retained partial case requires inspection")
            audits.append(read(output / "CROSSFORCE_AUDIT.json"))
            continue
        if shutil.disk_usage(HERE).free < protocol["minimum_start_free_bytes"]:
            raise RuntimeError("Storage gate: preserve and archive before continuing")
        output.mkdir()
        spec = read(case["historical_spec"])
        spec.update(crossforce=case, crossforce_protocol_sha256=sha(HERE / "CROSSFORCE_PROTOCOL.json"))
        write(output / "SPEC.json", spec)
        context = spec["context"]
        verify_runtime(context["runtime_manifest_path"], context["runtime_manifest_sha256"])
        command = [
            str(PYTHON),
            str(HERE / "crossforce_worker.py"),
            "--repo",
            str(REPOSITORY),
            "--out",
            str(output / "job"),
            "--native-ft",
            "--friction",
            str(context["friction"]),
            "--policy-seed",
            str(context["policy_seed"]),
        ]
        environment = os.environ.copy()
        environment.pop("AF_COLLECTION_CONTEXT", None)
        environment.update(
            AF_P4_PHYSICAL_SURFACE_CAMERA="1",
            AF_ORIGINAL_SQUEEZE_INNER="1",
            AF_INFERENCE_CONTEXT=str(output / "SPEC.json"),
            PYTHONUTF8="1",
            PATH=str(BASE / "runtime_bin") + os.pathsep + environment.get("PATH", "/usr/bin:/bin"),
        )
        write(output / "PROCESS.json", {"command": command, "cwd": str(REPOSITORY), "started_utc": now()})
        with (output / "worker.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write(output / "PID.json", {"pid": process.pid})
            stopped_for_disk = False
            while process.poll() is None:
                if shutil.disk_usage(HERE).free < protocol["disk_reserve_bytes"]:
                    stopped_for_disk = True
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                    break
                time.sleep(10)
        write(output / "EXIT.json", {"exit_code": process.returncode, "disk_guard_stop": stopped_for_disk, "finished_utc": now()})
        if process.returncode:
            raise RuntimeError("Retained worker failure: " + case["id"])
        job = output / "job"
        result = read(job / "CROSSFORCE_RESULT.json")
        seal = read(job / "PREACTION_SELECTION_LOCK.json")
        historical_lock = read(case["historical_lock"])
        if seal["state_sha256"] != historical_lock["state_sha256"] or seal["first_chunk_sha256"] != historical_lock["first_chunk_sha256"]:
            raise RuntimeError("Historical state/first chunk mismatch")
        from audit_original_collected_group import audit_branch, audit_query

        query_audit = audit_query(job / "query")
        branch_audits = [audit_branch(branch) for branch in sorted(job.glob("branch_*"))]
        historical = read(case["historical_result"])["outcomes"]
        matches = []
        for new in result["outcomes"]:
            old = [row for row in historical if row["force_N"] == new["force_N"]]
            for row in old:
                matches.append(
                    {
                        "force_N": new["force_N"],
                        "old_method": row["method"],
                        "old_success": row["success"],
                        "new_success": new["success"],
                        "old_result_sha256": row["result_sha256"],
                        "new_result_sha256": new["result_sha256"],
                        "result_exact": row["result_sha256"] == new["result_sha256"],
                    }
                )
        audit = {
            "passed": all(item["passed"] for item in branch_audits) and query_audit["passed"],
            "id": case["id"],
            "state_and_first_chunk_exact": True,
            "outcomes": result["outcomes"],
            "historical_matches": matches,
            "query_audit": query_audit,
            "branch_audits": branch_audits,
            "finished_utc": now(),
        }
        write(output / "CROSSFORCE_AUDIT.json", audit)
        if not audit["passed"]:
            raise RuntimeError("Audit failure: " + case["id"])
        audits.append(audit)
    complete = {
        "completed": True,
        "planned_rollouts": 4,
        "new_rollouts": 4,
        "cases": audits,
        "next_stage_not_launched": True,
        "finished_utc": now(),
    }
    write(HERE / "CROSSFORCE_COMPLETE.json", complete)
    print(json.dumps(complete))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.mode == "freeze":
        print(json.dumps(freeze()))
    else:
        execute()
