from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
REPO = BASE / "RoboTwin"
PYTHON = BASE / "venv_robotwin/bin/python3"
INFER = HERE / "run_liftstyle_context.py"
PARALLEL = max(1, int(os.environ.get("AF_SCRIPTED_PARALLEL", "4")))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def status(name: str, **fields) -> None:
    write(HERE / "STATUS.json", {"status": name, "updated_utc": now(), **fields})


def near(a, b, tol=1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def verify_plan():
    plan = HERE / "plan"
    lock = read(plan / "FREEZE_LOCK.json")
    checks = {
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
        "main_contexts_sha256": sha(plan / "MAIN_CONTEXTS.json"),
    }
    if checks != lock:
        raise ValueError("Frozen liftstyle v3 plan changed")
    protocol = read(plan / "PROTOCOL.json")
    for path, digest in protocol["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen source changed: " + path)
    smoke = read(plan / "SMOKE_CONTEXT.json")
    contexts = read(plan / "MAIN_CONTEXTS.json")
    n_force = len(protocol["forces_N"])
    n_ctx = int(protocol["main_contexts"])
    n_roll = int(protocol["main_rollouts"])
    if n_force != 32 or n_ctx != 24 or n_roll != 768:
        raise ValueError("Main denominator changed")
    if len(contexts) != n_ctx or sum(len(c["forces_N"]) for c in contexts) != n_roll:
        raise ValueError("Main contexts do not match protocol")
    if protocol.get("version") != "AF_DUMP_LIFTSTYLE_FEAS_V3":
        raise ValueError("Unexpected protocol version")
    return protocol, smoke, contexts


def compact(result: dict) -> dict:
    return {
        "context": result["context"],
        "completed_utc": now(),
        "outcomes": result["outcomes"],
        "motion_mode": "liftstyle_snapshot_fork_remainder",
        "matched_prefix_exact": result.get("matched_prefix_exact"),
        "all_failures_retained": True,
    }


def run_one(context: dict, category: str) -> dict:
    raw_root = HERE / f"{category}_raw"
    records = HERE / f"{category}_records"
    kept = HERE / f"{category}_kept"
    case = raw_root / context["id"]
    record_path = records / f"{context['id']}.json"
    kept_marker = kept / f"{context['id']}.json"
    if record_path.exists() and kept_marker.exists():
        return read(record_path)
    if case.exists():
        raise RuntimeError("Partial retained context requires inspection: " + str(case))
    case.mkdir(parents=True)
    write(case / "SPEC.json", {"context": context})
    command = [
        str(PYTHON),
        str(INFER),
        "--context",
        str(case / "SPEC.json"),
        "--out",
        str(case / "job"),
    ]
    write(case / "PROCESS.json", {"command": command, "cwd": str(REPO), "started_utc": now()})
    with (case / "worker.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env={**os.environ, "PYTHONUTF8": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write(case / "PID.json", {"pid": process.pid})
        process.wait()
    write(case / "EXIT.json", {"exit_code": process.returncode, "finished_utc": now()})
    result_path = case / "job/LIFTSTYLE_CONTEXT_RESULT.json"
    if process.returncode or not result_path.exists():
        raise RuntimeError("Lift-style context failed: " + context["id"])
    result = read(result_path)
    record = compact(result)
    write(case / "CASE_COMPACT.json", record)
    write(record_path, record)
    write(
        kept_marker,
        {
            "context_id": context["id"],
            "local_raw_path": str(case),
            "record_path": str(record_path),
            "retained_utc": now(),
        },
    )
    return record


def wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - half, center + half]


def aggregate(protocol: dict, records: list[dict]) -> dict:
    forces = [float(x) for x in protocol["forces_N"]]
    summary = []
    for force in forces:
        rows = [next(item for item in record["outcomes"] if near(item["force_N"], force)) for record in records]
        valid = [row for row in rows if not row.get("unstable_layout")]
        successes = sum(row["success"] for row in valid)
        contact = [row["contact_ratio"] for row in valid if row.get("contact_ratio") is not None]
        means = [row["measured_force_mean_n"] for row in valid if row.get("measured_force_mean_n") is not None]
        summary.append(
            {
                "force_N": force,
                "successes": successes,
                "contexts_valid": len(valid),
                "contexts_unstable": len(rows) - len(valid),
                "success_rate": successes / len(valid) if valid else None,
                "success_rate_wilson95": wilson(successes, len(valid)) if valid else None,
                "mean_contact_ratio": sum(contact) / len(contact) if contact else None,
                "mean_measured_force_n": sum(means) / len(means) if means else None,
            }
        )
    curves = []
    nondec = 0
    valid_curves = 0
    for record in records:
        if any(row.get("unstable_layout") for row in record["outcomes"]):
            curves.append({"context_id": record["context"]["id"], "unstable_layout": True, "nondecreasing": None})
            continue
        outcomes = {float(row["force_N"]): int(row["success"]) for row in record["outcomes"]}
        seq = [outcomes[force] for force in forces]
        ok = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
        nondec += int(ok)
        valid_curves += 1
        curves.append(
            {
                "context_id": record["context"]["id"],
                "success_by_force_N": {str(f): outcomes[f] for f in forces},
                "nondecreasing": ok,
            }
        )
    return {
        "completed": True,
        "contexts": len(records),
        "paired_rollouts": sum(len(r["outcomes"]) for r in records),
        "summary": summary,
        "per_context_curves": curves,
        "full_curve_nondecreasing": {
            "contexts": nondec,
            "total_valid": valid_curves,
            "fraction": nondec / valid_curves if valid_curves else None,
        },
        "motion_mode": "liftstyle_snapshot_fork_remainder",
        "force_band_N": protocol["force_band_N"],
        "claim_boundary": protocol["claim_boundary"],
        "finished_utc": now(),
    }


def smoke_gate(protocol: dict, record: dict) -> dict:
    rows = {float(row["force_N"]): row for row in record["outcomes"]}
    expected = [float(x) for x in protocol["smoke_forces_N"]]
    checks = {
        "commands_present": sorted(rows) == expected,
        "all_completed": all(not row.get("unstable_layout") for row in rows.values()),
        "matched_prefix_exact": bool(record.get("matched_prefix_exact")),
        "all_commands_ran": True,
        "force_limits_match_command": all(
            near(row["left_force_limit_n"], row["force_N"]) and near(row["right_force_limit_n"], row["force_N"])
            for row in rows.values()
            if not row.get("unstable_layout")
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "outcomes": record["outcomes"], "finished_utc": now()}


def main() -> None:
    lock_file = (HERE / "LIFTSTYLE_V3_QUEUE.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("A live liftstyle v3 queue already holds the lock") from error
    protocol, smoke, contexts = verify_plan()
    n_force = len(protocol["forces_N"])
    status("smoke_running", completed_contexts=0, completed_rollouts=0, current_context=smoke["id"], parallel=PARALLEL)
    smoke_record = run_one(smoke, "smoke")
    admission = smoke_gate(protocol, smoke_record)
    write(HERE / "SMOKE_ADMISSION.json", admission)
    if not admission["passed"]:
        status("smoke_failed", completed_contexts=0, completed_rollouts=0, current_context=None)
        raise RuntimeError("liftstyle v3 smoke failed; main not started")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    records_by_id = {}
    status("main_running", completed_contexts=0, completed_rollouts=0, current_context=contexts[0]["id"], parallel=PARALLEL)
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {pool.submit(run_one, context, "main"): context for context in contexts}
        done = 0
        for future in as_completed(futures):
            context = futures[future]
            record = future.result()
            records_by_id[context["id"]] = record
            done += 1
            verify_plan()
            status(
                "main_running",
                completed_contexts=done,
                completed_rollouts=done * n_force,
                current_context=context["id"],
                parallel=PARALLEL,
            )
    records = [records_by_id[c["id"]] for c in contexts]
    final = aggregate(protocol, records)
    write(HERE / "FINAL_RESULTS.json", final)
    status("complete", completed_contexts=24, completed_rollouts=768, current_context=None)
    print(json.dumps({"completed": True, "contexts": 24, "rollouts": 768}, indent=2), flush=True)


if __name__ == "__main__":
    main()
