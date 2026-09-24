"""Single-worker launcher for the frozen unseen-friction queue."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, TABERO, read, write


NVML_QUERY = ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
MIN_FREE_MIB = 10000
ALLOWED_RETRY_REASONS = {
    "INFRASTRUCTURE_CRASH", "ENOSPC_BEFORE_VALID_PROVENANCE", "POLICY_SERVER_FAILURE",
    "CORRUPTED_SNAPSHOT", "INCOMPLETE_PROVENANCE",
}


def policy_server_health(python: Path, out: Path, port: int):
    client_path = TABERO / "benchmarks/openpi/openpi-client/src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(client_path), env.get("PYTHONPATH", "")])
    code = (
        "import json; from openpi_client.websocket_client_policy import WebsocketClientPolicy; "
        f"p=WebsocketClientPolicy('127.0.0.1',{port}); "
        "m=p.get_server_metadata(); p._ws.close(); print(json.dumps(m))"
    )
    proc = subprocess.run([str(python), "-c", code], env=env, text=True, capture_output=True, timeout=30)
    if proc.returncode != 0:
        return False, {"exit_code": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    metadata = json.loads(proc.stdout.strip().splitlines()[-1])
    expected = read(out / f"POLICY_SERVER_{port}" / "SERVER_READY.json")
    valid = (metadata == expected and metadata.get("pid") == expected.get("pid")
             and metadata.get("checkpoint_sha256") == "0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17"
             and metadata.get("downstream_action_source") == "ONLINE_VLA"
             and metadata.get("checkpoint_loaded") is True)
    return valid, {"exit_code": proc.returncode, "metadata": metadata, "expected_match": metadata == expected}


def gpu_health(out: Path, python: Path, port: int):
    attempts = []
    value = None
    for attempt in range(10):
        proc = subprocess.run(NVML_QUERY, text=True, capture_output=True)
        row = {"attempt": attempt + 1, "exit_code": proc.returncode,
               "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
        attempts.append(row)
        if proc.returncode == 0:
            free, utilization = [int(x.strip()) for x in proc.stdout.split(",")]
            value = {"free_mib": free, "utilization_percent": utilization}
            break
        time.sleep(0.5)
    server_valid, server_evidence = policy_server_health(python, out, port)
    health = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "NVML_ATTEMPTS": attempts,
        "NVML_RESULT": value,
        f"POLICY_SERVER_{port}_VALID": server_valid,
        "POLICY_SERVER_EVIDENCE": server_evidence,
        "RESOURCE_GATE_VALID": bool(value and value["free_mib"] >= MIN_FREE_MIB and value["utilization_percent"] <= 70),
    }
    with (out / "GPU_HEALTH_GATE_LOG.jsonl").open("a") as handle:
        handle.write(json.dumps(health, sort_keys=True) + "\n")
    if not health["RESOURCE_GATE_VALID"]:
        raise RuntimeError("GPU resource gate failed: " + json.dumps(health, sort_keys=True))
    if not server_valid:
        raise RuntimeError("Frozen policy-server health gate failed: " + json.dumps(health, sort_keys=True))
    return health


def _job_path(out: Path, context_id: str, method: str):
    base = out / ("references" if method == "REFERENCE" else "branches")
    name = context_id if method == "REFERENCE" else f"{context_id}__{method}"
    canonical = base / name
    if not canonical.exists():
        return canonical
    completion = canonical / "WORKER_COMPLETION.json"
    if completion.exists() and read(completion).get("logical_success"):
        return canonical
    authorization = canonical / "RETRY_AUTHORIZATION.json"
    if not authorization.exists():
        raise RuntimeError(f"Prior invalid attempt retained without retry authorization: {canonical}")
    permit = read(authorization)
    if permit.get("allowed") is not True or permit.get("reason") not in ALLOWED_RETRY_REASONS:
        raise RuntimeError(f"Retry forbidden by frozen policy: {canonical}")
    retry_root = out / "retries" / name
    retry_root.mkdir(parents=True, exist_ok=True)
    attempt = 2
    while (retry_root / f"attempt_{attempt}").exists():
        previous = retry_root / f"attempt_{attempt}"
        done = previous / "WORKER_COMPLETION.json"
        if done.exists() and read(done).get("logical_success"):
            return previous
        auth = previous / "RETRY_AUTHORIZATION.json"
        if not auth.exists() or read(auth).get("allowed") is not True:
            raise RuntimeError(f"Retry chain stopped without authorization: {previous}")
        attempt += 1
    return retry_root / f"attempt_{attempt}"


def launch(out: Path, index: int, method: str, port: int = 18885):
    plan = read(out / "DEV_PLAN.json")["contexts"][index]
    allowed_roots = set(read(out / "CONTINUOUS_FRICTION_ROOTS.json")["roots"])
    if plan["root"] not in allowed_roots:
        raise RuntimeError("Context root outside frozen two-root plan")
    job = _job_path(out, plan["id"], method)
    completion = job / "WORKER_COMPLETION.json"
    if completion.exists() and read(completion).get("logical_success"):
        return job
    python = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
    health = gpu_health(out, python, port)
    job.mkdir(parents=True, exist_ok=False)
    write(job / "GPU_HEALTH_PREFLIGHT.json", health)
    env = os.environ.copy()
    env.update(
        PYTHONNOUSERSITE="1", OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y",
        PYTHONPATH=os.pathsep.join([
            str(python.parent.parent / "lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64"),
            str(out / "SOURCE_SNAPSHOT"), str(ROOT), str(TABERO),
            str(TABERO / "benchmarks/openpi/openpi-client/src")
        ]),
        TABERO_ROOT=str(TABERO),
        HDF5_TRAJ_SOURCE_DIR=str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        LIBERO_CONFIG_DIR=str(TABERO / "benchmarks/datasets/libero/config"),
        LIBERO_ASSETS_DATA_DIR=str(TABERO / "benchmarks/datasets/libero/USD"),
    )
    command = [str(python), "-u", str(out / "SOURCE_SNAPSHOT/worker.py"),
               "--out", str(out), "--job", str(job), "--context", str(index),
               "--method", method, "--port", str(port)]
    with (job / "WORKER.log").open("x") as log:
        proc = subprocess.Popen(command, cwd=TABERO, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(job / "PROCESS.json", {
            "pid": proc.pid, "command": command, "started_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_health": health,
        })
        print("START", job, proc.pid, flush=True)
        rc = proc.wait()
    done = read(completion) if completion.exists() else None
    write(job / "PROCESS_EXIT.json", {"exit_code": rc, "completion": done})
    print("END", job, done, flush=True)
    if not done or not done.get("logical_success"):
        raise RuntimeError(f"Invalid attempt retained; classification required before any retry: {job}")
    return job
