#!/usr/bin/env python3
"""Single-worker guarded launcher for MASS development and final queues."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
ISAAC_PYTHON = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
FRICTION_SOURCE = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT")
CHECKPOINT_SHA256 = "0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17"
METHODS = ("REFERENCE", "ACTIVEFORCING_MASS", "GT_MASS", "FIXED_4")
ALLOWED_RETRY_REASONS = {"INFRASTRUCTURE_CRASH", "POLICY_SERVER_FAILURE", "CORRUPTED_SNAPSHOT", "INCOMPLETE_PROVENANCE"}


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def server_health(out, port):
    client = TABERO / "benchmarks/openpi/openpi-client/src"
    env = os.environ.copy(); env["PYTHONPATH"] = os.pathsep.join([str(client), env.get("PYTHONPATH", "")])
    code = ("import json; from openpi_client.websocket_client_policy import WebsocketClientPolicy; "
            f"p=WebsocketClientPolicy('127.0.0.1',{port}); m=p.get_server_metadata(); p._ws.close(); print(json.dumps(m))")
    proc = subprocess.run([str(ISAAC_PYTHON), "-c", code], env=env, text=True, capture_output=True, timeout=30)
    if proc.returncode: return False, {"exit_code": proc.returncode, "stderr": proc.stderr[-2000:]}
    metadata = json.loads(proc.stdout.strip().splitlines()[-1])
    valid = metadata.get("checkpoint_sha256") == CHECKPOINT_SHA256 and metadata.get("checkpoint_loaded") is True and metadata.get("downstream_action_source") == "ONLINE_VLA"
    return valid, metadata


def health(out, port):
    attempts = []
    gpu = None
    for attempt in range(10):
        proc = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                              text=True, capture_output=True)
        attempts.append({"attempt": attempt + 1, "exit_code": proc.returncode,
                         "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()})
        if proc.returncode == 0:
            free, utilization = [int(x.strip()) for x in proc.stdout.split(",")]
            gpu = {"free_mib": free, "utilization_percent": utilization}; break
        time.sleep(0.5)
    server_valid, evidence = server_health(out, port)
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "NVML_ATTEMPTS": attempts,
              "NVML_RESULT": gpu, "POLICY_SERVER_VALID": server_valid, "POLICY_SERVER_EVIDENCE": evidence,
              "RESOURCE_GATE_VALID": bool(gpu and gpu["free_mib"] >= 10000 and gpu["utilization_percent"] <= 70)}
    with (Path(out) / "MASS_GPU_HEALTH_LOG.jsonl").open("a") as stream: stream.write(json.dumps(result, sort_keys=True) + "\n")
    if not result["RESOURCE_GATE_VALID"] or not server_valid: raise RuntimeError("MASS online resource/server gate failed: " + json.dumps(result))
    return result


def job_path(out, context_id, method):
    base = out / ("references" if method == "REFERENCE" else "branches")
    name = context_id if method == "REFERENCE" else f"{context_id}__{method}"
    canonical = base / name
    if not canonical.exists(): return canonical
    completion = canonical / "WORKER_COMPLETION.json"
    if completion.exists() and read(completion).get("logical_success"): return canonical
    authorization = canonical / "RETRY_AUTHORIZATION.json"
    if not authorization.exists(): raise RuntimeError("invalid prior attempt requires forensic authorization: " + str(canonical))
    permit = read(authorization)
    if permit.get("allowed") is not True or permit.get("reason") not in ALLOWED_RETRY_REASONS: raise RuntimeError("retry forbidden")
    retry_root = out / "retries" / name; retry_root.mkdir(parents=True, exist_ok=True); attempt = 2
    while (retry_root / f"attempt_{attempt}").exists(): attempt += 1
    return retry_root / f"attempt_{attempt}"


def launch(out, plan_path, manifest_path, index, method, port):
    out, plan_path, manifest_path = Path(out), Path(plan_path), Path(manifest_path)
    if method not in METHODS: raise ValueError("method outside MASS protocol")
    plan = read(plan_path); context = plan["contexts"][index]
    job = job_path(out, context["id"], method)
    if (job / "WORKER_COMPLETION.json").exists() and read(job / "WORKER_COMPLETION.json").get("logical_success"): return job
    preflight = health(out, port); job.mkdir(parents=True, exist_ok=False); write(job / "GPU_HEALTH_PREFLIGHT.json", preflight)
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", TABERO_ROOT=str(TABERO),
        PYTHONPATH=os.pathsep.join([str(FRICTION_SOURCE), str(HERE), str(ROOT), str(TABERO),
                                   str(TABERO / "benchmarks/openpi/openpi-client/src")]),
        HDF5_TRAJ_SOURCE_DIR=str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        LIBERO_CONFIG_DIR=str(TABERO / "benchmarks/datasets/libero/config"),
        LIBERO_ASSETS_DATA_DIR=str(TABERO / "benchmarks/datasets/libero/USD"))
    command = [str(ISAAC_PYTHON), "-u", str(HERE / "mass_online_worker.py"), "--out", str(out),
               "--job", str(job), "--plan", str(plan_path), "--runtime-manifest", str(manifest_path),
               "--context", str(index), "--method", method, "--port", str(port)]
    with (job / "WORKER.log").open("x") as log:
        process = subprocess.Popen(command, cwd=TABERO, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(job / "PROCESS.json", {"pid": process.pid, "command": command,
                                     "started_utc": datetime.now(timezone.utc).isoformat()})
        print("START", job, process.pid, flush=True); code = process.wait()
    completion = read(job / "WORKER_COMPLETION.json") if (job / "WORKER_COMPLETION.json").exists() else None
    write(job / "PROCESS_EXIT.json", {"exit_code": code, "completion": completion})
    print("END", job, completion, flush=True)
    if not completion or not completion.get("logical_success"): raise RuntimeError("invalid MASS online attempt retained")
    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True); parser.add_argument("--plan", required=True)
    parser.add_argument("--runtime-manifest", required=True); parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--method", choices=METHODS, required=True); parser.add_argument("--port", type=int, default=18885)
    args = parser.parse_args(); launch(args.out, args.plan, args.runtime_manifest, args.context, args.method, args.port)
