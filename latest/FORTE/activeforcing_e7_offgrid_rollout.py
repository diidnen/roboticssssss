#!/usr/bin/env python3
"""Execute the gate-authorized E7 exact-float DEV rollout manifest.

The parent mode is intentionally serial and requires an explicit host GPU
preflight acknowledgement.  Worker mode runs only inside the IsaacLab Python
environment and reuses the previously validated strict-preprobe wrapper.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import activeforcing_e7_continuous_repair as repair


ROOT = Path("/home/exouser/FORTE")
DEFAULT_OUT = ROOT / "activeforcing_e7_continuous_repair_20260902_113000"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
TABERO = Path("/home/exouser/Tabero")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
OPENPI = Path("/media/volume/newdata/exouser/openpi/src")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def worker() -> None:
    # This wrapper captures and restores the strict pre-probe last-hold state
    # and passes the native Python float directly to downstream_branch.
    import gnp_style_continuous_collect as collector
    collector.worker()


def build_target(source: dict, source_path: Path, run_dir: Path) -> tuple[Path, dict[str, dict]]:
    contexts: dict[str, list[dict]] = {}
    lookup = {}
    for req in source["requests"]:
        label = f"E7_{req['rollout_id']}_{req['planner']}"
        spec = {"force_N": float(req["requested_force_N"]), "repeat_index": int(req["repeat"]) - 1,
                "branch_label": label}
        contexts.setdefault(req["context_id"], []).append(spec)
        lookup[label] = req
    target = {
        "manifest_name": "E7_EXACT_FLOAT_OFFGRID_DEV_TARGETS", "split": "DEV", "no_test": True,
        "rounding": "none", "nearest_grid_snapping": False, "contexts": contexts,
        "source_rollout_manifest": str(source_path),
        "source_rollout_manifest_sha256": sha256(source_path),
    }
    path = run_dir / "E7_ISAAC_TARGET_MANIFEST.json"
    write_json(path, target)
    return path, lookup


def run_task(task: int, context_ids: list[str], target: Path, run_dir: Path, timeout_s: int) -> dict:
    log = run_dir / f"task{task}.log"
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(TABERO), str(OPENPI), str(ROOT)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(TABERO),
        "P5S0C_OUT": str(run_dir), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": str(task),
        "P5S0C_TARGET_MANIFEST": str(target), "P5S0C_CONTEXT_IDS": ",".join(context_ids),
        "P5S0C_SKIP_REPLAY": "1",
        "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
        "CONTINUOUS_STRICT_PREPROBE_WRAPPER": "1",
    })
    command = [str(ISAAC_PY), "-u", str(Path(__file__).resolve()), "--worker"]
    started = time.time()
    target_data = json.loads(target.read_text())
    expected_branches = sum(len(target_data["contexts"][cid]) for cid in context_ids)
    completed_after_result = False
    with log.open("w") as fh:
        process = subprocess.Popen(command, cwd=TABERO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout_s
        while process.poll() is None and time.time() < deadline:
            result_path = run_dir / f"task{task}" / "result.json"
            error_path = run_dir / f"task{task}" / "error.json"
            if result_path.exists() and not error_path.exists():
                try:
                    result = json.loads(result_path.read_text())
                    if int(result.get("contexts", 0)) == len(context_ids) and int(
                            result.get("primary_branches", 0)) == expected_branches:
                        completed_after_result = True
                        process.terminate()
                        break
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            time.sleep(2)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        else:
            process.wait()
    error_path = run_dir / f"task{task}" / "error.json"
    returncode = 0 if completed_after_result else int(process.returncode or 0)
    error = ""
    if error_path.exists():
        returncode = 1
        error = error_path.read_text(errors="replace")[:2000]
    if time.time() >= deadline and not completed_after_result:
        returncode = 124
    return {"task": task, "context_ids": context_ids, "expected_branches": expected_branches,
            "returncode": returncode, "process_returncode": process.returncode,
            "completed_after_result": completed_after_result, "error": error,
            "elapsed_wall_s": time.time() - started, "log": str(log)}


def extract_results(run_dir: Path, lookup: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for branch_path in sorted(run_dir.glob("task*/branches.csv")):
        branches = pd.read_csv(branch_path)
        for _, br in branches.iterrows():
            label = str(br.branch_label)
            if label not in lookup:
                continue
            req = lookup[label]
            telemetry = Path(br.telemetry_path)
            d = pd.read_csv(telemetry)
            _, load, _, _ = repair.loadbearing_mask(d)
            measured = d.measured_force_N.to_numpy(float)[load]
            commanded = float(req["requested_force_N"])
            rows.append({
                "rollout_id": req["rollout_id"], "context_id": req["context_id"], "root_id": req["root_id"],
                "task": int(req["task"]), "planner": req["planner"], "planning": req["planning"], "K": int(req["K"]),
                "requested_force_N": commanded, "controller_setpoint_N": commanded,
                "controller_exact_float_match": bool(float(br.requested_force_N) == commanded),
                "measured_loadbearing_force_N": float(np.mean(measured)) if len(measured) else np.nan,
                "tracking_error_N": float(np.mean(np.abs(measured - commanded))) if len(measured) else np.nan,
                "final_success": int(br.full_task_success_y), "failure_stage": str(br.get("failure_reason", "")),
                "state_parity": int(br.state_parity), "telemetry_path": str(telemetry), "telemetry_sha256": sha256(telemetry),
            })
    return pd.DataFrame(rows)


def execute(out: Path, timeout_s: int, gpu_preflight_ok: bool,
            manifest_name: str, run_name: str) -> None:
    if not gpu_preflight_ok:
        raise SystemExit("Refusing GPU work: run nvidia-smi/process audit, then pass --gpu-preflight-ok only when protected jobs are absent")
    source_path = out / manifest_name
    source = json.loads(source_path.read_text())
    gate = json.loads((out / "CONTINUOUS_DIRECT_VALID_GATE.json").read_text())
    is_full = source.get("protocol") == "E7_MATCHED_EXPECTED_UTILITY_FULL_DEV"
    authorization = source.get("authorized_to_execute_after_small_block", False) if is_full else source.get("authorized_to_execute", False)
    if gate["status"] != "PASS" or not authorization:
        raise SystemExit("Direct gate does not authorize Isaac execution")
    if source["split"] != "DEV" or not source["sealed_TEST_used"] is False:
        raise SystemExit("manifest split boundary failure")
    if is_full:
        small_audit_path = out / "E7_OFFGRID_DEV_EXECUTION_AUDIT.json"
        if not small_audit_path.exists() or json.loads(small_audit_path.read_text()).get("status") != "PASS":
            raise SystemExit("full matched rollout is blocked until exact-float small block PASS")
    run_dir = out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    target, lookup = build_target(source, source_path, run_dir)
    by_task = {}
    for req in source["requests"]:
        by_task.setdefault(int(req["task"]), set()).add(req["context_id"])
    records = []
    for task, ids in sorted(by_task.items()):
        records.append(run_task(task, sorted(ids), target, run_dir, timeout_s))
        if records[-1]["returncode"] != 0:
            break
    results = extract_results(run_dir, lookup)
    result_path = out / ("E7_MATCHED_FULL_DEV_RESULTS.csv" if is_full else "E7_OFFGRID_DEV_RESULTS.csv")
    results.to_csv(result_path, index=False)
    expected = len(source["requests"])
    exact = bool(len(results) == expected and results.controller_exact_float_match.all())
    parity = bool(len(results) == expected and results.state_parity.eq(1).all())
    tracking = bool(len(results) == expected and results.tracking_error_N.notna().all() and
                    (results.tracking_error_N <= 0.50).all())
    status = "PASS" if exact and parity and tracking and all(r["returncode"] == 0 for r in records) else "FAIL"
    audit = {"status": status, "expected_rollouts": expected, "observed_rollouts": len(results),
             "exact_float_match": exact, "state_parity": parity, "tracking_parity": tracking,
             "workers": records, "target_manifest": str(target), "target_manifest_sha256": sha256(target),
             "sealed_TEST_used": False}
    audit_path = out / ("E7_MATCHED_FULL_DEV_EXECUTION_AUDIT.json" if is_full else "E7_OFFGRID_DEV_EXECUTION_AUDIT.json")
    write_json(audit_path, audit)
    source["status"] = ("FULL_MATCHED_COMPLETE_" if is_full else "SMALL_BLOCK_COMPLETE_") + status
    source["execution_audit"] = str(audit_path)
    source["results"] = str(result_path)
    write_json(source_path, source)
    print(json.dumps(audit, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-preflight-ok", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-name", default="E7_OFFGRID_DEV_ROLLOUT_MANIFEST.json")
    parser.add_argument("--run-name", default="isaac_exact_float_small_block")
    parser.add_argument("--timeout-s", type=int, default=7200)
    args = parser.parse_args()
    if args.worker:
        worker()
    elif args.execute:
        execute(args.output, args.timeout_s, args.gpu_preflight_ok, args.manifest_name, args.run_name)
    else:
        raise SystemExit("use --execute after the host GPU/process preflight")


if __name__ == "__main__":
    main()
