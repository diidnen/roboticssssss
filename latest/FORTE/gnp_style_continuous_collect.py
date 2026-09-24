#!/usr/bin/env python3
"""Strict-preprobe GNP-style continuous-force collector for TRAIN.

The authoritative P5-S0-C runner is reused without changing its controller or
success definition.  A wrapper captures the scene after the final pre-probe
``hold`` step, lets the frozen P4-B probe finish, and then restores the captured
scene before P5-S0-C saves its branch snapshot.  Consequently every downstream
force branch starts from the strict pre-probe state while the probe trace remains
available only for the frozen friction estimator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


TABERO = Path("/home/exouser/Tabero")
RUNNER = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA")
OPENPI = Path("/media/volume/newdata/exouser/openpi/src")
PREPROBE_STEP = 45 + 35 + 70 + 40


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        if rows:
            w.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def append_capture_row(out: Path, row: dict) -> None:
    path = out / f"task{row['task']}" / "strict_preprobe_capture.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def worker() -> None:
    """Run inside the IsaacLab Python environment."""
    import torch

    p5 = load_module("p5_continuous_strict", RUNNER)
    original_import = p5.import_p4_probe

    def strict_import(task_id: int):
        p4 = original_import(task_id)
        original_run = p4.run_probe_episode

        def strict_run(env, *, seed_idx: int, mu: float, trial_id: str, dt: float):
            base_step = env.step
            counter = {"n": 0, "snapshot": None}

            def capture_step(action):
                result = base_step(action)
                counter["n"] += 1
                if counter["n"] == PREPROBE_STEP:
                    counter["snapshot"] = env.scene.get_state(is_relative=True)
                return result

            env.step = capture_step
            try:
                rows, rec = original_run(env, seed_idx=seed_idx, mu=mu, trial_id=trial_id, dt=dt)
            finally:
                env.step = base_step
            if counter["snapshot"] is None:
                raise RuntimeError(f"STRICT_PREPROBE_CAPTURE_MISSING:{trial_id}:steps={counter['n']}")
            postprobe_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            env.reset_to(counter["snapshot"], torch.tensor([0], device=env.device), is_relative=True)
            preprobe_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            # A second restore verifies that the saved snapshot is stable before
            # the authoritative runner captures it for all branch restores.
            env.reset_to(counter["snapshot"], torch.tensor([0], device=env.device), is_relative=True)
            preprobe_hash_2 = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            if preprobe_hash != preprobe_hash_2:
                raise RuntimeError(f"STRICT_PREPROBE_RESTORE_UNSTABLE:{trial_id}")
            append_capture_row(Path(os.environ["P5S0C_OUT"]), {
                "task": int(task_id), "context_id": trial_id, "seed": int(seed_idx),
                "friction": float(mu), "capture_step": PREPROBE_STEP,
                "last_hold_step": int(max(r.step for r in rows if str(r.probe_phase) == "hold")),
                "probe_out_present": int(any(str(r.probe_phase) == "probe_out" for r in rows)),
                "preprobe_state_hash": preprobe_hash, "second_restore_hash": preprobe_hash_2,
                "postprobe_state_hash_forbidden": postprobe_hash,
                "preprobe_restore_stable": 1, "branch_snapshot_semantics": "STRICT_PREPROBE_LAST_HOLD",
            })
            return rows, rec

        p4.run_probe_episode = strict_run
        return p4

    p5.import_p4_probe = strict_import
    raise SystemExit(p5.worker_main())


def build_manifest(protocol_path: Path, out: Path) -> Path:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    contexts: dict[str, list[dict]] = {}
    rows: list[dict] = []
    for ctx in protocol["train_context_population"]:
        cid = ctx["context_id"]
        specs = []
        samples = sorted(ctx["continuous_force_samples"], key=lambda x: int(x["stratum_index"]))
        for sample in samples:
            force = float(sample["requested_force_N"])
            for repeat in range(1, int(protocol["continuous_collection"]["valid_repeats_per_force"]) + 1):
                tag = f"{force:.8f}".rstrip("0").rstrip(".").replace(".", "p")
                label = f"GNP_S{int(sample['stratum_index'])}_F{tag}_R{repeat}"
                spec = {"force_N": float(force), "repeat_index": repeat - 1, "branch_label": label}
                specs.append(spec)
                rows.append({
                    "context_id": cid, "root_id": ctx["root_id"], "task": int(ctx["task"]),
                    "friction_band": ctx["friction_band"], "friction": float(ctx["friction"]),
                    "stratum_index": int(sample["stratum_index"]),
                    "stratum_low_N": float(sample["stratum_low_N"]),
                    "stratum_high_N": float(sample["stratum_high_N"]),
                    "force_N": float(force), "repeat": repeat, "branch_label": label,
                    "expected_branch_id": f"{cid}_{label}_F{float(force):g}",
                    "decision_state": "STRICT_PREPROBE_LAST_HOLD", "scientific_retry": 0,
                })
        contexts[cid] = specs
    manifest = {
        "manifest_name": "GNP_STYLE_CONTINUOUS_TRAIN_TARGET_MANIFEST",
        "protocol_sha256": sha256(protocol_path), "runner": str(RUNNER),
        "runner_sha256": sha256(RUNNER), "contexts": contexts,
        "expected_contexts": len(contexts), "expected_branches": len(rows),
        "split": "TRAIN", "no_dev": True, "no_test": True,
        "branch_state": "restored strict pre-probe last-hold snapshot",
    }
    path = out / "CONTINUOUS_STRICT_PREPROBE_TARGET_MANIFEST.json"
    write_json(path, manifest)
    write_csv(out / "CONTINUOUS_STRICT_PREPROBE_TARGET_MANIFEST.csv", rows)
    return path


def run_one(out: Path, manifest: Path, task: int, context_ids: list[str], timeout_s: int) -> dict:
    log = out / "logs" / f"task{task}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(TABERO), str(OPENPI)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(TABERO),
        "P5S0C_OUT": str(out), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": str(task),
        "P5S0C_TARGET_MANIFEST": str(manifest), "P5S0C_CONTEXT_IDS": ",".join(context_ids),
        "P5S0C_SKIP_REPLAY": "1",
        "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
        "CONTINUOUS_STRICT_PREPROBE_WRAPPER": "1",
    })
    command = [str(ISAAC_PY), "-u", str(Path(__file__).resolve()), "--worker"]
    start = time.time()
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(command, cwd=TABERO, env=env, stdout=f, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout_s
        expected_branches = sum(len(json.loads(manifest.read_text())["contexts"][cid]) for cid in context_ids)
        completed_after_result = False
        while p.poll() is None and time.time() < deadline:
            result_path = out / f"task{task}" / "result.json"
            error_path = out / f"task{task}" / "error.json"
            if result_path.exists() and not error_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if int(result.get("contexts", 0)) == len(context_ids) and int(result.get("primary_branches", 0)) == expected_branches:
                        completed_after_result = True
                        p.terminate()
                        break
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            time.sleep(2)
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
        else:
            p.wait()
    error_path = out / f"task{task}" / "error.json"
    effective = 0 if completed_after_result else int(p.returncode or 0)
    error = ""
    if error_path.exists():
        effective = 1
        error = error_path.read_text(encoding="utf-8", errors="replace")[:2000]
    return {
        "task": task, "context_ids": context_ids, "expected_contexts": len(context_ids),
        "expected_branches": expected_branches, "returncode": effective,
        "process_returncode": p.returncode, "elapsed_wall_s": time.time() - start,
        "completed_after_result": completed_after_result, "log": str(log), "error": error,
    }


def orchestrate(protocol: Path, out: Path, timeout_s: int) -> None:
    if not protocol.exists():
        raise FileNotFoundError(protocol)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(protocol, out)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    by_task: dict[int, list[str]] = defaultdict(list)
    protocol_obj = json.loads(protocol.read_text(encoding="utf-8"))
    for ctx in protocol_obj["train_context_population"]:
        by_task[int(ctx["task"])].append(ctx["context_id"])
    processes = []
    # Launch each task independently, matching the previously validated
    # four-worker corrected-telemetry collection pattern.
    for task, ids in sorted(by_task.items()):
        processes.append((task, ids, subprocess.Popen([
            sys.executable, "-u", str(Path(__file__).resolve()), "--one-task",
            "--protocol", str(protocol), "--out", str(out), "--task", str(task),
            "--timeout-s", str(timeout_s),
        ], cwd=Path(__file__).resolve().parent)))
    records = []
    for task, ids, proc in processes:
        proc.wait()
        record_path = out / f"TASK{task}_COLLECTOR_RECORD.json"
        if record_path.exists():
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
        else:
            records.append({"task": task, "context_ids": ids, "returncode": proc.returncode, "error": "collector record missing"})
    write_json(out / "CONTINUOUS_COLLECTION_RUN_MANIFEST.json", {
        "status": "COMPLETE" if all(int(r.get("returncode", 1)) == 0 for r in records) else "ENGINEERING_FAILURE",
        "protocol": str(protocol), "protocol_sha256": sha256(protocol),
        "target_manifest": str(manifest), "target_manifest_sha256": sha256(manifest),
        "expected_contexts": len(data["contexts"]), "expected_branches": int(data["expected_branches"]),
        "workers": records, "scientific_failures_retried": 0,
    })
    write_csv(out / "CONTINUOUS_COLLECTION_RUN_MANIFEST.csv", records)
    if any(int(r.get("returncode", 1)) != 0 for r in records):
        raise RuntimeError("one or more collection workers failed; inspect run manifest")


def one_task(args) -> None:
    # The parent freezes this once before launching task workers.  Rebuilding
    # it concurrently here creates a partial-write/read race on large TRAIN
    # manifests and is scientifically unnecessary.
    manifest = Path(args.out) / "CONTINUOUS_STRICT_PREPROBE_TARGET_MANIFEST.json"
    if not manifest.exists():
        raise FileNotFoundError(f"parent-frozen target manifest missing: {manifest}")
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    ids = [c["context_id"] for c in protocol["train_context_population"] if int(c["task"]) == int(args.task)]
    rec = run_one(Path(args.out), manifest, int(args.task), ids, int(args.timeout_s))
    write_json(Path(args.out) / f"TASK{args.task}_COLLECTOR_RECORD.json", rec)
    if rec["returncode"] != 0:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--one-task", action="store_true")
    ap.add_argument("--protocol")
    ap.add_argument("--out")
    ap.add_argument("--task", type=int)
    ap.add_argument("--timeout-s", type=int, default=10800)
    args = ap.parse_args()
    if args.worker:
        worker()
    elif args.one_task:
        one_task(args)
    else:
        orchestrate(Path(args.protocol), Path(args.out), int(args.timeout_s))


if __name__ == "__main__":
    main()
