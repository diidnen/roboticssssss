#!/usr/bin/env python3
"""Prospective strict-preprobe RGB/feature/outcome collector.

The authoritative P5-S0-C runner supplies the frozen controller and corrected
physical telemetry. This wrapper changes only the population/branch manifest
and captures a disk-restorable state plus the frozen π0 visual side channel at
the last pre-probe hold step.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


TABERO = Path("/home/exouser/Tabero")
RUNNER = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64")
CLIENT_SRC = TABERO / "benchmarks/openpi/openpi-client/src"
FORTE = Path("/home/exouser/FORTE")
PREPROBE_STEP = 45 + 35 + 70 + 40
TASKS = [0, 1, 5, 6]
INSTRUCTIONS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def sha256_bytes(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def clone_cpu(x):
    import torch
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: clone_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clone_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(clone_cpu(v) for v in x)
    return copy.deepcopy(x)


def move_device(x, device):
    import torch
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_device(v, device) for v in x)
    return x


def save_visual_capture(env, p5, task_id: int, context_id: str, snapshot, step: int, client) -> dict:
    base = env.scene["agentview_cam"].data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8, copy=True)
    wrist = env.scene["eye_in_hand_cam"].data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8, copy=True)
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy().astype(np.float32, copy=True)
    request = {
        "image": base,
        "wrist_image": wrist,
        "state": eef,
        "tactile_marker_motion": np.zeros((9, 198, 2), dtype=np.float32),
        "prompt": INSTRUCTIONS[task_id],
    }
    response = client.infer(request)
    feature = np.asarray(response["diagnostics_visual_feature"], dtype=np.float32)
    if feature.shape != (4096,):
        raise RuntimeError(f"unexpected visual feature shape {feature.shape} for {context_id}")
    out = Path(os.environ["PVP_COLLECTION_OUT"])
    split_dir = out / "visual"
    snap_dir = out / "snapshots"
    split_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"{context_id}_camera0_rgb.npy", base)
    np.save(split_dir / f"{context_id}_camera1_rgb.npy", wrist)
    np.save(split_dir / f"{context_id}_feature.npy", feature)
    snapshot_path = snap_dir / f"{context_id}.pt"
    import torch
    snapshot_cpu = clone_cpu(snapshot)
    torch.save(snapshot_cpu, snapshot_path)
    state_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
    camera_cfg = {
        "camera0_name": "agentview_cam", "camera1_name": "eye_in_hand_cam",
        "camera0_cfg": repr(env.scene["agentview_cam"].cfg),
        "camera1_cfg": repr(env.scene["eye_in_hand_cam"].cfg),
        "rgb_shape": list(base.shape), "rgb_dtype": str(base.dtype),
    }
    return {
        "context_id": context_id, "task": task_id, "capture_step": step,
        "camera0_rgb_path": str(split_dir / f"{context_id}_camera0_rgb.npy"),
        "camera1_rgb_path": str(split_dir / f"{context_id}_camera1_rgb.npy"),
        "visual_feature_path": str(split_dir / f"{context_id}_feature.npy"),
        "snapshot_path": str(snapshot_path),
        "camera0_rgb_sha256": sha256_bytes(base), "camera1_rgb_sha256": sha256_bytes(wrist),
        "visual_feature_sha256": sha256_bytes(feature), "snapshot_state_hash": state_hash,
        "visual_feature_shape": list(feature.shape), "visual_feature_dtype": str(feature.dtype),
        "camera_config_sha256": hashlib.sha256(json.dumps(camera_cfg, sort_keys=True).encode()).hexdigest(),
        "camera_config": camera_cfg, "timestamp_unix": time.time(),
        "timestamp_semantics": "after final pre-probe hold step and before any probe action",
        "same_x_for_all_branches": 1, "diagnostic_action_ignored": 1,
    }


def existing_visual_capture(out: Path, context_id: str) -> dict | None:
    """Return a previously committed capture for an interrupted context.

    A worker can be interrupted after the strict pre-probe capture but before
    the context row is committed.  Reusing that capture is important: it
    avoids making a second visual request and keeps the resumed branches tied
    to the originally persisted pre-probe state.
    """
    manifest = out / "visual_alignment_worker.csv"
    if not manifest.exists():
        return None
    rows = list(csv.DictReader(manifest.open(newline="")))
    matches = [r for r in rows if r.get("context_id") == context_id]
    if not matches:
        return None
    row = matches[-1]
    required = [
        row.get("snapshot_path", ""), row.get("camera0_rgb_path", ""),
        row.get("camera1_rgb_path", ""), row.get("visual_feature_path", ""),
    ]
    if not all(Path(x).exists() for x in required):
        return None
    return row


def worker() -> None:
    import torch
    p5 = load_module("prospective_p5_runner", RUNNER)
    # P5-S0-C's original writer rewrites cumulative in-memory rows.  Keep
    # resume runs append-safe so a worker that is intentionally restricted to
    # unfinished contexts cannot truncate earlier committed contexts.
    original_write_csv = p5.write_csv

    def resume_safe_write_csv(path, rows):
        path = Path(path)
        if path.name in {"context.csv", "branches.csv", "parity.csv", "root_state_parity.csv", "same_force_replay.csv"} and path.exists():
            try:
                old = list(csv.DictReader(path.open(newline="")))
                key = "context_id" if path.name == "context.csv" else ("branch_id" if path.name == "branches.csv" else None)
                if key:
                    seen = {str(r.get(key, "")) for r in old}
                    rows = old + [r for r in rows if str(r.get(key, "")) not in seen]
                elif path.name == "parity.csv":
                    seen = {(str(r.get("context_id", "")), str(r.get("branch_label", ""))) for r in old}
                    rows = old + [r for r in rows if (str(r.get("context_id", "")), str(r.get("branch_label", ""))) not in seen]
                elif path.name == "root_state_parity.csv":
                    seen = {str(r.get("context_id", "")) for r in old}
                    rows = old + [r for r in rows if str(r.get("context_id", "")) not in seen]
                else:
                    seen = {(str(r.get("context_id", "")), str(r.get("replay_index", ""))) for r in old}
                    rows = old + [r for r in rows if (str(r.get("context_id", "")), str(r.get("replay_index", ""))) not in seen]
            except Exception:
                # Preserve the authoritative writer if an old diagnostic file
                # is malformed; the later audit will report the issue.
                pass
        return original_write_csv(path, rows)

    p5.write_csv = resume_safe_write_csv
    target_path = Path(os.environ["P5S0C_TARGET_MANIFEST"])
    target = json.loads(target_path.read_text())["contexts"]
    context_rows = []
    with (Path(os.environ["PVP_COLLECTION_OUT"]) / "visual_alignment_worker.csv").open("a", newline="") as fh:
        visual_writer = None

        original_import = p5.import_p4_probe

        def strict_import(task_id: int):
            p4 = original_import(task_id)
            original_run = p4.run_probe_episode

            def strict_run(env, *, seed_idx: int, mu: float, trial_id: str, dt: float):
                nonlocal visual_writer
                base_step = env.step
                counter = {"n": 0, "snapshot": None, "capture": None}
                # The client is created lazily inside the Isaac worker so all
                # requests use the same frozen diagnostic server connection.
                from openpi_client.websocket_client_policy import WebsocketClientPolicy
                client = WebsocketClientPolicy("127.0.0.1", int(os.environ.get("VISUAL_PI0_PORT", "18881")))

                def capture_step(action):
                    result = base_step(action)
                    counter["n"] += 1
                    if counter["n"] == PREPROBE_STEP:
                        counter["snapshot"] = clone_cpu(env.scene.get_state(is_relative=True))
                        prior = existing_visual_capture(Path(os.environ["PVP_COLLECTION_OUT"]), trial_id)
                        if prior is not None:
                            # The previous worker died after committing the
                            # capture. Restore that exact state before the
                            # probe continues, and reuse the same x/RGB files.
                            disk_snapshot = torch.load(
                                prior["snapshot_path"], map_location="cpu", weights_only=False
                            )
                            env.reset_to(
                                move_device(disk_snapshot, env.device),
                                torch.tensor([0], device=env.device),
                                is_relative=True,
                            )
                            counter["snapshot"] = clone_cpu(disk_snapshot)
                            counter["capture"] = prior
                        else:
                            # Use the in-memory GPU state for the immediate
                            # probe return; the disk copy is independently
                            # checked below.
                            counter["capture"] = save_visual_capture(
                                env, p5, task_id, trial_id,
                                env.scene.get_state(is_relative=True),
                                counter["n"], client,
                            )
                    return result

                env.step = capture_step
                try:
                    rows, rec = original_run(env, seed_idx=seed_idx, mu=mu, trial_id=trial_id, dt=dt)
                finally:
                    env.step = base_step
                if counter["snapshot"] is None or counter["capture"] is None:
                    raise RuntimeError(f"PROSPECTIVE_CAPTURE_MISSING:{trial_id}:steps={counter['n']}")
                # Reload the genuinely persisted CPU snapshot, restore it, and
                # require exact hash parity before P5-S0-C branches continue.
                snapshot_path = Path(counter["capture"]["snapshot_path"])
                disk_snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
                env.reset_to(move_device(disk_snapshot, env.device), torch.tensor([0], device=env.device), is_relative=True)
                restored_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
                expected_hash = counter["capture"]["snapshot_state_hash"]
                if restored_hash != expected_hash:
                    raise RuntimeError(f"PROSPECTIVE_DISK_RESTORE_MISMATCH:{trial_id}:{expected_hash}:{restored_hash}")
                # A second restore from the same disk object is the strict
                # repeatability check used for all scientific branches.
                env.reset_to(move_device(disk_snapshot, env.device), torch.tensor([0], device=env.device), is_relative=True)
                restored_hash_2 = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
                if restored_hash_2 != expected_hash:
                    raise RuntimeError(f"PROSPECTIVE_SECOND_RESTORE_MISMATCH:{trial_id}")
                counter["capture"].update({"restored_state_hash": restored_hash, "second_restore_hash": restored_hash_2, "restore_exact": 1})
                if visual_writer is None:
                    visual_writer = csv.DictWriter(fh, fieldnames=list(counter["capture"]))
                    if fh.tell() == 0:
                        visual_writer.writeheader()
                if existing_visual_capture(Path(os.environ["PVP_COLLECTION_OUT"]), trial_id) is None:
                    visual_writer.writerow(counter["capture"])
                    fh.flush()
                return rows, rec

            p4.run_probe_episode = strict_run
            return p4

        p5.import_p4_probe = strict_import

        # Replace only the context plan; all P5-S0-C environment, controller,
        # outcome and corrected telemetry code remains authoritative.
        prospective = json.loads(Path(os.environ["PVP_CONTEXT_MANIFEST"]).read_text())
        p5.context_plan_for_task = lambda task_id: [
            {"root_id": r["root_id"], "task": int(r["task"]), "root_index": int(r["root_index"]),
             "root_seed": int(r["root_seed"]), "split": r["split"], "friction_band": r["friction_band"],
             "hidden_friction_analysis_only": float(r["mu_GT"]), "context_id": r["context_id"]}
            for r in prospective if int(r["task"]) == int(task_id)
        ]
        raise SystemExit(p5.worker_main())


def run_task(out: Path, target: Path, context_manifest: Path, task: int, timeout_s: int) -> dict:
    target_obj = json.loads(target.read_text())
    all_ids = [cid for cid in target_obj["contexts"] if cid.startswith(f"pv_train_t{task}_") or cid.startswith(f"pv_dev_t{task}_")]
    task_out = out / f"task{task}"
    completed_ids = set()
    completed_csv = task_out / f"task{task}" / "context.csv"
    if completed_csv.exists():
        completed_ids = {
            str(r["context_id"])
            for r in csv.DictReader(completed_csv.open(newline=""))
            if str(r.get("strict_matched", "0")) == "1"
        }
    ids = [cid for cid in all_ids if cid not in completed_ids]
    log = out / "logs" / f"task{task}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(TABERO), str(CLIENT_SRC), str(FORTE)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "TABERO_ROOT": str(TABERO),
        "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
        "P5S0C_OUT": str(task_out), "P5S0C_WORKER": "1", "P5S0C_TASK_ID": str(task),
        "P5S0C_TARGET_MANIFEST": str(target), "P5S0C_CONTEXT_IDS": ",".join(ids), "P5S0C_SKIP_REPLAY": "1",
        "PVP_COLLECTION_OUT": str(out), "PVP_CONTEXT_MANIFEST": str(context_manifest),
        "VISUAL_PI0_PORT": os.environ.get("VISUAL_PI0_PORT", "18881"),
    })
    if not ids:
        return {
            "task": task, "contexts": len(all_ids), "resumed_contexts": 0,
            "skipped_completed": len(completed_ids), "returncode": 0,
            "elapsed_wall_s": 0.0, "log": str(log), "task_out": str(task_out),
        }
    start = time.time()
    with log.open("w") as fh:
        proc = subprocess.Popen([str(ISAAC_PY), "-u", str(Path(__file__).resolve()), "--worker"], cwd=TABERO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate(); proc.wait(timeout=60); rc = 124
    return {
        "task": task, "contexts": len(all_ids), "resumed_contexts": len(ids),
        "skipped_completed": len(completed_ids), "returncode": rc,
        "elapsed_wall_s": time.time() - start, "log": str(log), "task_out": str(task_out),
    }


def orchestrate(split: str, out: Path, timeout_s: int, smoke: bool = False) -> None:
    root = out.parent
    target = root / f"PROSPECTIVE_{split}_TARGET_MANIFEST.json"
    context_manifest = root / "PROSPECTIVE_CONTEXT_MANIFEST.csv"
    # Convert the CSV to a JSON list once; the worker reads only this frozen
    # context manifest and never uses outcomes to select or alter contexts.
    rows = list(csv.DictReader(context_manifest.open()))
    prospective = []
    for r in rows:
        r["task"] = int(r["task"]); r["root_index"] = int(r["root_index"]); r["root_seed"] = int(r["root_seed"]); r["mu_GT"] = float(r["mu_GT"])
        if r["split"] == split:
            prospective.append(r)
    # Worker context plan needs all TRAIN/DEV rows because p5 filters by IDs.
    plan_path = out / "ALL_CONTEXTS_FOR_WORKER.json"
    write_json(plan_path, [dict(r) for r in rows])
    tasks = TASKS
    if smoke:
        full_target = json.loads(target.read_text())
        first_id = sorted(full_target["contexts"])[0]
        target = out / "SMOKE_TARGET_MANIFEST.json"
        write_json(target, {"manifest_name": "SMOKE_ONLY_NO_SCIENTIFIC_OUTCOMES", "split": split, "contexts": {first_id: full_target["contexts"][first_id][:1]}})
        tasks = [0]
    records = []
    for task in tasks:
        records.append(run_task(out, target, plan_path, task, timeout_s))
    write_json(out / "COLLECTION_RUN_MANIFEST.json", {"split": split, "workers": records, "status": "COMPLETE" if all(int(r["returncode"]) == 0 for r in records) else "ENGINEERING_FAILURE"})
    with (out / "COLLECTION_RUN_MANIFEST.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    if any(int(r["returncode"]) != 0 for r in records):
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--split", choices=["TRAIN", "DEV"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--timeout-s", type=int, default=21600)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.worker:
        worker()
    else:
        orchestrate(args.split, args.out, args.timeout_s, args.smoke)


if __name__ == "__main__":
    main()
