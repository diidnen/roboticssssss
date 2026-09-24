#!/usr/bin/env python3
"""Capture real post-P4-B states for the frozen E7 DEV contexts.

This is a capture-only wrapper around the authoritative P5/P4-B runtime.  It
does not alter the query: the snapshot is taken only after run_probe_episode
returns, then restored twice to verify state-hash parity.  P5 writes the
authoritative physical history and query-quality record as usual.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path


TABERO = Path("/home/exouser/Tabero")
P5_RUNNER = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone_cpu(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def move_device(value, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_device(item, device) for item in value)
    return value


def _load_query_start_reference(path: Path) -> dict:
    """Read only the historical last-hold telemetry used for reconstruction QA."""
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    holds = [row for row in rows if row.get("probe_phase") == "hold"]
    if not holds:
        raise RuntimeError(f"no historical hold row in {path}")
    return holds[-1]


def _query_start_observation(env, p4) -> dict:
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    gp = obs["policy"]["gripper_pos"][0].detach().cpu().numpy()
    f = obs["policy"]["gripper_net_force"][0]
    if f.ndim == 3:
        f = f[-1]
    left = f[0].detach().cpu().numpy()
    right = f[1].detach().cpu().numpy()
    obj = env.scene[p4.OBJ_NAME].data.root_pos_w[0].detach().cpu().numpy()
    left_n = float(abs(left[2]))
    right_n = float(abs(right[2]))
    return {
        "eef": [float(x) for x in eef[:3]],
        "object": [float(x) for x in obj],
        "aperture": float(gp.reshape(-1)[0]),
        "left_contact": int(float((left * left).sum()) ** 0.5 > 0.15 and left_n > 0.15),
        "right_contact": int(float((right * right).sum()) ** 0.5 > 0.15 and right_n > 0.15),
        "left_normal_force_N": left_n,
        "right_normal_force_N": right_n,
    }


def _query_start_parity(current: dict, reference: dict) -> dict:
    def vec(name):
        return [float(reference[k]) for k in name]

    ref_eef = vec(["eef_x", "eef_y", "eef_z"])
    ref_obj = vec(["object_x_priv", "object_y_priv", "object_z_priv"])
    eef_err = max(abs(a - b) for a, b in zip(current["eef"], ref_eef))
    obj_err = max(abs(a - b) for a, b in zip(current["object"], ref_obj))
    aperture_err = abs(current["aperture"] - float(reference["gripper_opening"]))
    force_err = max(
        abs(current["left_normal_force_N"] - float(reference["measured_fn"]) / 2.0),
        abs(current["right_normal_force_N"] - float(reference["measured_fn"]) / 2.0),
    )
    return {
        "eef_max_abs_error_m": eef_err,
        "object_max_abs_error_m": obj_err,
        "aperture_abs_error_m": aperture_err,
        "normal_force_per_finger_error_N": force_err,
        "bilateral_contact_match": int(
            current["left_contact"] == 1 and current["right_contact"] == 1
        ),
        "pass": int(
            eef_err <= 5e-4
            and obj_err <= 5e-4
            and aperture_err <= 5e-4
            and force_err <= 0.5
            and current["left_contact"] == 1
            and current["right_contact"] == 1
        ),
    }


def main() -> int:
    import torch

    out = Path(os.environ["P5S0C_OUT"])
    snapshot_dir = out / "POSTQUERY_STATE_SNAPSHOTS"
    record_dir = out / "POSTQUERY_STATE_RECORDS"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)

    p5 = load_module("e7_postquery_authoritative_p5", P5_RUNNER)
    original_import = p5.import_p4_probe
    reconstruction_snapshot = os.environ.get("P5S0C_RECONSTRUCTION_SNAPSHOT", "")
    reconstruction_reference = os.environ.get("P5S0C_RECONSTRUCTION_REFERENCE", "")
    reconstruction_at_step = int(os.environ.get("P5S0C_RECONSTRUCTION_AT_STEP", "150"))
    reconstruction_record = {}

    def capture_import(task_id: int):
        p4 = original_import(task_id)
        original_run = p4.run_probe_episode

        if reconstruction_snapshot:
            if task_id != int(os.environ.get("P5S0C_TASK_ID", "-1")):
                raise RuntimeError("reconstruction snapshot task mismatch")
            if not reconstruction_reference:
                raise RuntimeError("P5S0C_RECONSTRUCTION_REFERENCE is required")
            reference = _load_query_start_reference(Path(reconstruction_reference))

            def reconstruct_run(env, *, seed_idx: int, mu: float, trial_id: str, dt: float):
                snapshot_cpu = torch.load(
                    reconstruction_snapshot, map_location="cpu", weights_only=False
                )
                original_step = env.step
                count = {"n": 0}
                # The historical scene snapshot contains the physical gripper
                # preload, but P4-B's local d_pred variable is not part of the
                # IsaacLab scene state.  Replaying the next hold action with
                # the failed run's stale d_pred immediately destroys that
                # preload.  During reconstruction only, keep the original
                # controller's force channels and law untouched while restoring
                # the historical aperture target that corresponds to the
                # captured query-start state.
                historical_aperture = float(reference["gripper_opening"])
                restore_window = {"active": False}

                def step_with_reconstruction(action):
                    if restore_window["active"]:
                        action = action.clone()
                        action[0, 6] = historical_aperture
                    result = original_step(action)
                    count["n"] += 1
                    if count["n"] == reconstruction_at_step:
                        env.reset_to(
                            move_device(snapshot_cpu, env.device),
                            torch.tensor([0], device=env.device),
                            is_relative=True,
                        )
                        restore_window["active"] = True
                    if count["n"] == reconstruction_at_step + 40:
                        restore_window["active"] = False
                        reconstruction_record["query_start_observation"] = _query_start_observation(env, p4)
                        reconstruction_record["query_start_parity"] = _query_start_parity(
                            reconstruction_record["query_start_observation"], reference
                        )
                        reconstruction_record["query_start_reference"] = reconstruction_reference
                        reconstruction_record["reconstruction_snapshot"] = reconstruction_snapshot
                        reconstruction_record["reconstruction_restore_step"] = count["n"] - 40
                        reconstruction_record["reconstruction_state_before_probe"] = count["n"]
                        if not reconstruction_record["query_start_parity"]["pass"]:
                            raise RuntimeError("QUERY_START_PARITY_FAIL")
                    return result

                env.step = step_with_reconstruction
                try:
                    return original_run(
                        env, seed_idx=seed_idx, mu=mu, trial_id=trial_id, dt=dt
                    )
                finally:
                    env.step = original_step

        def capture_after_query(env, *, seed_idx: int, mu: float, trial_id: str, dt: float):
            run_fn = reconstruct_run if reconstruction_snapshot else original_run
            rows, record = run_fn(
                env, seed_idx=seed_idx, mu=mu, trial_id=trial_id, dt=dt
            )
            snapshot_gpu = env.scene.get_state(is_relative=True)
            snapshot_cpu = clone_cpu(snapshot_gpu)
            state_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            snapshot_path = snapshot_dir / f"{trial_id}.pt"
            temporary_path = snapshot_dir / f".{trial_id}.pt.tmp"
            torch.save(snapshot_cpu, temporary_path)
            temporary_path.replace(snapshot_path)

            env_ids = torch.tensor([0], device=env.device)
            env.reset_to(move_device(snapshot_cpu, env.device), env_ids, is_relative=True)
            first_restore_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
            env.reset_to(move_device(snapshot_cpu, env.device), env_ids, is_relative=True)
            second_restore_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))

            phases = sorted({str(row.probe_phase) for row in rows})
            capture_record = {
                "context_id": trial_id,
                "task": int(task_id),
                "root_seed": int(seed_idx),
                "friction": float(mu),
                "capture_semantics": "AFTER_AUTHORITATIVE_RUN_PROBE_EPISODE_RETURN",
                "capture_after_probe": True,
                "probe_rows": len(rows),
                "probe_phases": phases,
                "probe_stop_trigger": str(record.get("stop_trigger", "")),
                "probe_failure": int(record.get("probe_failure", 0)),
                "contact_retained": int(record.get("contact_retained", 0)),
                "return_completed": int(record.get("return_completed", 0)),
                "snapshot_path": str(snapshot_path),
                "state_hash": state_hash,
                "first_restore_hash": first_restore_hash,
                "second_restore_hash": second_restore_hash,
                "restore_parity_pass": int(
                    state_hash == first_restore_hash == second_restore_hash
                ),
                "p5_runner": str(P5_RUNNER),
                "p4_probe": str(p5.P4_COLLECT),
                "physical_history_path": str(
                    out / "P5S0C_PROBE_TELEMETRY" / f"{trial_id}_probe_timesteps.csv"
                ),
            }
            if reconstruction_record:
                capture_record["query_start_reconstruction"] = reconstruction_record
            record_path = record_dir / f"{trial_id}.json"
            record_path.write_text(
                json.dumps(capture_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return rows, record

        p4.run_probe_episode = capture_after_query
        return p4

    p5.import_p4_probe = capture_import
    return p5.worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
