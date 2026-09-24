#!/usr/bin/env python3
"""Measurement-only target-object force calibration.

This script is intentionally outside the formal E3/old720 pipelines.  It
replays the existing successful Task-5 approach with the current formal
ForcePositionAction, snapshots the first verified bilateral object contact,
and restores that scene state before changing only the force slots to 2/3/4/5/6
N.  It records the raw object-filtered ContactSensor force matrix and the
formal controller debug fields.  No formal source, data, evaluator, or
benchmark result is modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/home/exouser/Tabero")
OUT = Path(os.environ.get("TARGET_FORCE_CALIBRATION_OUT", str(REPO / "E3_E6_E7_LANES" / "TARGET_OBJECT_FORCE_CALIBRATION_20260904_r6")))
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
TASK_SUITE = "libero_10"
TASK_ID = 5
OBJECT = "black_book_1"
TARGET = "desk_caddy_1"
SEED = 7400
REPLAY = Path("/media/volume/newdata/exouser/activeforcing_e3/TASK5_FORCE_PHYSICS_PILOT_20260902_062448/TRAIN_root7400_mu0.6_F7N/raw_policy/b5_t5_mu0.6_exp000_action_chunks.npz")
CALIBRATION_FORCES = [2.0, 3.0, 4.0, 5.0, 6.0]
REPLAY_FORCE = 7.0
CONTACT_EPS = 0.15
MAX_CONTINUATION = 500
REPLAY_STEPS_PER_CHUNK = 10
STATIC_HOLD_STEPS = 100


def npv(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def q_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(a) for a in q]
    qv = np.array([x, y, z], dtype=float)
    return v + 2.0 * (w * np.cross(qv, v) + np.cross(qv, np.cross(qv, v)))


def names_of(sensor: Any) -> list[str]:
    names = getattr(sensor, "body_names", None)
    if names is None:
        names = getattr(getattr(sensor, "data", None), "body_names", None)
    return [str(x) for x in names] if names is not None else []


def current_matrix(sensor: Any) -> np.ndarray:
    """Return current force_matrix_w as (body, filter, xyz), preserving all filters."""
    arr = npv(sensor.data.force_matrix_w).astype(float)
    if arr.ndim != 4 or arr.shape[0] < 1 or arr.shape[-1] != 3:
        raise RuntimeError(f"unexpected force_matrix_w shape {arr.shape}")
    return arr[0]


def side_indices(names: list[str], n: int) -> tuple[list[int], list[int]]:
    low = [x.lower() for x in names]
    li = [i for i, x in enumerate(low) if "leftfinger" in x or "left_finger" in x]
    ri = [i for i, x in enumerate(low) if "rightfinger" in x or "right_finger" in x]
    # The configured contact_grasp sensor is known to have the two Panda
    # finger bodies in order; retain this fallback explicitly in the manifest.
    if not li or not ri:
        if n >= 2:
            li, ri = [0], [1]
        else:
            raise RuntimeError(f"cannot map bilateral bodies: names={names}, n={n}")
    return li, ri


def object_force(env: Any) -> dict[str, Any]:
    sensor = env.scene[f"contact_grasp_{OBJECT}"]
    mat = current_matrix(sensor)
    names = names_of(sensor)
    li, ri = side_indices(names, mat.shape[0])
    left = mat[li].sum(axis=(0, 1))
    right = mat[ri].sum(axis=(0, 1))
    # Keep every raw per-body/per-filter contribution as JSON for later audit.
    net = npv(sensor.data.net_forces_w)
    return {
        "matrix": mat,
        "net": net,
        "body_names": names,
        "left_indices": li,
        "right_indices": ri,
        "left_world": left,
        "right_world": right,
    }


def all_net_force(env: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sensor = env.scene["contact_gripper"]
    arr = npv(sensor.data.net_forces_w)
    if arr.ndim == 4:
        arr = arr[0, -1]
    elif arr.ndim == 3:
        arr = arr[0]
    arr = np.asarray(arr, float).reshape(-1, 3)
    names = names_of(sensor)
    li, ri = side_indices(names, arr.shape[0])
    return arr[li].sum(axis=0), arr[ri].sum(axis=0), names


def frame_pose(env: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    f = env.scene[name]
    return npv(f.data.target_pos_w)[0, 0].astype(float), npv(f.data.target_quat_w)[0, 0].astype(float)


def joints(env: Any) -> tuple[list[str], list[float]]:
    r = env.scene["robot"]
    return [str(x) for x in getattr(r.data, "joint_names", [])], npv(r.data.joint_pos)[0].astype(float).tolist()


def controller_debug(env: Any) -> dict[str, Any]:
    try:
        dbg = env.action_manager.get_term("arm_action").debug_info or {}
        out = {}
        for k, v in dbg.items():
            a = npv(v)
            out[k] = a.reshape(-1).tolist() if a.size else []
        return out
    except Exception as exc:
        return {"debug_error": repr(exc)}


def snapshot_row(env: Any, obs: Any, step: int, requested: float, phase: str, event: str = "") -> dict[str, Any]:
    obj = env.scene[OBJECT]
    obj_p = npv(obj.data.root_pos_w)[0].astype(float)
    obj_q = npv(obj.data.root_quat_w)[0].astype(float)
    lp, lq = frame_pose(env, "left_gripper_frame")
    rp, rq = frame_pose(env, "right_gripper_frame")
    of = object_force(env)
    la, ra, all_names = all_net_force(env)
    ln = q_apply(lq, np.array([0.0, 0.0, 1.0]))
    rn = q_apply(rq, np.array([0.0, 0.0, 1.0]))
    lw, rw = of["left_world"], of["right_world"]
    contact_pos = getattr(env.scene[f"contact_grasp_{OBJECT}"].data, "contact_pos_w", None)
    contact_pos_json = ""
    if contact_pos is not None:
        cp = npv(contact_pos)
        # ContactSensor exposes an aggregated position per sensor body, not a
        # per-contact normal/impulse list.  Preserve the raw array verbatim.
        contact_pos_json = json.dumps(cp.tolist())
    left_local = q_apply(np.array([lq[0], -lq[1], -lq[2], -lq[3]]), lw)
    right_local = q_apply(np.array([rq[0], -rq[1], -rq[2], -rq[3]]), rw)
    dbg = controller_debug(env)
    jj, jp = joints(env)
    bilateral = int(np.linalg.norm(lw) >= CONTACT_EPS and np.linalg.norm(rw) >= CONTACT_EPS)
    return {
        "step": step, "requested_force_N": requested, "phase": phase, "event": event,
        "object": OBJECT, "target": TARGET,
        "object_pos_w": json.dumps(obj_p.tolist()), "object_quat_w": json.dumps(obj_q.tolist()),
        "eef_pos_w": json.dumps(npv(obs["policy"]["eef_pose"])[0, :3].astype(float).tolist()),
        "left_finger_pos_w": json.dumps(lp.tolist()), "right_finger_pos_w": json.dumps(rp.tolist()),
        "left_closing_normal_w": json.dumps(ln.tolist()), "right_closing_normal_w": json.dumps(rn.tolist()),
        "gripper_aperture_m": float(np.linalg.norm(lp - rp)),
        "left_all_net_world": json.dumps(la.tolist()), "right_all_net_world": json.dumps(ra.tolist()),
        "left_all_net_norm_N": float(np.linalg.norm(la)), "right_all_net_norm_N": float(np.linalg.norm(ra)),
        "left_object_force_world": json.dumps(lw.tolist()), "right_object_force_world": json.dumps(rw.tolist()),
        "left_object_force_local": json.dumps(left_local.tolist()), "right_object_force_local": json.dumps(right_local.tolist()),
        "left_object_force_norm_N": float(np.linalg.norm(lw)), "right_object_force_norm_N": float(np.linalg.norm(rw)),
        "left_object_normal_N": float(abs(np.dot(lw, ln))), "right_object_normal_N": float(abs(np.dot(rw, rn))),
        "object_fmin_N": float(min(np.linalg.norm(lw), np.linalg.norm(rw))),
        "object_fmean_N": float((np.linalg.norm(lw) + np.linalg.norm(rw)) / 2.0),
        "object_fsum_N": float(np.linalg.norm(lw) + np.linalg.norm(rw)),
        "object_bilateral_squeeze_N": float(2.0 * min(np.linalg.norm(lw), np.linalg.norm(rw))),
        "left_object_contact": int(np.linalg.norm(lw) >= CONTACT_EPS),
        "right_object_contact": int(np.linalg.norm(rw) >= CONTACT_EPS),
        "bilateral_object_contact": bilateral,
        "object_force_matrix_w": json.dumps(of["matrix"].tolist()),
        "object_contact_pos_w": contact_pos_json,
        "object_sensor_body_names": json.dumps(of["body_names"]),
        "all_sensor_body_names": json.dumps(all_names),
        "joint_names": json.dumps(jj), "joint_pos": json.dumps(jp),
        "controller_debug": json.dumps(dbg, separators=(",", ":")),
        "controller_f_sq_meas_N": float(dbg.get("f_sq_meas", [np.nan])[0]) if dbg.get("f_sq_meas") else np.nan,
        "controller_f_sq_meas_raw_N": float(dbg.get("f_sq_meas_raw", [np.nan])[0]) if dbg.get("f_sq_meas_raw") else np.nan,
        "controller_d_pred_m": float(dbg.get("d_pred", [np.nan])[0]) if dbg.get("d_pred") else np.nan,
        "controller_d_cmd_m": float(dbg.get("d_cmd", [np.nan])[0]) if dbg.get("d_cmd") else np.nan,
        "controller_d_actual_m": float(dbg.get("d_actual", [np.nan])[0]) if dbg.get("d_actual") else np.nan,
        "object_height_m": float(obj_p[2]),
    }


def set_force(action: np.ndarray, force: float) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).copy()
    if a.shape[-1] < 13:
        raise RuntimeError(f"expected 13D action, got {a.shape}")
    a[7:13] = 0.0
    a[9] = 0.5 * force
    a[12] = 0.5 * force
    return a


def state_hash(state: Any) -> str:
    def canonical(x: Any) -> Any:
        if hasattr(x, "detach"):
            return canonical(x.detach().cpu().numpy())
        if isinstance(x, np.ndarray):
            return {"dtype": str(x.dtype), "shape": list(x.shape), "data": x.tolist()}
        if isinstance(x, dict):
            return {str(k): canonical(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [canonical(v) for v in x]
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        return x
    payload = json.dumps(canonical(state), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def load_actions() -> list[np.ndarray]:
    d = np.load(REPLAY, allow_pickle=False)
    keys = sorted(k for k in d.files if k.startswith("chunk_"))
    # Match the already validated replay consumer exactly: one 10-step prefix
    # from each 50-step action chunk, rather than concatenating all chunk rows.
    return [np.asarray(d[k], dtype=np.float32)[i] for k in keys for i in range(min(REPLAY_STEPS_PER_CHUNK, len(d[k])))]


def run() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing calibration directory: {OUT}")
    if not REPLAY.exists():
        raise FileNotFoundError(REPLAY)
    OUT.mkdir(parents=True, exist_ok=False)
    os.environ.update({
        "HDF5_TRAJ_SOURCE_DIR": str(REPO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(REPO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(REPO / "benchmarks/datasets/libero/USD"),
        "TASK_SUITE": TASK_SUITE, "TASK_ID": str(TASK_ID),
        "ENABLE_CLOSED_LOOP_FORCE_CONTROLLER": "0",
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
    })
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        setup_task_objects(TASK_SUITE, TASK_ID)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 30.0
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        actions = load_actions()
        obs, _ = env.reset(seed=SEED)
        rows: list[dict[str, Any]] = []
        reset_row = snapshot_row(env, obs, 0, REPLAY_FORCE, "PRE_CONTACT", "direct_reset")
        rows.append(reset_row)
        prefix: list[np.ndarray] = []
        valid = None
        contact_action = None
        for i, raw in enumerate(actions):
            prefix.append(set_force(raw, REPLAY_FORCE))
            obs, _, term, trunc, _ = env.step(torch.from_numpy(prefix[-1]).reshape(1, -1).to(env.device))
            row = snapshot_row(env, obs, i + 1, REPLAY_FORCE, "REPLAY_TO_VALID_CONTACT")
            if i in {0, 9, 49, 84, 99}:
                of_debug = object_force(env)
                print(json.dumps({"step": i + 1, "matrix_shape": list(of_debug["matrix"].shape), "matrix_max_norm": float(np.linalg.norm(of_debug["matrix"], axis=-1).max()), "net_shape": list(np.asarray(of_debug["net"]).shape), "net_max": float(np.abs(np.asarray(of_debug["net"])).max()), "body_names": of_debug["body_names"]}), flush=True)
            if row["bilateral_object_contact"]:
                row["event"] = "first_verified_bilateral_contact"
                rows.append(row)
                valid = {
                    "state": env.scene.get_state(is_relative=True),
                    "state_hash": state_hash(env.scene.get_state(is_relative=True)),
                    "prefix_len": len(prefix),
                    "row": row,
                }
                contact_action = prefix[-1].copy()
                break
            rows.append(row)
            if bool(term[0].item()) or bool(trunc[0].item()):
                break
        if valid is None:
            raise RuntimeError("existing replay did not produce bilateral target-object contact")

        summary: list[dict[str, Any]] = []
        branch_rows: list[dict[str, Any]] = []
        for force in CALIBRATION_FORCES:
            env.reset(seed=SEED)
            # Recreate the same verified contact state, then restore the exact
            # state before each force branch.  Prefix and snapshot are fixed;
            # force changes only after the validated bilateral state.
            env.reset_to(valid["state"], torch.tensor([0], device=env.device), is_relative=True)
            post_hash = state_hash(env.scene.get_state(is_relative=True))
            obs = env.observation_manager.compute()
            start_z = float(npv(env.scene[OBJECT].data.root_pos_w)[0, 2])
            bilateral_seen = False
            last_phase = "STABLE_GRASP"
            phase_counts = {"STABLE_GRASP": 0, "LIFT": 0, "TRANSPORT": 0, "POST": 0}
            local_rows: list[dict[str, Any]] = []
            snapshot_eef = np.asarray(json.loads(valid["row"]["eef_pos_w"]), dtype=float)
            static_mode = os.environ.get("TARGET_FORCE_STATIC_HOLD", "0") == "1"
            if static_mode:
                continuation = [contact_action.copy() for _ in range(STATIC_HOLD_STEPS)]
            else:
                continuation = actions[valid["prefix_len"]:valid["prefix_len"] + MAX_CONTINUATION]
            for j, raw in enumerate(continuation, start=1):
                obs, _, term, trunc, _ = env.step(torch.from_numpy(set_force(raw, force)).reshape(1, -1).to(env.device))
                now_z = float(npv(env.scene[OBJECT].data.root_pos_w)[0, 2])
                eef = npv(obs["policy"]["eef_pose"])[0, :3]
                if not static_mode and j > 20 and now_z - start_z > 0.01 and phase_counts["LIFT"] == 0:
                    last_phase = "LIFT"
                # Transport is marked only after lift and when the commanded
                # EEF has materially moved horizontally from the snapshot.
                if not static_mode and last_phase == "LIFT" and now_z - start_z > 0.03 and np.linalg.norm(eef[:2] - snapshot_eef[:2]) > 0.03:
                    last_phase = "TRANSPORT"
                phase = last_phase
                r = snapshot_row(env, obs, j, force, phase)
                r.update({"state_hash_after_restore": post_hash, "source_action_index": valid["prefix_len"] + j - 1})
                local_rows.append(r)
                phase_counts[phase] += 1
                bilateral_seen = bilateral_seen or bool(r["bilateral_object_contact"])
                if bool(term[0].item()) or bool(trunc[0].item()):
                    break
            branch_rows.extend(local_rows)
            def vals(name: str, phase: str) -> np.ndarray:
                return np.asarray([float(r[name]) for r in local_rows if r["phase"] == phase and np.isfinite(float(r[name]))], float)
            stats = {"force_N": force, "n_steps": len(local_rows), "snapshot_state_hash": valid["state_hash"], "restore_state_hash": post_hash, "state_parity": int(post_hash == valid["state_hash"]), "bilateral_seen": int(bilateral_seen)}
            for phase in ("STABLE_GRASP", "LIFT", "TRANSPORT"):
                x = vals("object_bilateral_squeeze_N", phase)
                for suffix, fn in (("mean", np.mean), ("median", np.median), ("peak", np.max), ("p10", lambda a: np.percentile(a, 10)), ("p90", lambda a: np.percentile(a, 90))):
                    stats[f"{phase.lower()}_force_{suffix}_N"] = float(fn(x)) if len(x) else np.nan
                stats[f"{phase.lower()}_n"] = int(len(x))
                stats[f"{phase.lower()}_bilateral_fraction"] = float(np.mean([r["bilateral_object_contact"] for r in local_rows if r["phase"] == phase])) if any(r["phase"] == phase for r in local_rows) else np.nan
            summary.append(stats)

        all_rows = rows + branch_rows
        write_csv(OUT / "TARGET_OBJECT_FORCE_CALIBRATION_TIMESERIES.csv", all_rows)
        write_csv(OUT / "TARGET_OBJECT_FORCE_CALIBRATION_SUMMARY.csv", summary)
        write_json(OUT / "TARGET_OBJECT_FORCE_CALIBRATION_MANIFEST.json", {
            "status": "COMPLETE", "read_only": True, "formal_source_modified": False,
            "formal_data_modified": False, "formal_benchmark_rerun": False,
            "task_suite": TASK_SUITE, "task_id": TASK_ID, "object": OBJECT, "target": TARGET,
            "seed": SEED, "replay": str(REPLAY), "calibration_forces_N": CALIBRATION_FORCES,
            "replay_force_N": REPLAY_FORCE, "contact_threshold_N": CONTACT_EPS,
            "sensor": f"contact_grasp_{OBJECT}",
            "sensor_semantics": "force_matrix_w summed over all filtered target-object contact contributions per Panda finger body",
            "sensor_body_names": valid["row"]["object_sensor_body_names"],
            "first_bilateral_step": valid["prefix_len"], "snapshot_state_hash": valid["state_hash"],
            "snapshot_is_existing_success_replay_contact": True,
            "phase_rule": "STABLE_GRASP until object height rises >0.01m; LIFT after that until >0.03m; TRANSPORT after >0.03m; no release/settle included in 500-step continuation",
            "static_hold_mode": os.environ.get("TARGET_FORCE_STATIC_HOLD", "0") == "1",
            "raw_contact_normals": "not exposed by this Isaac ContactSensor data object; finger closing normals are explicit frame +Z and both world/local projections are retained",
        })
        print(json.dumps({"out": str(OUT), "summary": summary, "first_bilateral_step": valid["prefix_len"]}, indent=2, default=str), flush=True)
        # Isaac Sim 5.1 may abort while destructing a partially initialized
        # camera/Replicator object.  All artifacts are already closed/flushed;
        # terminate this isolated diagnostic process without touching other
        # Kit processes or any formal result.
        os._exit(0)
    finally:
        pass


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(run())
