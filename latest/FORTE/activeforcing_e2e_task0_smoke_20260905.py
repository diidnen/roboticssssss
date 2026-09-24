#!/usr/bin/env python3
"""Task-0 ActiveForcing end-to-end smoke.

This lane is deliberately local to the smoke output.  It reuses the current
P5-S0-C runner, P4-B probe, and native ForcePositionAction without editing
either source.  The probe worker saves a restorable post-probe scene state;
each lift/hold branch is a fresh Isaac process restoring that state.
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
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
OUT = ROOT / "analysis/results/activeforcing_e2e_task0_smoke_20260905"
VALIDATION = ROOT / "analysis/results/current_runtime_setpoint_mapping_validation_20260905"
P5_PATH = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"
P4_PATH = TABERO / "analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py"
TPI_PATH = TABERO / "analysis/trajectory_physical_imagination.py"
EVIDENCE_PATH = TABERO / "analysis/activeforcing_historical_transfer_20260904/posterior/evidence_46d.py"
EVIDENCE_NORM = TABERO / "analysis/activeforcing_historical_transfer_20260904/posterior/NORMALIZATION_46D.json"
BELIEF_DIR = ROOT / "activeforcing_full_claim_closure_20260902_062809/E6_E7/E6_E7_LOCKED_HANDOFF"
FEAS_DIR = ROOT / "analysis/results/final_probe_continuous_posterior_rebuild_20260904"
ISAAC_PY = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
HOST_PY = Path("/usr/bin/python3")
WARP_CORE = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64")
OPENPI = TABERO / "benchmarks/openpi/openpi-client/src"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
TASK = 0
FMAX = 5.0
DT = 0.05
HOLD_STEPS = 20
LIFT_STEPS = 50
POST_LIFT_HOLD_STEPS = 30
CONTEXT_MANIFEST = VALIDATION / "VALIDATION_CONTEXT_MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clone_cpu(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(v) for v in value)
    return value


def move_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_device(v, device) for v in value)
    return value


def contexts() -> list[dict[str, Any]]:
    data = json.loads(CONTEXT_MANIFEST.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for job in data["jobs"]:
        if int(job["task"]) != TASK:
            continue
        cid = str(job["context_id"])
        spec = out.setdefault(cid, {"context_id": cid, "task": TASK})
        # Existing manifest embeds the root/friction in the context id; obtain
        # the authoritative values from the P5 context plan in the runner.
        spec["repeat_job_dirs"] = spec.get("repeat_job_dirs", []) + [job["job_dir"]]
    if len(out) != 3:
        raise RuntimeError(f"expected exactly 3 frozen task0 contexts, got {len(out)}")
    p5 = load_module("task0_context_plan", P5_PATH)
    plans = {str(x["context_id"]): dict(x) for x in p5.context_plan_for_task(TASK)}
    for cid, spec in out.items():
        if cid not in plans:
            raise RuntimeError(f"frozen context absent from P5 plan: {cid}")
        spec.update(plans[cid])
    return [out[k] for k in sorted(out)]


def launch(env_extra: dict[str, str], log: Path, mode: str, **args: str) -> int:
    env = os.environ.copy()
    env.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join([str(WARP_CORE), str(TABERO), str(OPENPI)]),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
        "TABERO_ROOT": str(TABERO), "P5S0C_OUT": str(OUT),
        "HDF5_TRAJ_SOURCE_DIR": str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        "LIBERO_CONFIG_DIR": str(TABERO / "benchmarks/datasets/libero/config"),
        "LIBERO_ASSETS_DATA_DIR": str(TABERO / "benchmarks/datasets/libero/USD"),
        **env_extra,
    })
    cmd = [str(ISAAC_PY), "-u", str(Path(__file__).resolve()), "--worker", "--mode", mode]
    for k, v in args.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        return int(subprocess.run(cmd, cwd=TABERO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=21600).returncode)


def run_probe_worker(cid: str) -> int:
    import torch
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        p5 = load_module("activeforcing_smoke_p5_probe", P5_PATH)
        p5.OUT = OUT
        plan = next(x for x in contexts() if x["context_id"] == cid)
        p4 = p5.import_p4_probe(TASK)
        setup_task_objects(p5.TASK_SUITE, TASK)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        obs, _ = env.reset(seed=int(plan["root_seed"]))
        rows, rec = p4.run_probe_episode(env, seed_idx=int(plan["root_seed"]), mu=float(plan["hidden_friction_analysis_only"]), trial_id=cid, dt=DT)
        q = p5.derive_p4b_probe_quality(p4, rows, rec)
        probe_path = OUT / "PROBE_TELEMETRY" / f"{cid}.csv"
        write_csv(probe_path, [vars(x) if hasattr(x, "__dict__") else dict(x) for x in rows])
        state = clone_cpu(env.scene.get_state(is_relative=True))
        hq = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
        state_path = OUT / "PROBE_TELEMETRY" / f"{cid}_post_probe_state.pt"
        torch.save({"state": state, "state_hash": hq, "context_id": cid}, state_path)
        write_json(OUT / "PROBE_TELEMETRY" / f"{cid}_probe_result.json", {
            **plan, "probe_record": rec, "probe_quality": q, "probe_qualified": int(q["probe_qualified"]),
            "probe_rows": len(rows), "probe_path": str(probe_path), "post_probe_state_path": str(state_path),
            "post_probe_state_hash": hq, "single_physical_probe": True,
        })
        return 0 if q["probe_qualified"] else 2
    except Exception as exc:
        write_json(OUT / "PROBE_TELEMETRY" / f"{cid}_ERROR.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        return 1
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        try: app.close()
        except Exception: pass


def execute_branch_in_env(env: Any, p4: Any, p5: Any, plan: dict[str, Any], method: str, force: float, repeat: int, state: Any, state_hash: str) -> dict[str, Any]:
    import torch
    cid = str(plan["context_id"])
    env.reset_to(state, torch.tensor([0], device=env.device), is_relative=True)
    restored_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    eef_aa = p4._aa(eef[3:7]); cmd_pos = eef[:3].copy()
    obj_name = p5.TASK_OBJECTS[TASK]
    obj0 = env.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy().copy()
    lift = cmd_pos.copy(); lift[2] += 0.142
    stages = [("branch_hold", HOLD_STEPS, cmd_pos.copy()), ("lift", LIFT_STEPS, lift.copy()), ("lift_hold", POST_LIFT_HOLD_STEPS, lift.copy())]
    try:
        gids, _ = env.scene["robot"].find_joints(env.cfg.gripper_joint_names)
        d_pred = float(env.scene["robot"].data.joint_pos[0, gids].mean().detach().cpu().item())
        d_pred = float(np.clip(d_pred, p4.D_CLOSED, p4.D_OPEN))
    except Exception:
        d_pred = float(p4.D_CLOSED)
    rows: list[dict[str, Any]] = []; force_series: list[float] = []; lift_success = 0; step = 0; prev = cmd_pos.copy()
    branch_label = f"{method}_F{force:.4f}_R{repeat}"
    for phase, n, target in stages:
        start = prev.copy()
        for i in range(n):
            cmd_pos = p4._interp(start, target, i, n)
            action = p4._make_action(cmd_pos, eef_aa, d_pred, float(force), env.device)
            obs, _, term, trunc, _ = env.step(action); step += 1
            dbg = p4._dbg(env); f_native = float(p4._f(dbg.get("f_sq_meas"), 0.0)); f_target = float(p4._f(dbg.get("f_sq_pred_eff"), 0.0))
            obj = p5.target_object_force_snapshot(env, p4, obj_name); objp = env.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy(); dz = float(objp[2] - obj0[2])
            bilateral = int(obj["target_object_bilateral_contact"]); lift_success = int(lift_success or dz >= 0.01); force_series.append(float(obj["F_obj_bilateral_n"]))
            rows.append({"context_id": cid, "task": TASK, "method": method, "requested_force_N": float(force), "repeat": repeat, "step": step, "phase": phase, "F_target_eff_n": f_target, "native_measured_force_N": f_native, "object_filtered_bilateral_force_N": float(obj["F_obj_bilateral_n"]), "object_filtered_left_normal_N": float(obj["F_obj_left_normal_n"]), "object_filtered_right_normal_N": float(obj["F_obj_right_normal_n"]), "target_object_bilateral_contact": bilateral, "object_z_m": float(objp[2]), "object_dz_m": dz, "gripper_aperture_pred": float(d_pred), "restored_state_hash": restored_hash})
            d_pred = float(p4._force_servo(d_pred, f_native, float(force)))
            if bool(term[0].item()) or bool(trunc[0].item()): break
        prev = target.copy()
    lift_index = next((i for i, r in enumerate(rows) if float(r["object_dz_m"]) >= 0.01), None)
    # The contract is lift completion followed by a dedicated 30-step hold;
    # do not start the hold window at the first transient 1-cm rise.
    hold_rows = [r for r in rows if r["phase"] == "lift_hold"]
    hold_success = int(lift_index is not None and len(hold_rows) == POST_LIFT_HOLD_STEPS and all(int(r["target_object_bilateral_contact"]) and float(r["object_dz_m"]) >= 0.005 for r in hold_rows))
    arr = np.asarray(force_series, dtype=float)
    out = {**plan, "method": method, "requested_force_N": float(force), "repeat": repeat, "branch_label": branch_label, "restored_state_hash": restored_hash, "reference_state_hash": state_hash, "branch_state_parity": int(restored_hash == state_hash), "lift_success": lift_success, "hold_30_step_success": hold_success, "lift_index": lift_index, "hold_rows": len(hold_rows), "measured_mean_force_N": float(arr.mean()) if arr.size else 0.0, "measured_top5_force_N": float(np.quantile(arr, 0.95)) if arr.size else 0.0, "force_exposure_Ns": float(arr.sum() * DT), "peak_measured_force_N": float(arr.max()) if arr.size else 0.0, "telemetry_rows": len(rows), "valid_branch": 1}
    out_dir = OUT / ("ACTIVEFORCING_TRACES" if method == "ACTIVEFORCING" else "MAXFORCE_TRACES")
    write_csv(out_dir / f"{cid}_{method}_F{force:.4f}_R{repeat}_PAIRED.csv", rows)
    write_json(out_dir / f"{cid}_{method}_F{force:.4f}_R{repeat}_PAIRED.json", out)
    return out


def run_paired_worker(cid: str, repeat: int) -> int:
    import torch
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        p5 = load_module("activeforcing_smoke_paired", P5_PATH); p5.OUT = OUT
        plan = next(x for x in contexts() if x["context_id"] == cid)
        p4 = p5.import_p4_probe(TASK); setup_task_objects(p5.TASK_SUITE, TASK)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1); cfg.episode_length_s = 45.0
        env = gym.make(ENV_ID, cfg=cfg).unwrapped; env.reset(seed=int(plan["root_seed"]))
        probe_rows, probe_rec = p4.run_probe_episode(env, seed_idx=int(plan["root_seed"]), mu=float(plan["hidden_friction_analysis_only"]), trial_id=f"{cid}_R{repeat}", dt=DT)
        quality = p5.derive_p4b_probe_quality(p4, probe_rows, probe_rec)
        probe_path = OUT / "PROBE_TELEMETRY" / f"{cid}_repeat{repeat}.csv"; write_csv(probe_path, [vars(x) if hasattr(x, "__dict__") else dict(x) for x in probe_rows])
        if not quality["probe_qualified"]: raise RuntimeError(f"invalid physical probe: {quality}")
        runtime_state = env.scene.get_state(is_relative=True); state = clone_cpu(runtime_state); state_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env)); state_path = OUT / "PROBE_TELEMETRY" / f"{cid}_repeat{repeat}_post_probe_state.pt"; torch.save({"state": state, "state_hash": state_hash, "context_id": cid, "repeat": repeat}, state_path)
        # Checkpoints were serialized with the host NumPy ABI.  Keep Isaac
        # responsible for physical probe/execution and run the frozen offline
        # selector in the compatible host Python process.
        selector_env = os.environ.copy(); selector_env.pop("PYTHONPATH", None); selector_env.pop("PYTHONNOUSERSITE", None); selector_env["AF_SMOKE_OUT"] = str(OUT)
        selector_cmd = [str(HOST_PY), str(Path(__file__).resolve()), "--worker", "--mode", "selector", "--context", cid, "--probe-path", str(probe_path), "--tag", f"_repeat{repeat}"]
        selector_rc = subprocess.run(selector_cmd, cwd=TABERO, env=selector_env, timeout=600).returncode
        if selector_rc != 0: raise RuntimeError(f"offline selector failed rc={selector_rc}")
        bundle = json.loads((OUT / "FRICTION_POSTERIOR" / f"{cid}_repeat{repeat}.json").read_text())
        posterior = bundle["posterior"]; selected = float(bundle["selected"]["force_N"])
        write_json(OUT / "FRICTION_POSTERIOR" / f"{cid}_repeat{repeat}.json", {**bundle, "probe_record": probe_rec, "probe_quality": quality, "state_path": str(state_path), "state_hash": state_hash})
        execute_branch_in_env(env, p4, p5, plan, "ACTIVEFORCING", selected, repeat, runtime_state, state_hash)
        execute_branch_in_env(env, p4, p5, plan, "MAXFORCE", FMAX, repeat, runtime_state, state_hash)
        return 0
    except Exception as exc:
        write_json(OUT / "PROBE_TELEMETRY" / f"{cid}_repeat{repeat}_ERROR.json", {"error": repr(exc), "traceback": traceback.format_exc()}); return 1
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        try: app.close()
        except Exception: pass


def run_branch_worker(cid: str, method: str, force: float, repeat: int) -> int:
    import torch
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        p5 = load_module("activeforcing_smoke_p5_branch", P5_PATH)
        p5.OUT = OUT
        plan = next(x for x in contexts() if x["context_id"] == cid)
        p4 = p5.import_p4_probe(TASK)
        setup_task_objects(p5.TASK_SUITE, TASK)
        cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        env = gym.make(ENV_ID, cfg=cfg).unwrapped
        data = torch.load(OUT / "PROBE_TELEMETRY" / f"{cid}_post_probe_state.pt", map_location="cpu", weights_only=False)
        state = move_device(data["state"], env.device)
        env.reset_to(state, torch.tensor([0], device=env.device), is_relative=True)
        restored_hash = p5.stable_hash_obj(p5.restorable_snapshot_for_hash(env))
        obs = env.observation_manager.compute()
        eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
        eef_aa = p4._aa(eef[3:7])
        cmd_pos = eef[:3].copy()
        obj_name = p5.TASK_OBJECTS[TASK]
        obj0 = env.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy().copy()
        lift = cmd_pos.copy(); lift[2] += 0.142
        stages = [("branch_hold", HOLD_STEPS, cmd_pos.copy()), ("lift", LIFT_STEPS, lift.copy()), ("lift_hold", POST_LIFT_HOLD_STEPS, lift.copy())]
        try:
            gids, _ = env.scene["robot"].find_joints(env.cfg.gripper_joint_names)
            d_pred = float(env.scene["robot"].data.joint_pos[0, gids].mean().detach().cpu().item())
            d_pred = float(np.clip(d_pred, p4.D_CLOSED, p4.D_OPEN))
        except Exception:
            d_pred = float(p4.D_CLOSED)
        d_takeover = d_pred
        rows: list[dict[str, Any]] = []
        force_series: list[float] = []
        lift_success = 0
        step = 0
        prev = cmd_pos.copy()
        branch_label = f"{method}_F{force:.4f}_R{repeat}"
        for phase, n, target in stages:
            start = prev.copy()
            for i in range(n):
                cmd_pos = p4._interp(start, target, i, n)
                action = p4._make_action(cmd_pos, eef_aa, d_pred, float(force), env.device)
                obs, _, term, trunc, _ = env.step(action); step += 1
                dbg = p4._dbg(env)
                f_native = float(p4._f(dbg.get("f_sq_meas"), 0.0))
                f_target = float(p4._f(dbg.get("f_sq_pred_eff"), 0.0))
                obj = p5.target_object_force_snapshot(env, p4, obj_name)
                objp = env.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy()
                dz = float(objp[2] - obj0[2])
                bilateral = int(obj["target_object_bilateral_contact"])
                lift_success = int(lift_success or dz >= 0.01)
                if phase in ("branch_hold", "lift", "lift_hold"):
                    force_series.append(float(obj["F_obj_bilateral_n"]))
                rows.append({
                    "context_id": cid, "task": TASK, "method": method, "requested_force_N": float(force), "repeat": repeat,
                    "step": step, "phase": phase, "F_target_eff_n": f_target, "native_measured_force_N": f_native,
                    "object_filtered_bilateral_force_N": float(obj["F_obj_bilateral_n"]),
                    "object_filtered_left_normal_N": float(obj["F_obj_left_normal_n"]), "object_filtered_right_normal_N": float(obj["F_obj_right_normal_n"]),
                    "target_object_bilateral_contact": bilateral, "object_z_m": float(objp[2]), "object_dz_m": dz,
                    "gripper_aperture_pred": float(d_pred), "restored_state_hash": restored_hash,
                })
                if phase in ("branch_hold", "lift", "lift_hold"):
                    d_pred = float(p4._force_servo(d_pred, f_native, float(force)))
                if bool(term[0].item()) or bool(trunc[0].item()):
                    break
            prev = target.copy()
        lift_index = next((i for i, r in enumerate(rows) if float(r["object_dz_m"]) >= 0.01), None)
        hold_rows = rows[lift_index + 1: lift_index + 31] if lift_index is not None else []
        hold_success = int(lift_index is not None and len(hold_rows) >= 30 and all(int(r["target_object_bilateral_contact"]) and float(r["object_dz_m"]) >= 0.005 for r in hold_rows))
        arr = np.asarray(force_series, dtype=float)
        mean = float(arr.mean()) if arr.size else 0.0
        top5 = float(np.quantile(arr, 0.95)) if arr.size else 0.0
        exposure = float(arr.sum() * DT)
        out = {
            **plan, "method": method, "requested_force_N": float(force), "repeat": repeat, "branch_label": branch_label,
            "restored_state_hash": restored_hash, "reference_state_hash": data["state_hash"], "branch_state_parity": int(restored_hash == data["state_hash"]),
            "lift_success": lift_success, "hold_30_step_success": hold_success, "lift_index": lift_index, "hold_rows": len(hold_rows),
            "measured_mean_force_N": mean, "measured_top5_force_N": top5, "force_exposure_Ns": exposure,
            "peak_measured_force_N": float(arr.max()) if arr.size else 0.0, "telemetry_rows": len(rows),
        }
        out_dir = OUT / ("ACTIVEFORCING_TRACES" if method == "ACTIVEFORCING" else "MAXFORCE_TRACES")
        write_csv(out_dir / f"{cid}_{method}_F{force:.4f}_R{repeat}.csv", rows)
        write_json(out_dir / f"{cid}_{method}_F{force:.4f}_R{repeat}.json", out)
        return 0
    except Exception as exc:
        write_json(OUT / "ACTIVEFORCING_TRACES" / f"{cid}_{method}_F{force:.4f}_R{repeat}_ERROR.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        return 1
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        try: app.close()
        except Exception: pass


def posterior_from_probe(probe_path: Path) -> dict[str, Any]:
    import torch
    evidence = load_module("activeforcing_evidence46d", EVIDENCE_PATH)
    feat = evidence.Evidence46D(EVIDENCE_NORM)
    arr = feat.normalize_dynamic(feat.csv(probe_path))
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.projection = torch.nn.Sequential(torch.nn.Linear(46, 16), torch.nn.ReLU()); self.gru = torch.nn.GRU(16, 16, batch_first=True); self.mu_head = torch.nn.Linear(16, 1); self.log_sigma_head = torch.nn.Linear(16, 1)
        def forward(self, x):
            z = self.projection(x); _, h = self.gru(z); h = h[-1]; return self.mu_head(h).squeeze(1), self.log_sigma_head(h).squeeze(1).clamp(-5.0, 1.5)
    mus, logs = [], []
    for i in range(3):
        p = BELIEF_DIR / f"PHYSICAL_BELIEF_member_{i}.pt"; d = torch.load(p, map_location="cpu", weights_only=False)
        m = Net(); m.load_state_dict(d["state_dict"]); m.eval()
        with torch.no_grad(): mu, ls = m(torch.tensor(arr[None], dtype=torch.float32))
        mus.append(float(mu.item())); logs.append(float(ls.item()))
    q = np.quantile(np.asarray(mus), [0.05, 0.5, 0.95]).tolist()
    result = {"member_means": mus, "member_log_sigma": logs, "posterior_samples_mu": mus, "posterior_sample_count": len(mus), "posterior_representation": "three frozen physical-belief ensemble members; no GT or fixed-mu substitution", "mean": float(np.mean(mus)), "std": float(np.std(mus, ddof=1)), "q05": q[0], "q50": q[1], "q95": q[2], "feature_rows": int(len(arr)), "feature_dim": int(arr.shape[1])}
    from activeforcing_belief_contract import annotate
    return annotate(probe_path, result)


def utility(p: float, f: float) -> float:
    return float(p * (FMAX - f) / FMAX + (1.0 - p) * (-1.0))


def infer_curves(cid: str, posterior: dict[str, Any], probe_path: Path | None = None, tag: str = "", *, checkpoint_paths: list[Path] | None = None, output_dir: Path | None = None, diagnostic_only: bool = False) -> dict[str, Any]:
    from activeforcing_belief_contract import require_input_support
    if diagnostic_only:
        if output_dir is None or output_dir.resolve() == OUT.resolve():
            raise ValueError("Diagnostic inference requires a separate output directory")
    else:
        require_input_support(posterior)
    import torch
    feature_builder = load_module("activeforcing_canonical_features", ROOT / "activeforcing_feasibility_features.py")
    # Frozen arm prefix: use the already validated task0 branch telemetry only
    # for commands/phases; replace its physical state with this trial's probe
    # observation, so no branch outcome is admitted to the selector.
    existing = sorted((VALIDATION / "TASK0_TRACES").glob(f"{cid}_*.csv"))[0]
    canonical, feature_provenance = feature_builder.from_saved_probe(
        cid, probe_path or (OUT / "PROBE_TELEMETRY" / f"{cid}.csv"),
        existing, TASK, 3.0, float(posterior["member_means"][0]))
    # Same architecture and normalization as the fixed clean feasibility
    # rebuild; each model seed is averaged before posterior marginalization.
    grid = np.round(np.arange(3.0, 5.0001, 0.01), 2)
    curves = []
    models = []
    class FeasibilityOnly(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.command_gru = torch.nn.GRU(17, 64, batch_first=True); self.condition = torch.nn.Sequential(torch.nn.Linear(54, 64), torch.nn.ReLU()); self.head = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        def forward(self, step, cond):
            _, h = self.command_gru(step); z = torch.cat([h[-1], self.condition(cond)], dim=-1); return self.head(z).squeeze(-1)
    paths = checkpoint_paths or [FEAS_DIR / f"POSTERIOR_FEASIBILITY_seed{i}.pt" for i in range(3)]
    for ckpath in paths:
        saved = torch.load(ckpath, map_location="cpu", weights_only=False)
        m = FeasibilityOnly(); m.load_state_dict(saved["state_dict"]); m.eval()
        models.append((ckpath, m, np.asarray(saved["normalization_mean"], np.float32), np.asarray(saved["normalization_std"], np.float32)))
    for f in grid:
        per_mu = []
        per_mu_seed = []
        for mu in posterior["member_means"]:
            x = canonical.copy(); x[:, 17] = float(f) / 8.0; x[:, 18] = float(mu)
            seed_probs = []
            for _, m, xm, xs in models:
                xn = (x - xm) / np.maximum(xs, 1e-6)
                with torch.no_grad(): seed_probs.append(float(torch.sigmoid(m(torch.tensor(xn[None, :, :17]), torch.tensor(xn[None, 0, 17:]))).item()))
            per_mu_seed.append(seed_probs); per_mu.append(float(np.mean(seed_probs)))
        p = float(np.mean(per_mu)); curves.append({"force_N": float(f), "p_success": p, "expected_utility": utility(p, float(f)), "posterior_member_p_success": per_mu, "model_seed_p_success": per_mu_seed})
    selected = max(curves, key=lambda x: (x["expected_utility"], -x["force_N"]))
    out = {"context_id": cid, "posterior": posterior, "force_bounds_N": [3.0, 5.0], "grid_step_N": 0.01, "marginalization": "mean over three physical-belief member means, with three clean feasibility seeds averaged within each mu", "selected": selected, "feasibility_p_monotonic": bool(all(curves[i+1]["p_success"] + 1e-9 >= curves[i]["p_success"] for i in range(len(curves)-1))), "curves": curves}
    out["feature_provenance"] = feature_provenance
    out["checkpoint_paths"] = [str(p) for p in paths]
    out["diagnostic_only"] = diagnostic_only
    out["final_method_deployment_approved"] = False
    destination = output_dir or OUT
    write_csv(destination / "SUCCESS_FORCE_CURVES" / f"{cid}{tag}.csv", curves)
    write_csv(destination / "UTILITY_FORCE_CURVES" / f"{cid}{tag}.csv", curves)
    write_json(destination / "FRICTION_POSTERIOR" / f"{cid}.json", posterior)
    write_json(destination / "SUCCESS_FORCE_CURVES" / f"{cid}{tag}.json", out)
    return out


def run_selector_worker(cid: str, probe_path: str, tag: str) -> int:
    try:
        posterior = posterior_from_probe(Path(probe_path))
        curves = infer_curves(cid, posterior, probe_path=Path(probe_path), tag=tag)
        write_json(OUT / "FRICTION_POSTERIOR" / f"{cid}{tag}.json", {
            "posterior": posterior,
            "selected": curves["selected"],
            "feasibility_p_monotonic": curves["feasibility_p_monotonic"],
            "probe_path": probe_path,
            "single_physical_probe": True,
            "posterior_marginalization": True,
            "expected_utility": True,
        })
        return 0
    except Exception as exc:
        write_json(OUT / "FRICTION_POSTERIOR" / f"{cid}{tag}_ERROR.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        return 1


def prepare() -> int:
    for d in ["PROBE_TELEMETRY", "FRICTION_POSTERIOR", "SUCCESS_FORCE_CURVES", "UTILITY_FORCE_CURVES", "ACTIVEFORCING_TRACES", "MAXFORCE_TRACES"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    ctx = contexts()
    manifest = {"schema": "ACTIVEFORCING_TASK0_SMOKE_V1", "task": TASK, "contexts": ctx, "context_count": 3, "single_physical_probe": True, "requery": False, "force_domain_N": [3.0, 5.0], "maxforce_setpoint_N": FMAX, "repeats": 2, "controller_changed": False, "probe_changed": False, "model_retrained": False, "utility_changed": False}
    write_json(OUT / "ACTIVEFORCING_TASK0_SMOKE_MANIFEST.json", manifest)
    write_json(OUT / "EXECUTION_CONTRACT.json", {"source": str(VALIDATION / "CURRENT_EXECUTION_CONTRACT.json"), "source_sha256": sha256(VALIDATION / "CURRENT_EXECUTION_CONTRACT.json"), "current_runner": str(P5_PATH), "current_runner_sha256": sha256(P5_PATH), "probe": str(P4_PATH), "probe_sha256": sha256(P4_PATH), "env_id": ENV_ID, "physics_rate_hz": 60.0, "outer_rate_hz": 20.0, "branch_contract": "restore post-probe state; same arm command path; only setpoint differs; 20 hold + 50 lift + 30 hold", "hold_success": "first lift >=0.01m then next 30 rows bilateral and dz>=0.005m", "force_metrics": "object-filtered bilateral true force; mean/top5/exposure over 100 active rows"})
    for plan in ctx:
        cid = plan["context_id"]
        for repeat in (1, 2):
            rc = launch({"AF_SMOKE_CONTEXT": cid}, OUT / "PROBE_TELEMETRY" / f"{cid}_PAIRED_R{repeat}.log", "paired", context=cid, repeat=repeat)
            if rc != 0:
                raise RuntimeError(f"paired probe/branches failed for {cid} repeat={repeat}, rc={rc}")
    return finalize()


def finalize() -> int:
    import glob
    rows = []
    for d in [OUT / "ACTIVEFORCING_TRACES", OUT / "MAXFORCE_TRACES"]:
        for p in d.glob("*.json"):
            if p.name.endswith("ERROR.json"): continue
            item = json.loads(p.read_text())
            if int(item.get("valid_branch", 0)) == 1:
                # Repair the derived hold label for already completed paired
                # traces using the frozen post-lift 30-step window.
                csv_path = p.with_suffix(".csv")
                if csv_path.exists():
                    with csv_path.open(newline="", encoding="utf-8") as f:
                        tr = list(csv.DictReader(f))
                    hold_rows = [r for r in tr if r.get("phase") == "lift_hold"]
                    item["hold_rows"] = len(hold_rows)
                    item["hold_30_step_success"] = int(len(hold_rows) == POST_LIFT_HOLD_STEPS and all(int(r["target_object_bilateral_contact"]) and float(r["object_dz_m"]) >= 0.005 for r in hold_rows))
                    write_json(p, item)
                rows.append(item)
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 branch results, got {len(rows)}")
    for r in rows:
        r["setpoint_saving_vs_5N"] = (FMAX - float(r["requested_force_N"])) / FMAX
    af = [r for r in rows if r["method"] == "ACTIVEFORCING"]; mx = [r for r in rows if r["method"] == "MAXFORCE"]
    def mean(key, arr): return float(np.mean([float(x[key]) for x in arr]))
    per_context = []
    for cid in sorted({r["context_id"] for r in rows}):
        a = [r for r in af if r["context_id"] == cid]; m = [r for r in mx if r["context_id"] == cid]
        x = json.loads((OUT / "FRICTION_POSTERIOR" / f"{cid}_repeat1.json").read_text())
        def saving(k): return (mean(k,m)-mean(k,a))/mean(k,m) if mean(k,m) else 0.0
        per_context.append({"context_id": cid, "mu_posterior_mean": x["posterior"]["mean"], "mu_posterior_std": x["posterior"]["std"], "selected_force_N": x["selected"]["force_N"], "predicted_success": x["selected"]["p_success"], "AF_lift_success": mean("lift_success",a), "AF_hold_success": mean("hold_30_step_success",a), "AF_mean_force": mean("measured_mean_force_N",a), "AF_top5_force": mean("measured_top5_force_N",a), "AF_force_exposure": mean("force_exposure_Ns",a), "Max_lift_success": mean("lift_success",m), "Max_hold_success": mean("hold_30_step_success",m), "Max_mean_force": mean("measured_mean_force_N",m), "Max_top5_force": mean("measured_top5_force_N",m), "Max_force_exposure": mean("force_exposure_Ns",m), "mean_force_saving": saving("measured_mean_force_N"), "top5_force_saving": saving("measured_top5_force_N"), "exposure_saving": saving("force_exposure_Ns")})
    fields = list(per_context[0].keys()); write_csv(OUT / "PER_CONTEXT_RESULTS.csv", per_context); write_csv(OUT / "PER_TRIAL_RESULTS.csv", rows)
    monotonic = all(json.loads((OUT / "SUCCESS_FORCE_CURVES" / (c["context_id"] + "_repeat1.json")).read_text())["feasibility_p_monotonic"] for c in per_context)
    summary = {
        "task": TASK, "context_count": 3, "activeforcing_valid_branches": len(af), "maxforce_valid_branches": len(mx),
        "activeforcing_lift_success_rate": mean("lift_success", af), "activeforcing_hold_success_rate": mean("hold_30_step_success", af),
        "maxforce_lift_success_rate": mean("lift_success", mx), "maxforce_hold_success_rate": mean("hold_30_step_success", mx),
        "mean_selected_setpoint_N": mean("requested_force_N", af), "min_selected_setpoint_N": min(float(r["requested_force_N"]) for r in af), "max_selected_setpoint_N": max(float(r["requested_force_N"]) for r in af),
        "AF_mean_measured_force_N": mean("measured_mean_force_N", af), "Max_mean_measured_force_N": mean("measured_mean_force_N", mx),
        "AF_mean_top5_force_N": mean("measured_top5_force_N", af), "Max_mean_top5_force_N": mean("measured_top5_force_N", mx),
        "AF_mean_force_exposure_Ns": mean("force_exposure_Ns", af), "Max_mean_force_exposure_Ns": mean("force_exposure_Ns", mx),
        "mean_force_reduction_percent": 100 * (mean("measured_mean_force_N", mx) - mean("measured_mean_force_N", af)) / mean("measured_mean_force_N", mx),
        "top5_force_reduction_percent": 100 * (mean("measured_top5_force_N", mx) - mean("measured_top5_force_N", af)) / mean("measured_top5_force_N", mx),
        "exposure_reduction_percent": 100 * (mean("force_exposure_Ns", mx) - mean("force_exposure_Ns", af)) / mean("force_exposure_Ns", mx),
        "feasibility_posterior_monotonic": monotonic, "activeforcing_collapses_to_max_force": all(abs(float(r["requested_force_N"]) - FMAX) < 1e-9 for r in af),
        "activeforcing_selector_too_aggressive": bool(mean("hold_30_step_success", af) < mean("hold_30_step_success", mx)),
        "matched_branch_parity": all(int(r["branch_state_parity"]) == 1 for r in rows),
        "controller_changed": False, "probe_changed": False, "model_retrained": False, "utility_changed": False,
    }
    write_json(OUT / "ACTIVEFORCING_VS_MAXFORCE_SUMMARY.json", {"summary": summary, "per_context": per_context})
    contract = json.loads((OUT / "EXECUTION_CONTRACT.json").read_text())
    contract.update({"physical_belief_model": [str(BELIEF_DIR / f"PHYSICAL_BELIEF_member_{i}.pt") for i in range(3)], "physical_belief_model_sha256": [sha256(BELIEF_DIR / f"PHYSICAL_BELIEF_member_{i}.pt") for i in range(3)], "feasibility_model": [str(FEAS_DIR / f"POSTERIOR_FEASIBILITY_seed{i}.pt") for i in range(3)], "feasibility_model_sha256": [sha256(FEAS_DIR / f"POSTERIOR_FEASIBILITY_seed{i}.pt") for i in range(3)], "selector": {"posterior_marginalization": "mean over the three physical-belief member posterior means, then mean over three clean feasibility seeds within each member", "utility": "p_success*(5-F)/5 + (1-p_success)*(-1)", "force_domain_N": [3.0, 5.0], "grid_step_N": 0.01}, "probe": {"single_physical_probe_per_repeat": True, "requery": False}})
    write_json(OUT / "EXECUTION_CONTRACT.json", contract)
    report = "# ActiveForcing task0 E2E smoke\n\n" + json.dumps(summary, indent=2) + "\n\n" + json.dumps({"contexts": [x["context_id"] for x in per_context], "posterior_means": [x["mu_posterior_mean"] for x in per_context], "posterior_stds": [x["mu_posterior_std"] for x in per_context], "selected_setpoints_N": [x["selected_force_N"] for x in per_context], "p_success_predicted": [x["predicted_success"] for x in per_context], "single_physical_probe": True, "posterior_marginalization": True, "expected_utility": True, "hold_definition": "30 rows in phase=lift_hold after commanded lift", "maxforce_setpoint_N": FMAX}, indent=2) + "\n\nThis is a three-context, two-repeat smoke; it is not a statistical significance claim. Force semantics are `CONTINUOUS_GRASP_FORCE_SETPOINT`; reported force effects are measured object-filtered interaction force.\n"
    (OUT / "ACTIVEFORCING_E2E_SMOKE_REPORT.md").write_text(report, encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--worker", action="store_true"); ap.add_argument("--mode", choices=["probe", "branch", "paired", "selector", "finalize"]); ap.add_argument("--context", default=""); ap.add_argument("--method", default=""); ap.add_argument("--force", type=float, default=0.0); ap.add_argument("--repeat", type=int, default=0); ap.add_argument("--probe-path", default=""); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global OUT
    OUT = Path(os.environ.get("AF_SMOKE_OUT", str(OUT)))
    if a.worker:
        if a.mode == "probe": return run_probe_worker(a.context)
        if a.mode == "branch": return run_branch_worker(a.context, a.method, a.force, a.repeat)
        if a.mode == "paired": return run_paired_worker(a.context, a.repeat)
        if a.mode == "selector": return run_selector_worker(a.context, a.probe_path, a.tag)
        if a.mode == "finalize": return finalize()
        raise RuntimeError("worker requires mode")
    return prepare()


if __name__ == "__main__":
    raise SystemExit(main())
