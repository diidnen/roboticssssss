#!/usr/bin/env python3
"""P5-S0 true matched post-probe branching preflight.

This is intentionally small and simulator-scoped. It imports the frozen P4B
common probe implementation, executes one probe for each requested physical
context, saves the exact post-probe scene state, and resets to that state before
each force branch.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path("/home/exouser/Tabero")
RESULTS_ROOT = REPO / "analysis/results"
P4 = RESULTS_ROOT / "p4_contact_conditioned_probe_20260822_184213"
TASKS = [0, 1, 5, 6]
TASK_OBJECTS = {0: "alphabet_soup_1", 1: "cream_cheese_1", 5: "tomato_sauce_1", 6: "butter_1"}
BASKET_NAME = "basket_1"
TASK_SUITE = "libero_object"
ENV_ID = "Isaac-Libero-Franka-Hybrid-Tactile-v0"
F_DEV = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
SEED = 5050


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


OUT = Path(os.environ.get("P5S0_TRUE_OUT", RESULTS_ROOT / f"p5s0_true_matched_preflight_{now_tag()}"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha_state(state) -> str:
    import torch

    chunks = []

    def visit(x):
        if isinstance(x, dict):
            for key in sorted(x):
                chunks.append(str(key).encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)
        elif isinstance(x, torch.Tensor):
            chunks.append(x.detach().cpu().contiguous().numpy().tobytes())
        else:
            chunks.append(repr(x).encode())

    visit(state)
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def deterministic_mus(n: int) -> list[float]:
    rng = np.random.default_rng(SEED)
    return [float(x) for x in rng.uniform(0.2, 1.0, size=n)]


def import_p4_probe(task_id: int):
    os.environ["P4_TASK_ID"] = str(task_id)
    os.environ["P4_VARIANT"] = "P4B"
    os.environ["P4_OUT"] = str(OUT / "P5S0_FROZEN_P4B_IMPORT")
    os.environ["P4_RESUME"] = "0"
    path = P4 / "scripts/p4_collect_probe.py"
    old_stdout, old_stderr = sys.stdout, sys.stderr
    spec = importlib.util.spec_from_file_location(f"p5s0_p4b_task{task_id}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        sys.stdout.close()
    except Exception:
        pass
    sys.stdout, sys.stderr = old_stdout, old_stderr
    return mod


def downstream_branch(env, p4, *, task_id: int, force: float, trial_id: str, dt: float) -> dict:
    obj_name = TASK_OBJECTS[task_id]
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    eef_aa = p4._aa(eef[3:7])
    cmd_pos = eef[:3].copy()
    obj0_w = env.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy()
    basket_b, _ = p4._pose_in_base(env, BASKET_NAME)

    lift = cmd_pos.copy()
    lift[2] = max(lift[2] + 0.142, cmd_pos[2] + 0.08)
    transit = basket_b.copy()
    transit[2] = max(lift[2], basket_b[2] + 0.18)
    place = basket_b.copy()
    place[2] = basket_b[2] + 0.10

    phases = [
        ("branch_hold", 20, cmd_pos.copy(), "track"),
        ("lift", 50, lift, "track"),
        ("transit", 110, transit, "freeze"),
        ("over_basket", 30, transit, "freeze"),
        ("place", 40, place, "freeze"),
        ("release", 50, place, "open"),
        ("settle", 50, place, "open"),
    ]
    d_pred = p4.D_CLOSED
    step = 0
    dropped = lift_success = pick_success = lost_in_transit = timed_out = 0
    max_basket = peak_force = integrated_force = 0.0
    force_samples = []
    term_reason = ""

    for phase, n_steps, target, mode in phases:
        start = cmd_pos.copy()
        for i in range(n_steps):
            cmd_pos = p4._interp(start, target, i, n_steps)
            if mode == "open":
                d_pred = p4.D_OPEN
            action = p4._make_action(cmd_pos, eef_aa, d_pred, force if mode != "open" else 0.0, env.device)
            obs, _, term, trunc, _ = env.step(action)
            step += 1
            f_sq = p4._f(p4._dbg(env).get("f_sq_meas"), 0.0)
            peak_force = max(peak_force, f_sq)
            integrated_force += f_sq * dt
            if mode == "track":
                d_pred = p4._force_servo(d_pred, f_sq, force)
            if mode != "open":
                force_samples.append(f_sq)
            try:
                dropped = int(bool(env.termination_manager.get_term("object_1_dropped")[0].item()))
            except Exception:
                pass
            obj_p = env.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy()
            obj_dz = float(obj_p[2] - obj0_w[2])
            if obj_dz >= 0.01:
                pick_success = 1
            if obj_dz >= 0.03:
                lift_success = 1
            if lift_success and phase in ("transit", "over_basket") and (dropped or obj_dz < 0.02):
                lost_in_transit = 1
            try:
                import torch

                cs = env.scene[f"contact_{BASKET_NAME}_{obj_name}"]
                bc = float(torch.linalg.vector_norm(cs.data.force_matrix_w.reshape(-1, 3)[0]).item())
                max_basket = max(max_basket, bc)
            except Exception:
                pass
            if bool(term[0].item()) or bool(trunc[0].item()):
                try:
                    if bool(env.termination_manager.get_term("time_out")[0].item()):
                        timed_out = 1
                        term_reason = "time_out"
                except Exception:
                    term_reason = "term"
                break
        else:
            continue
        break

    success = int(lift_success == 1 and max_basket > 0.05 and dropped == 0)
    measured_mean = float(np.mean(force_samples)) if force_samples else 0.0
    return {
        "branch_id": trial_id,
        "task": task_id,
        "requested_force_N": float(force),
        "measured_force_mean_N": measured_mean,
        "measured_force_peak_N": float(peak_force),
        "steady_state_mean_N": measured_mean,
        "force_tracking_error_N": measured_mean - float(force),
        "integrated_force": float(integrated_force),
        "pick_success": pick_success,
        "lift_success": lift_success,
        "transport_retention": int(lift_success == 1 and lost_in_transit == 0),
        "place_success": int(max_basket > 0.05),
        "full_task_success_y": success,
        "failure_reason": "" if success else (term_reason or "full_task_failure"),
        "dropped": dropped,
        "lost_in_transit": lost_in_transit,
        "timeout": timed_out,
        "steps": step,
        "t_episode_s": step * dt,
        "label_source": "TRUE_POST_PROBE_RESET_MATCHED_BRANCH",
    }


def main() -> None:
    from isaaclab.app import AppLauncher

    n_frictions = int(os.environ.get("P5S0_N_FRICTIONS", "1"))
    n_contexts_per_task = int(os.environ.get("P5S0_CONTEXTS_PER_TASK", "1"))
    tasks = [int(x) for x in os.environ.get("P5S0_TASKS", "0").split(",") if x.strip()]
    forces = [float(x) for x in os.environ.get("P5S0_FORCES", ",".join(str(x) for x in F_DEV)).split(",") if x.strip()]
    mus = deterministic_mus(n_frictions)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "P5S0_RAW_PROBE_TELEMETRY").mkdir(exist_ok=True)
    write_csv(
        OUT / "P5S0_TRUE_PREFLIGHT_REQUESTED_MANIFEST.csv",
        [
            {"task": task, "hidden_friction_analysis_only": mu, "seed": 3000 + seed_i}
            for task in tasks
            for mu in mus
            for seed_i in range(n_contexts_per_task)
        ],
    )

    app_launcher = AppLauncher(headless=True, enable_cameras=True, num_envs=1)
    simulation_app = app_launcher.app
    context_rows: list[dict] = []
    branch_rows: list[dict] = []
    try:
        import gymnasium as gym
        import tac_manip.tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects

        for task_id in tasks:
            if task_id not in TASKS:
                raise ValueError(f"P5S0 task must be one of {TASKS}, got {task_id}")
            p4 = import_p4_probe(task_id)
            setup_task_objects(TASK_SUITE, task_id)
            env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)
            env_cfg.episode_length_s = 45.0
            env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
            dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
            for seed_i in range(n_contexts_per_task):
                seed = 3000 + seed_i
                for mu in mus:
                    context_id = f"p5s0_true_t{task_id}_s{seed}_mu{mu:.6f}"
                    probe_rows, probe_rec = p4.run_probe_episode(env, seed_idx=seed, mu=mu, trial_id=context_id, dt=dt)
                    sq = env.scene.get_state(is_relative=True)
                    sq_hash = sha_state(sq)
                    write_csv(OUT / "P5S0_RAW_PROBE_TELEMETRY" / f"{context_id}.csv", [asdict(r) for r in probe_rows])
                    context_rows.append(
                        {
                            "context_id": context_id,
                            "task": task_id,
                            "hidden_friction_analysis_only": float(mu),
                            "seed": seed,
                            "probe_primitive": "P4B_common_contact_frame_shear",
                            "probe_qualified": int(probe_rec.get("qualified", 0)),
                            "contact_retained": int(1 - int(probe_rec.get("contact_lost_probe", 0))),
                            "drop": int(probe_rec.get("dropped", 0)),
                            "stop_reason": probe_rec.get("stop_trigger", ""),
                            "actual_displacement_mm": float(probe_rec.get("actual_probe_displacement_mm", np.nan)),
                            "post_probe_state_hash": sq_hash,
                            "label_source": "TRUE_SINGLE_PROBE_CONTEXT",
                        }
                    )
                    for force in forces:
                        env.reset_to(sq, torch.tensor([0], device=env.device), is_relative=True)
                        br = downstream_branch(
                            env,
                            p4,
                            task_id=task_id,
                            force=force,
                            trial_id=f"{context_id}_F{force:g}",
                            dt=dt,
                        )
                        br["context_id"] = context_id
                        br["hidden_friction_analysis_only"] = float(mu)
                        br["seed"] = seed
                        br["post_probe_state_hash"] = sq_hash
                        branch_rows.append(br)
                        write_csv(OUT / "P5S0_CONTEXT_MANIFEST.csv", context_rows)
                        write_csv(OUT / "P5S0_BRANCH_MANIFEST.csv", branch_rows)
                        write_csv(OUT / "P5S0_FULL_TASK_BRANCH_RESULTS.csv", branch_rows)
            env.close()
        write_csv(OUT / "P5S0_CONTEXT_MANIFEST.csv", context_rows)
        write_csv(OUT / "P5S0_BRANCH_MANIFEST.csv", branch_rows)
        write_csv(OUT / "P5S0_FULL_TASK_BRANCH_RESULTS.csv", branch_rows)
        (OUT / "P5S0_TRUE_PREFLIGHT_STATUS.json").write_text(
            json.dumps(
                {
                    "status": "TRUE_MATCHED_PREFLIGHT_COMPLETE",
                    "contexts": len(context_rows),
                    "branches": len(branch_rows),
                    "tasks": tasks,
                    "forces": forces,
                    "reset_post_probe_branch_parity": True,
                    "probe": "P4B_common_contact_frame_shear",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        (OUT / "P5S0_TRUE_PREFLIGHT_ERROR.json").write_text(
            json.dumps({"error": traceback.format_exc()}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    print(OUT)


if __name__ == "__main__":
    main()
