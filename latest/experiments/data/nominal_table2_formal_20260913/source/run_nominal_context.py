#!/usr/bin/env python3
"""Run one formal Native/Nominal frozen-pi0 context from a natural reset.

No probe, ActiveForcing, force controller, or gripper arbitration is used.  The
13-dimensional postprocessed pi0 action is submitted byte-for-byte.  Outcome
and force metrics reuse the frozen Table-II runtime definitions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


TABERO = Path("/home/exouser/Tabero")
FORTE = Path("/home/exouser/FORTE")
FINAL = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_vla_v1")
SOURCE = FINAL / "SOURCE_SNAPSHOT"
P5 = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"
EXPECTED_CHECKPOINT_SHA256 = "0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17"
OPEN_INTENT_THRESHOLD_M = 0.039
CONTACT_THRESHOLD_N = 0.15
HORIZON = 350
REPLAN_STEPS = 10
TASKS = {
    0: ("alphabet_soup_1", "pick up the alphabet soup and place it in the basket"),
    1: ("cream_cheese_1", "pick up the cream cheese and place it in the basket"),
    5: ("tomato_sauce_1", "pick up the tomato sauce and place it in the basket"),
    6: ("butter_1", "pick up the butter and place it in the basket"),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clean(value):
    if hasattr(value, "detach"):
        return clean(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean(value), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def set_friction(env, name: str, mu: float, torch):
    view = env.scene[name].root_physx_view
    props = view.get_material_properties().clone()
    props[..., :2] = float(mu)
    view.set_material_properties(props, torch.arange(props.shape[0], dtype=torch.int32))
    got = view.get_material_properties().detach().cpu().numpy()
    if not np.allclose(got[..., :2], mu, rtol=0, atol=1e-6):
        raise RuntimeError("friction readback mismatch")
    return got


def settle(env, p4, torch, steps=20):
    obs = env.observation_manager.compute()
    eef = obs["policy"]["eef_pose"][0].detach().cpu().numpy()
    pos, aa = eef[:3].copy(), p4._aa(eef[3:7])
    for _ in range(steps):
        action = p4._make_action(pos, aa, p4.D_OPEN, 0.0, env.device)
        env.step(action)


def native_semantic_outcome(outcome, trace):
    """Remove only the AF canonical-command invariant from the frozen label.

    The AF controller rewrites an open request to exactly float32(0.04), so the
    original runtime verifies both semantic release intent and exact controller
    output.  Native VLA has no such rewrite; values >=0.039 are the frozen
    semantic release definition.  All physical task criteria stay unchanged.
    """
    opened = len(trace) >= 20 and all(r["vla_release_intent"] for r in trace[-20:])
    unheld = bool(outcome.get("unheld_last20"))
    inside = bool(outcome.get("inside_last50"))
    support = bool(outcome.get("final_support_contact"))
    lift = bool(outcome.get("lift_success"))
    dropped = bool(outcome.get("dropped"))
    timeout = any(r["terminations"].get("time_out", False) for r in trace)
    reset = any(r["physical_reset_requested"] for r in trace)
    complete = len(trace) == HORIZON
    flags = {"NO_LIFT": not lift, "DROP": dropped, "TIMEOUT": timeout,
        "RESET": reset, "INCOMPLETE_HORIZON": not complete,
        "OUTSIDE_AUTHORED_REGION": not inside,
        "NOT_RELEASED": not (opened and unheld),
        "NO_FINAL_SUPPORT_CONTACT": not support}
    outcome.update({
        "label_version": "NOMINAL_NATIVE_OPEN_INTENT_0P039_WITH_FROZEN_PHYSICAL_TASK_CRITERIA_V1",
        "full_task_success_y": int(not any(flags.values())),
        "place_success": int(inside and opened and unheld and support),
        "failure_reasons": [k for k, v in flags.items() if v],
        "opened_last20": opened,
        "native_release_semantics": "raw gripper command >=0.039; no canonical 0.04 rewrite",
        "af_exact_0p04_controller_invariant_not_applied": True,
    })
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18895)
    ap.add_argument("--task", type=int, choices=sorted(TASKS), required=True)
    ap.add_argument("--root", type=int, required=True)
    ap.add_argument("--band", required=True)
    ap.add_argument("--friction", type=float, required=True)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    obj, prompt = TASKS[args.task]
    plan = {"id": f"t{args.task}_r{args.root}_{args.band.lower()}__NOMINAL_VLA",
            "task": args.task, "root": args.root, "band": args.band,
            "mu": args.friction, "object": obj, "target": "basket_1"}
    write(out / "PROTOCOL.json", {
        "method": "NOMINAL_VLA", "plan": plan, "prompt": prompt,
        "probe_used": False, "activeforcing_used": False,
        "force_or_gripper_override_used": False, "commanded_force_N": None,
        "execution": "natural reset, 20 open-hold settling steps, then byte-identical postprocessed pi0 actions",
        "comparison_note": "same root/task/friction strata as Table II; not post-probe state-paired",
        "metric_contract": "ONLINE_VLA_350STEP_WHOLE_MESH_RELEASE_SUPPORT_V1 plus contact-conditioned bilateral squeeze",
        "runner_sha256": sha(Path(__file__).resolve()),
    })

    os.environ.update({
        "HDF5_TRAJ_SOURCE_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5",
        "LIBERO_CONFIG_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/config",
        "LIBERO_ASSETS_DATA_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/USD",
        "TASK_SUITE": "libero_object", "TASK_ID": str(args.task),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
    })
    sys.path[:0] = [str(SOURCE), str(FORTE),
        str(FORTE / "analysis/results/current_runtime_recovery_v2_20260905"),
        str(TABERO), str(TABERO / "analysis"),
        str(TABERO / "benchmarks/openpi/openpi-client/src")]

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    trace = []
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        import full_task_label as legacy_label

        p5 = load_module(f"nominal_p5_t{args.task}", P5)
        p4 = p5.import_p4_probe(args.task)
        setup_task_objects("libero_object", args.task)
        cfg = parse_env_cfg(p4.ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        getattr(cfg.scene, "contact_grasp_" + obj).max_contact_data_count_per_prim = 128
        env = gym.make(p4.ENV_ID, cfg=cfg).unwrapped
        env.reset(seed=args.root)
        material = set_friction(env, obj, args.friction, torch)
        settle(env, p4, torch)
        z0 = float(env.scene[obj].data.root_pos_w[0, 2])

        runtime = load_module("nominal_frozen_table2_runtime", SOURCE / "runtime.py")
        geom = legacy_label.geom
        manifest = json.loads((FORTE / "analysis/results/current_runtime_recovery_v2_20260905/TASK_GEOMETRY_MANIFEST.json").read_text())
        taskgeo = next(x for x in manifest["tasks"] if x["task"] == args.task)
        if runtime.sha(taskgeo["label_geometry"]) != taskgeo["label_sha256"]:
            raise RuntimeError("label geometry changed")
        with np.load(taskgeo["label_geometry"]) as f:
            vertices = f["vertices"]
            region = geom.Region(f["basket_from_site"], f["half_size"])

        client = runtime.OnlineClient(out, plan, args.port)
        build, Buffer, adapter_sha = runtime.observation_builder()
        buffer = Buffer()
        write(out / "INTERFACE.json", {"checkpoint_sha256": client.metadata["checkpoint_sha256"],
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "observation_adapter_sha256": adapter_sha, "material_readback": material})

        terminals = []
        original_compute, original_reset = env.termination_manager.compute, env._reset_idx
        def compute():
            original_compute()
            failures = torch.zeros_like(env.termination_manager.terminated)
            for name in env.termination_manager.active_terms:
                if name != "success" and not env.termination_manager.get_term_cfg(name).time_out:
                    failures.logical_or_(env.termination_manager.get_term(name))
            env.termination_manager._terminated_buf.copy_(failures)
            return env.termination_manager.dones
        def reset(ids):
            if len(ids):
                terminals.append(runtime.observation(env, p4, p5, plan, geom))
        env.termination_manager.compute, env._reset_idx = compute, reset
        error = None
        try:
            chunk = proof = None
            for step in range(1, HORIZON + 1):
                if step == 1 or (step - 1) % REPLAN_STEPS == 0:
                    obs_now = env.observation_manager.compute()
                    cameras = {name: env.scene[name].data.output["rgb"][0].detach().cpu().numpy().copy()
                               for name in ("agentview_cam", "eye_in_hand_cam")}
                    payload = {k: v for k, v in build(env, obs_now, prompt, buffer).items() if isinstance(k, str)}
                    chunk, proof = client.infer(payload, step, cameras)
                index = (step - 1) % REPLAN_STEPS
                raw = np.ascontiguousarray(chunk[index, :13], dtype=np.float32)
                before = int(env.episode_length_buf[0])
                env.step(torch.from_numpy(raw.copy()).reshape(1, 13).to(env.device))
                rec = terminals[-1] if terminals else runtime.observation(env, p4, p5, plan, geom)
                release = bool(raw[6] >= OPEN_INTENT_THRESHOLD_M)
                row = {"branch_step": step, "request_id": proof["request_id"], "chunk_index": index,
                    "action": raw.tolist(), "action_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                    "raw_vla_gripper_command": float(raw[6]), "final_gripper_command": float(raw[6]),
                    "vla_release_intent": release, "physical_reset_requested": bool(terminals),
                    "native_action_submitted_without_override": True, **rec}
                trace.append(row)
                with (out / "ACTION_TRACE.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(clean(row), allow_nan=False) + "\n")
                if terminals:
                    break
                if int(env.episode_length_buf[0]) != before + 1:
                    raise RuntimeError("unexpected episode clock")
        except Exception as exc:
            error = {"error": repr(exc), "traceback": traceback.format_exc()}
        finally:
            env.termination_manager.compute, env._reset_idx = original_compute, original_reset

        outcome = runtime.evaluate(trace, vertices, region, geom, z0, error=error)
        if outcome.get("label_valid"):
            outcome = native_semantic_outcome(outcome, trace)
        nonrelease = [r for r in trace if not r["vla_release_intent"]]
        contact = [r["measured_bilateral_squeeze"] for r in nonrelease
                   if r["normal_force_N"][0] >= CONTACT_THRESHOLD_N and r["normal_force_N"][1] >= CONTACT_THRESHOLD_N]
        outcome["contact_conditional_squeeze_N"] = float(np.mean(contact)) if contact else None
        outcome["nonrelease_samples"] = len(nonrelease)
        outcome["bilateral_contact_samples"] = len(contact)
        outcome["contact_fraction"] = len(contact) / len(nonrelease) if nonrelease else None
        result = {"plan": plan, "method": "NOMINAL_VLA", "commanded_force_N": None,
            "steps": len(trace), "rpc_count": client.calls, "outcome": outcome,
            "online_vla_verified": error is None and client.calls == (len(trace) + REPLAN_STEPS - 1) // REPLAN_STEPS,
            "checkpoint_sha256": client.metadata["checkpoint_sha256"], "error": error}
        write(out / "RESULT.json", result)
        print("NOMINAL_CONTEXT_RESULT=" + json.dumps(clean(result), sort_keys=True), flush=True)
        return 0 if error is None and outcome.get("label_valid") else 2
    except Exception as exc:
        write(out / "ERROR.json", {"error": repr(exc), "traceback": traceback.format_exc(), "steps": len(trace)})
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        try: app.close()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
