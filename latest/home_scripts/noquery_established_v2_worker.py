#!/usr/bin/env python3
"""Engineering smoke for a true no-physical-probe arm on the frozen final runtime.

This deliberately starts from the same geometry-based established-grasp
initializer used by the final paper runtime, skips only probe_out/probe_back,
selects grip force under the frozen empirical training prior, and leaves the
online VLA rollout/controller/evaluator unchanged.

Outputs from this script are engineering validation evidence.  They are not
paper-admissible confirmatory evidence without a separately frozen protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
FINAL_SNAPSHOT = Path(
    "/media/volume/newdata/exouser/online_vla_activeforcing_20260907/"
    "final_ablation_confirmatory_v1/SOURCE_SNAPSHOT"
)
STAGE_MANIFEST = FORTE / "analysis/results/current_matched_stage1_648_v1_20260906/STAGE_MANIFEST.json"
P5_SOURCE = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> int:
    # Import the exact frozen final modules before launching Isaac Sim.
    for path in (
        FINAL_SNAPSHOT,
        FORTE,
        FORTE / "analysis/results/current_runtime_recovery_v2_20260905",
        FORTE / "analysis/results/current_runtime_core_snapshot_v2_20260905",
        FORTE / "analysis/results/current_runtime_sensor_repair_v3_candidate_20260905",
        FORTE / "analysis/results/current_multitask58_loader_candidate_20260905",
        FORTE / "analysis/results/current_runtime_branch_execution_v6_20260905",
        TABERO,
    ):
        sys.path.insert(0, str(path))

    from common import V5, clean, read, sha, write

    out = Path(args.out)
    job = out / "branches" / args.job_name
    job.mkdir(parents=True, exist_ok=False)
    contexts = read(Path(args.plan))["contexts"]
    plan = dict(contexts[args.context])
    plan["source_context_id"] = plan["id"]
    plan["id"] = f"{plan['id']}__TRUE_NO_PHYSICAL_QUERY_V2"

    source_hashes = {
        str(Path(__file__).resolve()): file_sha(Path(__file__).resolve()),
        str(FINAL_SNAPSHOT / "runtime.py"): sha(FINAL_SNAPSHOT / "runtime.py"),
        str(FINAL_SNAPSHOT / "arbitration.py"): sha(FINAL_SNAPSHOT / "arbitration.py"),
        str(FINAL_SNAPSHOT / "phase_free_feasibility.py"): sha(FINAL_SNAPSHOT / "phase_free_feasibility.py"),
        str(FINAL_SNAPSHOT / "worker.py"): sha(FINAL_SNAPSHOT / "worker.py"),
        str(STAGE_MANIFEST): sha(STAGE_MANIFEST),
        str(Path(args.plan)): sha(Path(args.plan)),
    }
    write(
        job / "ENGINEERING_PROTOCOL.json",
        {
            "role": "ENGINEERING_SMOKE_NOT_PAPER_EVIDENCE",
            "method": "TRUE_NO_PHYSICAL_QUERY_TRAINING_PRIOR",
            "context": plan,
            "only_intended_runtime_changes": [
                "stop canonical prefix after established-grasp hold",
                "do not execute probe_out, probe_back, or post-probe hold",
                "replace instance posterior with frozen 48-context TRAIN empirical prior",
            ],
            "unchanged": [
                "geometry grasp initializer",
                "online frozen VLA checkpoint and 50/10 requery schedule",
                "arm-action arbitration",
                "gripper force servo",
                "350-step whole-mesh full-task evaluator",
            ],
            "source_hashes": source_hashes,
        },
    )

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        import measurement_hooks
        from activeforcing_execution_snapshot import capture
        from activeforcing_command_handoff import command_from_probe
        from geometry_grasp_initializer import install
        from phase_free_feasibility import PhaseFreeFeasibility

        runtime = load_module("noquery_v2_frozen_runtime", FINAL_SNAPSHOT / "runtime.py")
        official_worker = load_module("noquery_v2_frozen_worker", FINAL_SNAPSHOT / "worker.py")
        p5 = load_module("noquery_v2_task_sensors", P5_SOURCE)
        p5.OUT = job
        p5.P4_COLLECT = FORTE / "activeforcing_current_probe.py"
        p4 = p5.import_p4_probe(int(plan["task"]))
        if p4.OBJ_NAME != plan["object"]:
            raise RuntimeError("task/object mismatch")

        setup_task_objects("libero_object", int(plan["task"]))
        cfg = parse_env_cfg(p4.ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        getattr(cfg.scene, "contact_grasp_" + p4.OBJ_NAME).max_contact_data_count_per_prim = 128
        receipt = measurement_hooks.configure(cfg, plan)
        if receipt is not None:
            write(job / "MEASUREMENT_CONFIG.json", receipt)
        env = gym.make(p4.ENV_ID, cfg=cfg).unwrapped
        env.reset(seed=int(plan["root"]))
        binding = measurement_hooks.bind(env, plan)
        if binding is not None:
            write(job / "MEASUREMENT_BINDINGS.json", binding)

        records: list[dict] = []
        original_step = env.step

        def recording_step(action):
            result = original_step(action)
            obj = env.scene[p4.OBJ_NAME]
            records.append(
                {
                    "step": len(records) + 1,
                    "action": clean(action),
                    "eef_pose": clean(result[0]["policy"]["eef_pose"]),
                    "finger_joints": clean(env.scene["robot"].data.joint_pos[0, -2:]),
                    "aperture_m": float(env.scene["robot"].data.joint_pos[0, -2:].sum()),
                    "object_position": clean(obj.data.root_pos_w),
                    "object_velocity": clean(obj.data.root_lin_vel_w),
                    "target_object_force": p5.target_object_force_snapshot(env, p4, p4.OBJ_NAME),
                    "patches": clean(measurement_hooks.patches(env, plan, float(cfg.sim.dt))),
                    "controller_debug": clean(p4._dbg(env)),
                }
            )
            return result

        env.step = recording_step
        install(p4, int(plan["task"]), job)
        # Preserve the canonical approach/descend/close/hold prefix and remove
        # only the physical diagnostic displacement and its return/hold tail.
        p4.MAX_OUT_STEPS = 0
        p4.RETURN_STEPS = 0
        p4.POST_HOLD_STEPS = 0
        rows, prefix_record = p4.run_probe_episode(
            env,
            seed_idx=int(plan["root"]),
            mu=float(plan["mu"]),
            trial_id=plan["id"],
            dt=0.05,
            termination_signal=None,
        )
        raw = [vars(row) for row in rows]
        phases = {row["probe_phase"] for row in raw}
        forbidden = phases.intersection({"probe_out", "probe_back", "probe_hold"})
        if forbidden:
            raise RuntimeError(f"physical query unexpectedly executed: {sorted(forbidden)}")
        hold = [row for row in raw if row["probe_phase"] == "hold"]
        if len(hold) < 10 or any(row["contact_state"] != "bilateral" for row in hold[-10:]):
            raise RuntimeError("established-grasp bilateral-contact gate failed")
        if int(prefix_record.get("dropped", 0)):
            raise RuntimeError("object dropped during established-grasp prefix")

        write_csv(job / "NOQUERY_PREFIX.csv", raw)
        write(job / "NOQUERY_CONTACT_READBACK.json", records)
        write(
            job / "NOQUERY_PREFIX_ADMISSION.json",
            {
                "passed": True,
                "prefix_steps": len(raw),
                "phases": sorted(phases),
                "bilateral_last10": True,
                "physical_query_executed": False,
                "episode_step": int(env.episode_length_buf[0]),
            },
        )

        saved = capture(env)
        saved.update(candidate_actions_already_executed=0, observation_step=len(raw), context=plan)
        torch.save(saved, job / "DECISION_STATE.pt")
        runtime.save_reference_observation(env, plan, job)
        prepared = runtime.first_chunk(env, plan, job, int(args.port), job)
        chunk = prepared[3]
        x = official_worker.make_sequence(
            raw,
            saved,
            np.asarray(records[-1]["eef_pose"], np.float32).reshape(-1),
            p4,
            int(plan["task"]),
            chunk,
        )
        with (job / "PREACTION_SEQUENCE.npy").open("xb") as handle:
            np.save(handle, x)

        stage = read(STAGE_MANIFEST)
        train_roots = set(stage["selected_roots"]["TRAIN"])
        prior_nodes = np.asarray(
            [item["mu"] for item in stage["contexts"] if item["root"] in train_roots], dtype=float
        )
        if len(prior_nodes) != 48 or np.any(prior_nodes <= 0):
            raise RuntimeError("frozen 48-context empirical prior unavailable")
        prior_weights = np.full(len(prior_nodes), 1.0 / len(prior_nodes))
        feasibility = PhaseFreeFeasibility()
        posterior = {
            "interface": feasibility.manifest["posterior_interface"],
            "candidate_actions_executed": 0,
            "hidden_friction_used": False,
            "integration_nodes": prior_nodes.tolist(),
            "integration_weights": prior_weights.tolist(),
        }
        write(job / "NOQUERY_TRAINING_PRIOR.json", posterior)
        decision = feasibility.select(x, posterior)
        decision.update(
            method="TRUE_NO_PHYSICAL_QUERY_TRAINING_PRIOR",
            probe_executed=False,
            selected_force_source="FROZEN_48_CONTEXT_TRAINING_PRIOR_EXPECTED_UTILITY",
            runtime_hidden_friction_used=False,
            source_context_id=plan["source_context_id"],
        )
        write(job / "PLANNER_DECISION.json", decision)

        handoff = command_from_probe(
            records[-1]["action"],
            float(saved["objects"]["arm_action"]["_gripper_abs_cmd"][0, 0]),
            p4.D_CLOSED,
            p4.D_OPEN,
        )
        write(
            job / "HANDOFF.json",
            {
                "command": handoff,
                "source": "last established-grasp hold action",
                "physical_query_executed": False,
            },
        )
        env.step = original_step
        result = runtime.rollout(
            env,
            p4,
            p5,
            plan,
            job,
            float(decision["selected_force_N"]),
            handoff,
            prepared,
        )
        logical = bool(result.get("online_vla_verified")) and bool(result.get("outcome", {}).get("label_valid"))
        write(
            job / "WORKER_COMPLETION.json",
            {
                "logical_success": logical,
                "task_success": result.get("outcome", {}).get("full_task_success_y"),
                "method": "TRUE_NO_PHYSICAL_QUERY_TRAINING_PRIOR",
                "engineering_smoke_only": True,
            },
        )
        return 0 if logical else 2
    except Exception as exc:
        import traceback

        (job / "WORKER_ERROR.json").write_text(
            json.dumps({"error": repr(exc), "traceback": traceback.format_exc()}, indent=2) + "\n"
        )
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--job-name", default="smoke_context0")
    parser.add_argument("--port", type=int, default=18885)
    raise SystemExit(main(parser.parse_args()))
