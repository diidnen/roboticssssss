#!/usr/bin/env python3
"""Paired AF versus strict No-Query from one captured pre-probe state.

Strict No-Query executes first from the live established-grasp state.  The
same exposed state, manager/controller caches, targets, materials, and RNG are
then restored before the canonical AF probe suffix resumes.  The snapshot API
does not expose PhysX contact/solver warm-start state; that limitation is
recorded rather than hidden.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np


FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
FINAL = Path(
    "/media/volume/newdata/exouser/online_vla_activeforcing_20260907/"
    "final_ablation_confirmatory_v1/SOURCE_SNAPSHOT"
)
P4_CANONICAL = FORTE / "activeforcing_current_probe.py"
P5_SOURCE = TABERO / "analysis/p5s0c_paired_boundary_probe_value.py"
SNAPSHOT_DIR = FORTE / "analysis/results/current_contract_restore_and_matched_pilot_20260905/initial_sources"
SNAPSHOT_GROUPS = ("state", "environment", "objects", "joint_targets", "materials", "rng")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identical(a, b) -> bool:
    """Exact recursive comparator with no dependency on feasibility code."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    if hasattr(b, "detach"):
        b = b.detach().cpu().numpy()
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            identical(a[key], b[key]) for key in a
        )
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(
            identical(x, y) for x, y in zip(a, b)
        )
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        return bool(
            np.array_equal(x, y, equal_nan=True)
            if x.dtype.kind in "fc"
            else np.array_equal(x, y)
        )
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return type(a) is type(b) and a == b


def write_json(path: Path, value) -> None:
    def clean(item):
        if hasattr(item, "detach"):
            return item.detach().cpu().numpy().tolist()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(clean(value), handle, indent=2, sort_keys=True, allow_nan=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_patch_data(sensor, dt: float) -> dict:
    """Exact contact-patch readback without importing the feasibility stack."""
    view = sensor.contact_physx_view
    force, points, normals, separation, counts, starts = [
        value.detach().cpu().numpy().copy() for value in view.get_contact_data(dt)
    ]
    normal_pairs = []
    for index in np.ndindex(counts.shape):
        start, count = int(starts[index]), int(counts[index])
        if count and start + count >= view.max_contact_data_count:
            raise RuntimeError("contact patch buffer possibly saturated")
        ids = slice(start, start + count)
        normal_pairs.append(
            {
                "pair": index,
                "count": count,
                "normal_forces": force[ids],
                "points": points[ids],
                "normals": normals[ids],
                "separations": separation[ids],
                "sum_normal_vector_world": (force[ids] * normals[ids]).sum(0),
            }
        )
    friction, fpoints, fcounts, fstarts = [
        value.detach().cpu().numpy().copy() for value in view.get_friction_data(dt)
    ]
    friction_pairs = []
    for index in np.ndindex(fcounts.shape):
        start, count = int(fstarts[index]), int(fcounts[index])
        if count and start + count >= view.max_contact_data_count:
            raise RuntimeError("friction patch buffer possibly saturated")
        ids = slice(start, start + count)
        friction_pairs.append(
            {
                "pair": index,
                "count": count,
                "friction_forces_world": friction[ids],
                "points": fpoints[ids],
                "sum_friction_vector_world": friction[ids].sum(0),
            }
        )
    return {
        "normal": normal_pairs,
        "friction": friction_pairs,
        "physics_dt": dt,
        "capacity": view.max_contact_data_count,
    }


def instrument_probe_source(destination: Path) -> dict:
    canonical = P4_CANONICAL.read_text()
    signature = (
        "def run_probe_episode(env, *, seed_idx: int, mu: float, trial_id: str, "
        "dt: float, termination_signal=None):"
    )
    replacement_signature = signature[:-2] + ", preprobe_callback=None):"
    anchor = "    probe_start = grasp.copy()\n    cmd_pos = probe_start.copy()\n\n    if not terminated:\n"
    replacement_anchor = (
        "    probe_start = grasp.copy()\n"
        "    cmd_pos = probe_start.copy()\n\n"
        "    if preprobe_callback is not None:\n"
        "        preprobe_callback(env=env, rows=rows, preload_target=preload_target, "
        "d_pred=d_pred, cmd_pos=cmd_pos.copy(), terminated=terminated)\n\n"
        "    if not terminated:\n"
    )
    if canonical.count(signature) != 1 or canonical.count(anchor) != 1:
        raise RuntimeError("canonical probe source no longer matches frozen instrumentation anchors")
    derived = canonical.replace(signature, replacement_signature).replace(anchor, replacement_anchor)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        handle.write(derived)
    return {
        "canonical_source": str(P4_CANONICAL),
        "canonical_sha256": file_sha(P4_CANONICAL),
        "instrumented_source": str(destination),
        "instrumented_sha256": file_sha(destination),
        "semantic_change": "one preprobe callback after canonical prefix and before first probe action",
        "canonical_lines_replaced": [signature, anchor],
    }


def main(args: argparse.Namespace) -> int:
    for path in (
        FINAL,
        FORTE,
        FORTE / "analysis/results/current_runtime_recovery_v2_20260905",
        FORTE / "analysis/results/current_runtime_core_snapshot_v2_20260905",
        FORTE / "analysis/results/current_runtime_sensor_repair_v3_candidate_20260905",
        FORTE / "analysis/results/current_multitask58_loader_candidate_20260905",
        FORTE / "analysis/results/current_runtime_branch_execution_v6_20260905",
        SNAPSHOT_DIR,
        TABERO,
    ):
        sys.path.insert(0, str(path))

    from common import INSTRUCTIONS, V5, array_sha, clean, payload_sha, read, sha, write

    out = Path(args.out)
    pair = out / "contexts" / args.job_name
    strict_job = pair / "STRICT_NOQUERY"
    af_job = pair / "ACTIVEFORCING"
    reference = pair / "PREPROBE_REFERENCE"
    acquisition = pair / "ACQUISITION"
    for directory in (strict_job, af_job, reference, acquisition):
        directory.mkdir(parents=True, exist_ok=False)

    plan = dict(read(Path(args.plan))["contexts"][args.context])
    freeze = read(Path(args.force_freeze))
    if freeze["selection_split"] != "TRAIN_ONLY" or freeze["test_outcomes_accessed"]:
        raise RuntimeError("fixed-force freeze is not TRAIN-only")
    fixed_force = float(freeze["frozen_strict_noquery_policy"]["force_by_task"][str(plan["task"])])
    if freeze["frozen_strict_noquery_policy"] != {
        "kind": "per_task_fixed_force",
        "force_by_task": freeze["frozen_strict_noquery_policy"]["force_by_task"],
        "uses_probe": False,
        "loads_belief": False,
        "loads_feasibility": False,
    }:
        raise RuntimeError("unexpected strict policy contract")

    instrumentation = instrument_probe_source(pair / "SOURCE" / "activeforcing_current_probe_hook.py")
    protocol = {
        "schema": "PAIRED_AF_STRICT_NOQUERY_PREPROBE_V1",
        "context": plan,
        "branch_order": ["STRICT_NOQUERY_LIVE_PREPROBE", "RESTORE_PREPROBE", "ACTIVEFORCING_PROBE_SUFFIX"],
        "strict_force_N": fixed_force,
        "strict_information_contract": {
            "physical_probe_actions": 0,
            "belief_modules_loaded_before_or_during_strict": False,
            "feasibility_modules_loaded_before_or_during_strict": False,
            "force_source": "frozen TRAIN-only task lookup",
        },
        "pairing_contract": {
            "same_captured_preprobe_state": True,
            "same_context_id_for_vla_seed_schedule": plan["id"],
            "same_vla_noise_seed_per_common_replan_step": True,
            "known_limit": "unexposed PhysX contact/solver warm-start state is not in snapshot API",
        },
        "force_freeze_path": str(Path(args.force_freeze).resolve()),
        "force_freeze_sha256": sha(Path(args.force_freeze)),
        "instrumentation": instrumentation,
    }
    write_json(pair / "PAIR_PROTOCOL.json", protocol)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, num_envs=1).app
    env = None
    strict_result = None
    af_result = None
    try:
        import gymnasium as gym
        import torch
        import tac_manip.tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tac_manip.utils.task_configs import setup_task_objects
        import measurement_hooks
        from activeforcing_execution_snapshot import capture, restore
        from activeforcing_command_handoff import command_from_probe
        from geometry_grasp_initializer import install
        from activeforcing_probe_friction_contract import FrictionProbeBudget
        from assembly_force_adapter import BODIES, merge_object_centric_patches

        runtime = load_module("paired_frozen_runtime", FINAL / "runtime.py")
        p5 = load_module("paired_task_sensors", P5_SOURCE)
        p5.OUT = acquisition
        p5.P4_COLLECT = pair / "SOURCE" / "activeforcing_current_probe_hook.py"
        p4 = p5.import_p4_probe(int(plan["task"]))
        if p4.OBJ_NAME != plan["object"]:
            raise RuntimeError("task/object mismatch")

        setup_task_objects("libero_object", int(plan["task"]))
        cfg = parse_env_cfg(p4.ENV_ID, device="cuda:0", num_envs=1)
        cfg.episode_length_s = 45.0
        getattr(cfg.scene, "contact_grasp_" + p4.OBJ_NAME).max_contact_data_count_per_prim = 128
        receipt = measurement_hooks.configure(cfg, plan)
        if receipt is not None:
            write_json(acquisition / "MEASUREMENT_CONFIG.json", receipt)
        env = gym.make(p4.ENV_ID, cfg=cfg).unwrapped
        env.reset(seed=int(plan["root"]))
        binding = measurement_hooks.bind(env, plan)
        if binding is not None:
            write_json(acquisition / "MEASUREMENT_BINDINGS.json", binding)

        records: list[dict] = []
        original_step = env.step

        def recording_step(action):
            result = original_step(action)
            obj = env.scene[p4.OBJ_NAME]
            records.append(
                {
                    "step": len(records) + 1,
                    "action": clean(action),
                    "policy_local_normal_projection": clean(result[0]["policy"]["gripper_net_force"]),
                    "eef_pose": clean(result[0]["policy"]["eef_pose"]),
                    "finger_joints": clean(env.scene["robot"].data.joint_pos[0, -2:]),
                    "aperture_m": float(env.scene["robot"].data.joint_pos[0, -2:].sum()),
                    "object_position": clean(obj.data.root_pos_w),
                    "object_velocity": clean(obj.data.root_lin_vel_w),
                    "object_quaternion": clean(obj.data.root_quat_w),
                    "target_object_force": p5.target_object_force_snapshot(env, p4, p4.OBJ_NAME),
                    "patches": clean(
                        merge_object_centric_patches(
                            raw_patch_data(env.scene["current_assembly_target"], float(cfg.sim.dt)),
                            BODIES,
                        )
                    ),
                    "controller_debug": clean(p4._dbg(env)),
                }
            )
            return result

        env.step = recording_step
        install(p4, int(plan["task"]), acquisition)

        def preprobe_callback(*, env, rows, preload_target, d_pred, cmd_pos, terminated):
            nonlocal strict_result
            raw = [vars(row).copy() for row in rows]
            phases = {row["probe_phase"] for row in raw}
            if len(raw) != 190 or phases != {"approach", "descend", "close", "hold"}:
                raise RuntimeError(f"unexpected established-grasp prefix: {len(raw)} {phases}")
            if terminated:
                raise RuntimeError("established-grasp prefix terminated")
            hold = [row for row in raw if row["probe_phase"] == "hold"]
            if len(hold) < 10 or any(row["contact_state"] != "bilateral" for row in hold[-10:]):
                raise RuntimeError("established-grasp bilateral-contact gate failed")

            write_csv(reference / "ESTABLISHED_PREFIX.csv", raw)
            write_json(reference / "ESTABLISHED_CONTACT_READBACK.json", records)
            obj = env.scene[p4.OBJ_NAME]
            mats = obj.root_physx_view.get_material_properties().cpu().numpy().reshape(-1, 3)
            if not np.allclose(mats[:, :2], plan["mu"], rtol=0, atol=1e-6):
                raise RuntimeError("simulator material readback mismatch")

            # Freeze the actual first VLA payload/camera cache, then capture all
            # exposed execution state at the shared pre-probe decision point.
            build, Buffer, adapter_sha = runtime.observation_builder()
            cameras = {
                name: env.scene[name].data.output["rgb"][0].detach().cpu().numpy().copy()
                for name in ("agentview_cam", "eye_in_hand_cam")
            }
            payload = build(env, env.observation_manager.compute(), INSTRUCTIONS[plan["task"]], Buffer())
            payload = {key: value for key, value in payload.items() if isinstance(key, str)}
            arrays = {f"payload__{key}": value for key, value in payload.items() if isinstance(value, np.ndarray)}
            arrays.update({f"camera__{key}": value for key, value in cameras.items()})
            with (reference / "COMMON_POSTPROBE_OBSERVATION.npz").open("xb") as handle:
                np.savez_compressed(handle, **arrays)

            saved = capture(env)
            saved.update(candidate_actions_already_executed=0, observation_step=len(raw), context=plan)
            torch.save(saved, reference / "DECISION_STATE.pt")
            write_json(
                reference / "COMMON_POSTPROBE_OBSERVATION.json",
                {
                    "path": str(reference / "COMMON_POSTPROBE_OBSERVATION.npz"),
                    "sha256": sha(reference / "COMMON_POSTPROBE_OBSERVATION.npz"),
                    "payload_sha256": payload_sha(payload),
                    "episode_step": int(env.episode_length_buf[0]),
                    "decision_state_sha256": sha(reference / "DECISION_STATE.pt"),
                    "observation_adapter_sha256": adapter_sha,
                    "camera_cache_snapshot": True,
                    "vla_actions_cached": False,
                    "candidate_actions_executed": 0,
                    "semantic_point": "PREPROBE_ESTABLISHED_GRASP",
                },
            )
            write_json(
                reference / "PREPROBE_METADATA.json",
                {
                    "context": plan,
                    "prefix_steps": len(raw),
                    "episode_step": int(env.episode_length_buf[0]),
                    "preload_target_N": preload_target,
                    "last_gripper_command": d_pred,
                    "cmd_pos": cmd_pos,
                    "actual_material_analysis_only": mats,
                    "candidate_actions_executed": 0,
                    "physical_probe_actions_executed": 0,
                    "snapshot_excluded": saved["excluded"],
                },
            )

            forbidden_modules = [
                name
                for name in sys.modules
                if name in {"continuous_belief", "phase_free_feasibility", "online_ablation_feasibility"}
                or "feasibility" in name.lower()
            ]
            if forbidden_modules:
                raise RuntimeError("strict branch loaded forbidden modules: " + repr(forbidden_modules))
            write_json(
                strict_job / "STRICT_INFORMATION_GATE.json",
                {
                    "passed": True,
                    "forbidden_modules_loaded": [],
                    "probe_phases_executed": [],
                    "physical_probe_actions_executed": 0,
                    "belief_loaded": False,
                    "feasibility_loaded": False,
                    "selected_force_N": fixed_force,
                    "force_source": "TRAIN_ONLY_PER_TASK_FIXED_FREEZE",
                },
            )

            env.step = original_step
            prepared = runtime.first_chunk(env, plan, strict_job, int(args.port), reference)
            write_json(
                strict_job / "INITIAL_ONLINE_CHUNK_IDENTITY.json",
                {
                    "observation_sha256": prepared[4]["observation_sha256"],
                    "noise_seed": prepared[4]["noise_seed"],
                    "noise_sha256": prepared[4]["noise_sha256"],
                    "action_sha256": prepared[4]["action_sha256"],
                },
            )
            handoff = command_from_probe(
                records[-1]["action"],
                float(saved["objects"]["arm_action"]["_gripper_abs_cmd"][0, 0]),
                p4.D_CLOSED,
                p4.D_OPEN,
            )
            write_json(strict_job / "PLANNER_DECISION.json", {
                "method": "STRICT_NOQUERY_PER_TASK_FIXED",
                "selected_force_N": fixed_force,
                "probe_executed": False,
                "belief_loaded": False,
                "feasibility_loaded": False,
                "candidate_actions_executed_before_decision": 0,
            })
            write_json(strict_job / "HANDOFF.json", {
                "command": handoff,
                "source": "last established-grasp hold action",
                "physical_probe_executed": False,
            })
            strict_result = runtime.rollout(env, p4, p5, plan, strict_job, fixed_force, handoff, prepared)

            restore(env, saved)
            restored = capture(env)
            equality = {key: identical(saved[key], restored[key]) for key in SNAPSHOT_GROUPS}
            torch.save(restored, reference / "LIVE_AFTER_RESTORE.pt")
            write_json(reference / "PREPROBE_RESTORE_GATE.json", {
                "passed": all(equality.values()),
                "checks": equality,
                "shared_snapshot_sha256": sha(reference / "DECISION_STATE.pt"),
                "live_after_restore_sha256": sha(reference / "LIVE_AFTER_RESTORE.pt"),
                "cache_fields_ignored": False,
                "rng_fields_ignored": False,
                "known_limit": "unexposed PhysX contact/solver warm-start state",
            })
            if not all(equality.values()):
                raise RuntimeError("preprobe restore equality failed: " + repr(equality))
            env.step = recording_step

        rows, probe_record = p4.run_probe_episode(
            env,
            seed_idx=int(plan["root"]),
            mu=float(plan["mu"]),
            trial_id=plan["id"],
            dt=0.05,
            termination_signal=FrictionProbeBudget(records, p4._quat_apply_np),
            preprobe_callback=preprobe_callback,
        )
        if strict_result is None:
            raise RuntimeError("preprobe callback did not execute")

        raw = [vars(row) for row in rows]
        write_csv(af_job / "RAW_PROBE.csv", raw)
        write_json(af_job / "CONTACT_PATCH_READBACK.json", records)
        reasons = [
            key
            for key in ("probe_failure", "contact_lost_probe", "dropped", "major_disturbance")
            if probe_record.get(key, 0)
        ]
        if not any(row["probe_phase"] == "probe_out" for row in raw):
            reasons.append("NO_PROBE_OUT_ACTION")
        if reasons:
            raise RuntimeError("AF probe not qualified: " + repr(reasons))

        af_saved = capture(env)
        af_saved.update(candidate_actions_already_executed=0, observation_step=len(raw), context=plan)
        torch.save(af_saved, af_job / "DECISION_STATE.pt")
        write_json(af_job / "PROBE_RESULT.json", {
            "probe_qualified": True,
            "steps": len(raw),
            "outward_steps": sum(row["probe_phase"] == "probe_out" for row in raw),
            "record": probe_record,
        })
        runtime.save_reference_observation(env, plan, af_job)

        # These imports occur only after strict execution and restoration.
        import continuous_belief
        from phase_free_feasibility import PhaseFreeFeasibility

        official_worker = load_module("paired_official_worker", FINAL / "worker.py")
        belief = continuous_belief.ContinuousBelief(
            V5 / "BELIEF_MANIFEST.json", manifest_sha256=sha(V5 / "BELIEF_MANIFEST.json")
        )
        posterior_full = belief.rows(
            raw,
            records,
            runtime_manifest_sha256=sha(V5 / "CURRENT_ACTIVEFORCING_RUNTIME_MANIFEST.json"),
        )
        posterior = {
            key: posterior_full[key]
            for key in (
                "interface",
                "feature_schema_id",
                "member_means",
                "member_log_sigmas",
                "posterior_moments",
                "integration_qa",
            )
        }
        posterior.update(
            integration_nodes=posterior_full["integration_nodes"].tolist(),
            integration_weights=posterior_full["integration_weights"].tolist(),
            candidate_actions_executed=0,
            hidden_friction_used=False,
        )
        write_json(af_job / "PREACTION_POSTERIOR.json", posterior)
        prepared = runtime.first_chunk(env, plan, af_job, int(args.port), af_job)
        write_json(af_job / "INITIAL_ONLINE_CHUNK_IDENTITY.json", {
            "observation_sha256": prepared[4]["observation_sha256"],
            "noise_seed": prepared[4]["noise_seed"],
            "noise_sha256": prepared[4]["noise_sha256"],
            "action_sha256": prepared[4]["action_sha256"],
        })
        sequence = official_worker.make_sequence(
            raw,
            af_saved,
            np.asarray(records[-1]["eef_pose"], np.float32).reshape(-1),
            p4,
            int(plan["task"]),
            prepared[3],
        )
        with (af_job / "PREACTION_SEQUENCE.npy").open("xb") as handle:
            np.save(handle, sequence)
        decision = PhaseFreeFeasibility().select(sequence, posterior)
        af_force = float(decision["selected_force_N"])
        decision.update(
            method="ACTIVEFORCING",
            executed_force_N=af_force,
            probe_executed=True,
            belief_loaded=True,
            feasibility_loaded=True,
        )
        write_json(af_job / "PLANNER_DECISION.json", decision)
        af_handoff = command_from_probe(
            records[-1]["action"],
            float(af_saved["objects"]["arm_action"]["_gripper_abs_cmd"][0, 0]),
            p4.D_CLOSED,
            p4.D_OPEN,
        )
        write_json(af_job / "HANDOFF.json", {"command": af_handoff, "source": "last physical probe action"})
        env.step = original_step
        af_result = runtime.rollout(env, p4, p5, plan, af_job, af_force, af_handoff, prepared)

        strict_initial = read(strict_job / "INITIAL_ONLINE_CHUNK_IDENTITY.json")
        af_initial = read(af_job / "INITIAL_ONLINE_CHUNK_IDENTITY.json")
        same_seed = (
            strict_initial["noise_seed"] == af_initial["noise_seed"]
            and strict_initial["noise_sha256"] == af_initial["noise_sha256"]
        )
        write_json(pair / "PAIR_RESULT.json", {
            "context": plan,
            "strict_force_N": fixed_force,
            "af_force_N": af_force,
            "same_preprobe_snapshot": read(reference / "PREPROBE_RESTORE_GATE.json")["passed"],
            "same_initial_vla_noise_seed_and_bytes": same_seed,
            "strict_execution_valid": bool(strict_result.get("online_vla_verified")) and bool(strict_result["outcome"].get("label_valid")),
            "af_execution_valid": bool(af_result.get("online_vla_verified")) and bool(af_result["outcome"].get("label_valid")),
            "strict_success": strict_result["outcome"].get("full_task_success_y"),
            "af_success": af_result["outcome"].get("full_task_success_y"),
            "strict_rpc_count": strict_result.get("rpc_count"),
            "af_rpc_count": af_result.get("rpc_count"),
            "known_snapshot_limit": "unexposed PhysX contact/solver warm-start state",
        })
        if not same_seed:
            raise RuntimeError("paired VLA noise seed/bytes differ")
        if not (
            strict_result.get("online_vla_verified")
            and strict_result["outcome"].get("label_valid")
            and af_result.get("online_vla_verified")
            and af_result["outcome"].get("label_valid")
        ):
            raise RuntimeError("one or both paired executions are invalid")
        write_json(pair / "WORKER_COMPLETION.json", {"logical_success": True, "pair_result": str(pair / "PAIR_RESULT.json")})
        return 0
    except Exception as exc:
        write_json(pair / "WORKER_ERROR.json", {"error": repr(exc), "traceback": traceback.format_exc()})
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
    parser.add_argument("--force-freeze", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--port", type=int, default=18885)
    raise SystemExit(main(parser.parse_args()))
