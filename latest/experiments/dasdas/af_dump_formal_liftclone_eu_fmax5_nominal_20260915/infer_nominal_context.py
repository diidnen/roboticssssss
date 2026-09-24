"""Matched Nominal on the EU Fmax5 prefix: P4 keep-hold, dump shear, native π0 remainder."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
AF = Path("/media/volume/dasdas/exouser/af_dump_formal_liftclone_eu_fmax5_20260915")
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
OLD = BASE / "experiments/af_dump_original_restore_20260912"
REPO = BASE / "RoboTwin"
sys.path[:0] = [str(HERE), str(OLD), str(REPO)]

import qualify_native_interfaces as base
from admit_original_query import admit
from establish_grasp import GRASP_FORCE_N, QUERY_DISPLACEMENT_M, QUERY_FORCE_N, scripted_establish_grasp
from native_visual_mirror import capture_visual_packet, observation_identity
from original_squeeze_inner import OriginalSqueezeInner
from qualify_original_p4_native import qualify as original_p4_query
from rootlocal_collection_contract import read, sha, verify_runtime, write

import infer_original_rootlocal as original

restore_initial_camera_cache = original.restore_initial_camera_cache
HARNESS_SENTINEL_N = 8.0

P4_FROZEN_NAMES = (
    "original_raw_rows.json",
    "patch_readbacks.json",
    "qualification.json",
    "original58_engineering.npy",
    "original_probe_summary.json",
)


def write_replace(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def formal_context_and_models():
    spec = read(os.environ["AF_FORMAL_CONTEXT"])
    context = spec["context"]
    models = Path(spec["models"])
    plan = Path(context["formal_protocol_path"]).parent
    lock = read(plan / "FREEZE_LOCK.json")
    checks = {
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
        "formal_contexts_sha256": sha(plan / "FORMAL_CONTEXTS.json"),
    }
    if checks != lock:
        raise ValueError("Frozen matched-Nominal plan changed")
    schedule = [read(plan / "SMOKE_CONTEXT.json"), *read(plan / "FORMAL_CONTEXTS.json")]
    if context not in schedule:
        raise ValueError("Unplanned matched-Nominal context")
    if context["root"] != 200002 or context["split"] not in ("SMOKE", "FORMAL_LIFTCLONE_EU_FMAX5_NOMINAL"):
        raise ValueError("Wrong matched-Nominal context scope")
    if context["methods"] != ["Nominal Frozen VLA"]:
        raise ValueError("This queue is Nominal-only")
    verify_runtime(context["runtime_manifest_path"], context["runtime_manifest_sha256"])
    completed = read(models / "TRAINING_COMPLETE.json")
    if not completed.get("completed") or completed.get("test_groups_executed") != 0:
        raise ValueError("Invalid frozen model state")
    if completed.get("executed_selector") != "expected_utility":
        raise ValueError("Paired AF deploy is not expected_utility")
    return spec, context, models


def _seal_decision(parent: Path, context: dict) -> None:
    if context["methods"] != ["Nominal Frozen VLA"]:
        raise ValueError("Nominal-only methods required")
    seal = read(parent / "PREACTION_SELECTION_LOCK.json")
    seal.update(
        branch_methods=["Nominal Frozen VLA"],
        forces_N=[HARNESS_SENTINEL_N],
        commanded_forces_N=[None],
        selected_force_raw_N=None,
        executed_selector="none",
        force_support=[0.25, 5.0],
        planner_grid_step=0.05,
        selector="none; native VLA remainder after matched EU Fmax5 prefix",
        inference_source_sha256=sha(Path(__file__)),
        formal_context_spec_sha256=sha(Path(os.environ["AF_FORMAL_CONTEXT"])),
        nominal_harness_force_sentinel_N=HARNESS_SENTINEL_N,
        nominal_harness_force_sentinel_used_for_branch_naming_only=True,
        feature_source="ONLINE_VLA_ACTION_CHUNK",
        protocol_note=(
            "matched Nominal: original P4, 12N force cap, dump shear, "
            "last-command handoff, native pi0 remainder; no EU force"
        ),
        established_grasp_N=GRASP_FORCE_N,
        query_force_N=QUERY_FORCE_N,
        force_clamped_to_0p5=False,
    )
    write_replace(parent / "PREACTION_SELECTION_LOCK.json", seal)
    os.environ["AF_ENGINEERING_FORCES"] = str(HARNESS_SENTINEL_N)


def _gripper_command_m(env, arm: str) -> float:
    scale = getattr(env.robot, arm + "_gripper_scale")
    if arm == "left":
        value = float(env.robot.get_left_gripper_val())
    else:
        value = float(env.robot.get_right_gripper_val())
    meters = float(scale[0] + float(np.clip(value, 0.0, 1.0)) * (scale[1] - scale[0]))
    return float(np.clip(meters, 0.0, 0.04))


def _attach_inner_from_last_command(env, arm: str, outer_command: float) -> tuple[object, list]:
    inner = OriginalSqueezeInner()
    env._af_original_squeeze_inner = inner
    measured = float(base.contacts(env, arm)["measured_squeeze_n"])
    if not np.isfinite(measured) or measured < 0:
        measured = 0.0
    actual = inner.step(measured, QUERY_FORCE_N, float(outer_command))
    scale = getattr(env.robot, arm + "_gripper_scale")
    env.robot.set_gripper((actual - scale[0]) / (scale[1] - scale[0]), arm, gripper_eps=0)
    env._af_original_outer_command = float(outer_command)
    return inner, [{"physics_step": 1, **inner.last}]


def _preserve_p4(out: Path) -> dict:
    missing = [name for name in P4_FROZEN_NAMES if not (out / name).exists()]
    if missing:
        raise ValueError("Original P4 artifacts missing: " + ", ".join(missing))
    hashes = {name: sha(out / name) for name in P4_FROZEN_NAMES}
    for name in ("native_controls.json", "original_squeeze_inner_trace.json"):
        source = out / name
        if not source.exists():
            raise ValueError("P4 did not write " + name)
        shutil.copy2(source, out / f"P4_{name}")
        hashes[f"P4_{name}"] = sha(out / f"P4_{name}")
    write(out / "P4_PRESERVED.json", {"frozen_p4_hashes": hashes, "not_overwritten": list(P4_FROZEN_NAMES)})
    return hashes


def _assert_p4_untouched(out: Path, hashes: dict) -> None:
    for name in P4_FROZEN_NAMES:
        if sha(out / name) != hashes[name]:
            raise ValueError("Frozen P4 belief evidence was overwritten: " + name)


def locked_query(env, out: Path) -> None:
    spec, context, models = formal_context_and_models()
    del models
    if env.af_contact_friction != context["friction"] or env._af_qualification_policy_seed != context["policy_seed"]:
        raise ValueError("Native setup differs from plan")
    if os.environ.get("AF_P4_PHYSICAL_SURFACE_CAMERA") != "1" or os.environ.get("AF_ORIGINAL_SQUEEZE_INNER") != "1":
        raise ValueError("Formal sensor and complete original squeeze loop required")
    original_p4_query(env, out)
    p4_hashes = _preserve_p4(out)
    import sapien

    body = env.deskbin.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    write(
        out / "ACTUAL_OBJECT_MATERIAL.json",
        {
            "shape_materials": [
                [float(s.physical_material.static_friction), float(s.physical_material.dynamic_friction)]
                for s in body.collision_shapes
            ]
        },
    )
    write(out / "ONLINE_QUERY_ADMISSION.json", admit(out, require_material=True))
    _assert_p4_untouched(out, p4_hashes)
    env.af_force_limit_n = GRASP_FORCE_N
    env.activate_activeforcing_candidate_force()
    p4_raw = read(out / "original_raw_rows.json")
    arm = scripted_establish_grasp(env, p4_raw=p4_raw)
    env._af_grasp_arm_tag = arm
    query = env.run_activeforcing_query(query_force_n=QUERY_FORCE_N, displacement_m=QUERY_DISPLACEMENT_M)
    if not query.get("final_bilateral_contact"):
        raise RuntimeError("Dump shear query lost contact after established grasp")
    write(out / "DUMP_SHEAR_QUERY.json", {k: v for k, v in query.items() if k != "trace"})
    write(out / "DUMP_SHEAR_QUERY_TRACE.json", query.get("trace") or [])
    grasp_report = getattr(env, "_af_established_grasp", {})
    write(
        out / "ESTABLISHED_GRASP.json",
        {
            "arm": arm,
            "grasp_force_N": GRASP_FORCE_N,
            "query_force_N": QUERY_FORCE_N,
            "query_displacement_m": QUERY_DISPLACEMENT_M,
            "take_action_cnt": int(env.take_action_cnt),
            "source": "original P4 then 12N force cap then dump shear; no re-grasp if P4 still holding",
            "skipped_script": bool(grasp_report.get("skipped_script")),
            "skip_reason": grasp_report.get("reason"),
            "p4_row_holding": grasp_report.get("p4_row_holding"),
            "live_hold": grasp_report.get("live"),
        },
    )
    outer = _gripper_command_m(env, arm)
    _inner, inner_trace = _attach_inner_from_last_command(env, arm, outer)
    write_replace(out / "original_squeeze_inner_trace.json", inner_trace)
    action13 = [0.0] * 13
    action13[6] = float(outer)
    write_replace(
        out / "native_controls.json",
        [
            {
                "step": 1,
                "original_action13": action13,
                "source": "last gripper command after dump shear query",
            }
        ],
    )
    _assert_p4_untouched(out, p4_hashes)
    initial = base.readback(env)
    initial_count = env.take_action_cnt
    from scripts.eval_policy_xpolicylab import (
        build_policy_client,
        close_policy_client,
        normalize_action_chunk,
        robotwin_obs_to_xpolicylab,
        xpolicylab_action_to_robotwin,
    )

    client = build_policy_client(
        {
            "protocol": "ws",
            "host": "localhost",
            "port": 6001,
            "evaluation_id": "liftclone-eu-fmax5-nominal-preaction-lock",
            "trial_id": context["id"],
        }
    )
    try:
        client.call(
            func_name="prepare_case",
            obs={
                "task_name": "dump_bin_bigbin",
                "seed": 200002,
                "policy_seed": context["policy_seed"],
                "instruction": env.get_instruction(),
                "action_type": "joint",
            },
        )
        client.call(func_name="reset")
        observation = env.get_obs()
        payload = robotwin_obs_to_xpolicylab(
            observation, instruction=env.get_instruction(), env_idx=0, frequency=30, task_env=env
        )
        client.call(func_name="update_obs", obs=payload)
        response = normalize_action_chunk(client.call(func_name="get_action"))
        actions = []
        for action in response:
            flat, kind = xpolicylab_action_to_robotwin(
                action, action_type="joint", current_observation=observation
            )
            if kind != "qpos":
                raise ValueError("Unexpected policy action type")
            actions.append(flat)
        actions = np.asarray(actions, np.float32)
    finally:
        close_policy_client(client)
    if base.readback(env) != initial or env.take_action_cnt != initial_count:
        raise ValueError("Decision construction changed physical state/action count")
    _assert_p4_untouched(out, p4_hashes)
    parent = out.parent
    np.save(parent / "PREACTION_PI0_CHUNK.npy", actions)
    cameras = {name: np.asarray(row["rgb"]).copy() for name, row in observation["observation"].items() if "rgb" in row}
    if not cameras:
        raise ValueError("Missing actual initial policy camera frames")
    np.savez_compressed(parent / "COMMON_POSTPROBE_CAMERA_CACHE.npz", **cameras)
    env._af_locked_camera_cache = cameras
    env._af_locked_native_action_count = initial_count
    env._af_locked_observation_identity = observation_identity(observation)
    env._af_locked_state_sha256 = base.digest(initial)
    seal = {
        "context_id": context["id"],
        "root": 200002,
        "candidate_actions_executed": 0,
        "labels_read": False,
        "state_sha256": base.digest(initial),
        "first_chunk_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "observation_identity": observation_identity(observation),
        "forces_N": [HARNESS_SENTINEL_N],
        "branch_methods": ["Nominal Frozen VLA"],
        "artifact_hashes": {
            name: sha(parent / name)
            for name in [
                "PREACTION_PI0_CHUNK.npy",
                "COMMON_POSTPROBE_CAMERA_CACHE.npz",
            ]
        },
        "context": context,
        "inference_source_sha256": sha(Path(__file__)),
        "handoff_command_m": float(outer),
        "handoff_source": "last gripper command after dump shear query",
        "belief_source": "original P4 rows preserved; force not selected from posterior",
        "feature_source": "ONLINE_VLA_ACTION_CHUNK",
        "p4_preserved": read(out / "P4_PRESERVED.json"),
        "matched_af_experiment": str(AF),
        "matched_af_context_id": context.get("matched_af_context_id"),
    }
    write(parent / "PREACTION_SELECTION_LOCK.json", seal)
    env._af_preacton_selection_lock = seal
    _seal_decision(parent, context)
    env._af_preacton_selection_lock = read(parent / "PREACTION_SELECTION_LOCK.json")


def nominal_child_run(env, pipe, out: Path, force, handoff, stock_drives, support, scope) -> None:
    del force, handoff, support
    env._update_render = lambda: None
    env.eval_video_path = None
    env.render_freq = 0
    env._af_original_squeeze_inner = None
    for joint, stiffness, damping, limit, mode in stock_drives:
        joint.set_drive_properties(stiffness, damping, limit, mode)
    original_step = env.scene.step
    trace_hash = hashlib.sha256()
    physics_steps = 0
    forces = []
    chunks = 0
    receipts = []

    def rpc(kind):
        pipe.send({"kind": kind, "packet": capture_visual_packet(env)})
        response = pipe.recv()
        if response["kind"] == "error":
            raise RuntimeError(response["error"])
        return response

    def get_obs(*args, **kwargs):
        del args, kwargs
        observation = rpc("observation")["observation"]
        env.now_obs = observation
        return observation

    env.get_obs = get_obs
    try:
        with gzip.open(out / "physics_trace.jsonl.gz", "wt") as trace:

            def step():
                nonlocal physics_steps
                original_step()
                contact = base.contacts(env, env._af_grasp_arm_tag)
                measured = float(contact["measured_squeeze_n"])
                forces.append(measured)
                physics_steps += 1
                row = {
                    "physics_step": physics_steps,
                    "contact": contact,
                    "state": base.readback(env),
                    "original_squeeze_inner": None,
                    "native_nominal_gripper": True,
                }
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":"))
                trace_hash.update(encoded.encode())
                trace.write(encoded + "\n")

            env.scene.step = step
            while not env.eval_success and env.take_action_cnt < env.step_lim:
                actions = np.asarray(rpc("chunk")["actions"], np.float32)
                if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
                    raise RuntimeError("Invalid native nominal policy chunk")
                np.save(out / f"chunk_{chunks:03d}.npy", actions)
                chunks += 1
                for action in actions:
                    offset = 0 if env._af_grasp_arm_tag == "left" else 7
                    receipts.append(
                        {
                            "native_action14": action.tolist(),
                            "raw_vla_arm_command": action[offset : offset + 6].tolist(),
                            "final_arm_command": action[offset : offset + 6].tolist(),
                            "native_vla_gripper_normalized": float(action[offset + 6]),
                            "force_controller_installed": False,
                            "gripper_force_override": False,
                            "commanded_force_N": None,
                        }
                    )
                    env.take_action(action.copy(), action_type="qpos")
                    if env.eval_success or env.take_action_cnt >= env.step_lim:
                        break
        result = {
            "completed": True,
            "method": "Nominal Frozen VLA",
            "success": bool(env.eval_success),
            "official_final_check": bool(env.check_success()),
            "native_actions": env.take_action_cnt,
            "native_horizon": env.step_lim,
            "physics_steps": physics_steps,
            "trace_sha256": trace_hash.hexdigest(),
            "force_setpoint_bilateral_n": None,
            "commanded_force_N": None,
            "measured_mean_squeeze_n": float(np.mean(forces)),
            "measured_max_squeeze_n": float(np.max(forces)),
            "chunks": chunks,
            "native_gripper_actions": len(receipts),
            "native_gripper_min": min(row["native_vla_gripper_normalized"] for row in receipts),
            "native_gripper_max": max(row["native_vla_gripper_normalized"] for row in receipts),
            "force_controller_installed": False,
            "gripper_force_override": False,
            "original_squeeze_inner_disabled_after_query": True,
            "old_irrecoverable_shortcut_used": False,
            "scope": scope,
        }
        write_replace(out / "arbitration.json", receipts)
        write_replace(out / "result.json", result)
        pipe.send({"kind": "done", "result": result})
    finally:
        env.scene.step = original_step


def bound_execution_module():
    source = OLD / "qualify_original_online_forks.py"
    baseline_text = source.read_text()
    tree = ast.parse(baseline_text)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qualify")
    edits = {"max_branches": 0, "scope": 0, "handoff_units": 0}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "max_branches":
                if ast.unparse(node.value) != "8 if collection is not None else 3":
                    raise RuntimeError("Harness branch gate drift")
                node.value = ast.Constant(1)
                edits["max_branches"] += 1
            elif node.targets[0].id == "scope":
                node.value = ast.Constant("formal Nominal matched EU Fmax5 prefix native remainder")
                edits["scope"] += 1
        if isinstance(node, ast.Compare) and "inner.last" in ast.unparse(node) and "drive_target" in ast.unparse(node):
            compact = ast.unparse(node).replace(" ", "")
            if compact not in (
                "abs(float(joints[0][0].drive_target[0])-inner.last['inner_aperture_m'])>1e-7",
                "abs(float(joints[0][0].drive_target[0])-inner.last['inner_aperture_m'])>1e-07",
            ):
                raise RuntimeError("Handoff unit check drift: " + ast.unparse(node))
            node.left = ast.parse(
                "abs((float(joints[0][0].drive_target[0]) - float(joints[0][2])) / float(joints[0][1]) "
                "- inner.last['inner_aperture_m'])",
                mode="eval",
            ).body
            node.comparators = [ast.Constant(1e-6)]
            edits["handoff_units"] += 1
    if edits != {"max_branches": 1, "scope": 1, "handoff_units": 1}:
        raise RuntimeError(f"Unexpected matched-Nominal harness edits: {edits}")
    if ast.dump(ast.parse(baseline_text)) == ast.dump(tree):
        raise RuntimeError("Harness edits did not change AST")
    module = ModuleType("_formal_nominal_eu_fmax5_prefix_harness")
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    module.child_run = nominal_child_run
    module.inference_binding_receipt = {
        "source_sha256": sha(source),
        "original_AST_recovered_exactly": True,
        "only_harness_changes": [
            "maximum one Nominal branch",
            "matched EU Fmax5 prefix Nominal descriptive scope",
            "handoff compares joint drive mapped to metres",
            "native nominal child_run",
        ],
        "force_branch_control_or_physics_changed": False,
        "probe": "original P4 for prefix; dump shear after 12N cap for handoff; remainder is native pi0",
    }
    return module


def qualify(env, out: Path) -> None:
    env._af_grasp_arm_tag = "left"
    module = bound_execution_module()
    module.qualify_query = locked_query
    live_renderer = module.native_observation_from_packet
    module.native_observation_from_packet = lambda current_env, packet: restore_initial_camera_cache(
        current_env, packet, live_renderer
    )
    module.qualify(env, out)
    _, context, _ = formal_context_and_models()
    seal = read(out / "PREACTION_SELECTION_LOCK.json")
    summary = read(out / "online_qualification.json")
    if len(summary["results"]) != 1 or summary["results"][0]["kind"] != "done":
        raise ValueError(f"Incomplete matched-Nominal context: {summary.get('results')}")
    if summary["first_chunk_hashes"] != [seal["first_chunk_sha256"]]:
        raise ValueError("First action chunk pairing failed")
    if summary["common_handoff_state_sha256"] != seal["state_sha256"]:
        raise ValueError("Common handoff state changed")
    branch = next(out.glob("branch_0_*"))
    result = read(branch / "result.json")
    if result["success"] != result["official_final_check"]:
        raise ValueError("Official success mismatch")
    if result["force_setpoint_bilateral_n"] is not None or result["commanded_force_N"] is not None:
        raise ValueError("Nominal branch used a commanded force")
    if result["force_controller_installed"] or result["gripper_force_override"]:
        raise ValueError("Nominal branch used a force override")
    if not result.get("original_squeeze_inner_disabled_after_query"):
        raise ValueError("Nominal remainder did not clear the squeeze inner")
    if seal.get("commanded_forces_N") != [None]:
        raise ValueError("Nominal selection lock has a commanded force")
    outcomes = [
        {
            "method": "Nominal Frozen VLA",
            "commanded_force_N": None,
            "success": int(result["success"]),
            "measured_mean_squeeze_N": result["measured_mean_squeeze_n"],
            "measured_max_squeeze_N": result["measured_max_squeeze_n"],
            "native_actions": result["native_actions"],
            "result_sha256": sha(branch / "result.json"),
        }
    ]
    write_replace(
        out / "FORMAL_CONTEXT_RESULT.json",
        {
            "completed": True,
            "context": context,
            "outcomes": outcomes,
            "paired_rollouts": 1,
            "selection_lock_sha256": sha(out / "PREACTION_SELECTION_LOCK.json"),
            "first_chunk_sha256": seal["first_chunk_sha256"],
            "common_handoff_state_sha256": seal["state_sha256"],
            "executed_selector": "none",
            "harness_binding": module.inference_binding_receipt,
            "claim_boundary": (
                "same-root matched Nominal on EU Fmax5 prefix; original P4, "
                "12N cap, dump shear, native pi0 remainder; old pre-grasp Nominal is not it"
            ),
        },
    )


if __name__ == "__main__":
    os.environ["AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP"] = "1"
    if "AF_FORMAL_CONTEXT" not in os.environ:
        raise RuntimeError("AF_FORMAL_CONTEXT is required")
    original.context_and_models = formal_context_and_models
    base.qualify = qualify
    base.main()
