"""One paired fixed-force context from a common qualified P4 handoff."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
OLD = HERE.parent / "af_dump_original_restore_20260912"
sys.path.insert(1, str(OLD))

import qualify_native_interfaces as base
from admit_original_query import admit
from native_cartesian_kinematics import NativeCartesianKinematics
from native_original_motion_features import build_features
from native_visual_mirror import observation_identity
from qualify_original_p4_native import qualify as original_query
from rootlocal_collection_contract import read, sha, verify_runtime


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def context_spec() -> tuple[dict, dict]:
    spec_path = Path(os.environ["AF_FIXED_SWEEP_CONTEXT"])
    spec = read(spec_path)
    context = spec["context"]
    plan = Path(context["protocol_path"]).parent
    lock = read(plan / "FREEZE_LOCK.json")
    checks = {
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
        "main_contexts_sha256": sha(plan / "MAIN_CONTEXTS.json"),
    }
    if checks != lock:
        raise ValueError("Frozen fixed-force plan changed")
    schedule = [read(plan / "SMOKE_CONTEXT.json"), *read(plan / "MAIN_CONTEXTS.json")]
    if context not in schedule:
        raise ValueError("Unplanned fixed-force context")
    if context["root"] != 200002 or context["split"] not in ("ENGINEERING_SMOKE", "MAIN_POSTHOC_RANGE"):
        raise ValueError("Wrong fixed-force scope")
    if sorted(context["forces_N"]) not in ([8.0, 15.0], [5.0, 8.0, 10.0, 12.0, 15.0]):
        raise ValueError("Unplanned fixed-force set")
    verify_runtime(context["runtime_manifest_path"], context["runtime_manifest_sha256"])
    if float(context["friction"]) != float(spec["friction"]):
        raise ValueError("SPEC friction mismatch")
    return spec, context


def locked_query(env, out: Path) -> None:
    _, context = context_spec()
    if float(env.af_contact_friction) != float(context["friction"]):
        raise ValueError("Runtime friction differs from frozen context")
    if int(env._af_qualification_policy_seed) != int(context["policy_seed"]):
        raise ValueError("Runtime policy seed differs from frozen context")
    if os.environ.get("AF_P4_PHYSICAL_SURFACE_CAMERA") != "1" or os.environ.get("AF_ORIGINAL_SQUEEZE_INNER") != "1":
        raise ValueError("Physical surface camera and full original squeeze loop are required")
    original_query(env, out)
    import sapien

    body = env.deskbin.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    write_json(
        out / "ACTUAL_OBJECT_MATERIAL.json",
        {
            "shape_materials": [
                [float(shape.physical_material.static_friction), float(shape.physical_material.dynamic_friction)]
                for shape in body.collision_shapes
            ]
        },
    )
    write_json(out / "ONLINE_QUERY_ADMISSION.json", admit(out, require_material=True))

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
            "evaluation_id": "fixed-force-5-15-preaction-lock",
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
    if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Invalid frozen first action chunk")
    if base.readback(env) != initial or env.take_action_cnt != initial_count:
        raise ValueError("Pre-action lock changed physical state/action count")

    raw = read(out / "original_raw_rows.json")
    patch = read(out / "patch_readbacks.json")
    kin = NativeCartesianKinematics(env, env._af_grasp_arm_tag)
    feature = build_features(
        raw,
        patch[-1]["finger_joints"],
        np.asarray(body.linear_velocity),
        kin.future_xyz_base(actions),
    )
    parent = out.parent
    write_json(parent / "PREACTION_FEATURE.json", feature)
    np.save(parent / "PREACTION_PI0_CHUNK.npy", actions)
    cameras = {
        name: np.asarray(row["rgb"]).copy()
        for name, row in observation["observation"].items()
        if "rgb" in row
    }
    if not cameras:
        raise ValueError("Missing actual initial policy camera frames")
    np.savez_compressed(parent / "COMMON_POSTPROBE_CAMERA_CACHE.npz", **cameras)
    env._af_locked_camera_cache = cameras
    env._af_locked_native_action_count = initial_count
    env._af_locked_observation_identity = observation_identity(observation)
    env._af_locked_state_sha256 = base.digest(initial)
    forces = [float(value) for value in context["forces_N"]]
    seal = {
        "context_id": context["id"],
        "root": 200002,
        "candidate_actions_executed": 0,
        "labels_read": False,
        "state_sha256": base.digest(initial),
        "first_chunk_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "observation_identity": observation_identity(observation),
        "forces_N": forces,
        "branch_methods": [f"Fixed-{value:g}N" for value in forces],
        "force_support_N": [0.5, 15.0],
        "artifact_hashes": {
            name: sha(parent / name)
            for name in [
                "PREACTION_FEATURE.json",
                "PREACTION_PI0_CHUNK.npy",
                "COMMON_POSTPROBE_CAMERA_CACHE.npz",
            ]
        },
        "context": context,
        "inference_source_sha256": sha(__file__),
    }
    write_json(parent / "PREACTION_SELECTION_LOCK.json", seal)
    env._af_preacton_selection_lock = seal
    os.environ["AF_ENGINEERING_FORCES"] = ",".join(str(value) for value in forces)


def restore_initial_camera_cache(env, packet, live_renderer):
    observation = live_renderer(env, packet)
    if packet["take_action_cnt"] != env._af_locked_native_action_count:
        return observation
    if base.digest(packet["state"]) != env._af_locked_state_sha256:
        raise ValueError("Initial camera cache cannot be reused at a different physical state")
    if observation_identity(observation)["proprio_sha256"] != env._af_locked_observation_identity["proprio_sha256"]:
        raise ValueError("Live nonimage inputs differ at the pre-action boundary")
    restored = deepcopy(observation)
    for name, array in env._af_locked_camera_cache.items():
        if np.asarray(restored["observation"][name]["rgb"]).shape != array.shape:
            raise ValueError("Native initial camera shape mismatch")
        restored["observation"][name]["rgb"] = array.copy()
    if observation_identity(restored) != env._af_locked_observation_identity:
        raise ValueError("Actual cached first-frame restoration mismatch")
    return restored


def bound_execution_module():
    source = OLD / "qualify_original_online_forks.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    baseline = ast.dump(tree)
    edits = []
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qualify")
    validation = "len(forces) > max_branches or len(forces) < 1 or (not all((0.5 <= x <= 8 for x in forces)))"
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "max_branches":
                if ast.unparse(node.value) != "8 if collection is not None else 3":
                    raise RuntimeError("Harness branch gate drift")
                edits.append((node, "value", node.value))
                node.value = ast.Constant(5)
            elif name == "support":
                edits.append((node, "value", node.value))
                node.value = ast.parse("(0.5, 15.0)", mode="eval").body
            elif name == "scope":
                edits.append((node, "value", node.value))
                node.value = ast.Constant("posthoc fixed-force 5--15 N same-root range diagnostic")
        elif isinstance(node, ast.If) and ast.unparse(node.test) == validation:
            edits.append((node, "test", node.test))
            node.test = ast.parse(
                "len(forces) > max_branches or len(forces) < 1 or not all(0.5 <= x <= 15 for x in forces)",
                mode="eval",
            ).body
    if len(edits) != 4:
        raise RuntimeError(f"Unexpected harness binding count: {len(edits)}")
    executable = deepcopy(tree)
    for node, attribute, previous in edits:
        setattr(node, attribute, previous)
    if ast.dump(tree) != baseline:
        raise RuntimeError("Undeclared original harness change")
    module = ModuleType("_fixed_force_5_15_harness")
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(executable), str(source), "exec"), module.__dict__)
    module.inference_binding_receipt = {
        "source_sha256": sha(source),
        "original_AST_recovered_exactly": True,
        "only_harness_changes": [
            "maximum five paired branches",
            "accepted force gate upper bound 8 to 15 N",
            "declared support [0.5,15] N",
            "descriptive posthoc scope",
        ],
        "force_controller_equations_changed": False,
        "physics_changed": False,
    }
    return module


def qualify(env, out: Path) -> None:
    module = bound_execution_module()
    module.qualify_query = locked_query
    live_renderer = module.native_observation_from_packet
    module.native_observation_from_packet = lambda current_env, packet: restore_initial_camera_cache(
        current_env, packet, live_renderer
    )
    module.qualify(env, out)
    _, context = context_spec()
    seal = read(out / "PREACTION_SELECTION_LOCK.json")
    summary = read(out / "online_qualification.json")
    forces = [float(value) for value in context["forces_N"]]
    if len(summary["results"]) != len(forces) or any(row["kind"] != "done" for row in summary["results"]):
        raise ValueError("Incomplete fixed-force paired context")
    if summary["first_chunk_hashes"] != [seal["first_chunk_sha256"]] * len(forces):
        raise ValueError("First action chunk pairing failed")
    if summary["common_handoff_state_sha256"] != seal["state_sha256"]:
        raise ValueError("Common handoff state changed")
    for name, digest in seal["artifact_hashes"].items():
        if sha(out / name) != digest:
            raise ValueError("Pre-action lock artifact changed")
    outcomes = []
    for index, force in enumerate(forces):
        branch = out / f"branch_{index}_{force:g}N"
        result = read(branch / "result.json")
        if float(result["force_setpoint_bilateral_n"]) != force:
            raise ValueError("Fixed-force branch setpoint mismatch")
        if result["success"] != result["official_final_check"]:
            raise ValueError("Official success mismatch")
        if read(branch / "original_motion_feature.json") != read(out / "PREACTION_FEATURE.json"):
            raise ValueError("Candidate motion feature differs across branches")
        outcomes.append(
            {
                "method": f"Fixed-{force:g}N",
                "commanded_force_N": force,
                "success": int(result["success"]),
                "measured_mean_squeeze_N": result["measured_mean_squeeze_n"],
                "measured_max_squeeze_N": result["measured_max_squeeze_n"],
                "native_actions": result["native_actions"],
                "result_sha256": sha(branch / "result.json"),
            }
        )
    write_json(
        out / "FIXED_SWEEP_CONTEXT_RESULT.json",
        {
            "completed": True,
            "context": context,
            "outcomes": outcomes,
            "paired_rollouts": len(forces),
            "selection_lock_sha256": sha(out / "PREACTION_SELECTION_LOCK.json"),
            "first_chunk_sha256": seal["first_chunk_sha256"],
            "common_handoff_state_sha256": seal["state_sha256"],
            "harness_binding": module.inference_binding_receipt,
            "claim_boundary": "posthoc same-root physical force-range diagnostic; no feasibility extrapolation claim",
        },
    )


if __name__ == "__main__":
    os.environ["AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP"] = "1"
    if "AF_FIXED_SWEEP_CONTEXT" not in os.environ:
        raise RuntimeError("AF_FIXED_SWEEP_CONTEXT is required")
    from rim20_formal_binding import install

    install()
    base.qualify = qualify
    base.main()
