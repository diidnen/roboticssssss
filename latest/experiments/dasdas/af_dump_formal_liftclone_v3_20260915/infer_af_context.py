"""Lift-clone dump online: established grasp → dump probe → one force → π0."""
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
import torch

HERE = Path(__file__).resolve().parent
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
OLD = BASE / "experiments/af_dump_original_restore_20260912"
REPO = BASE / "RoboTwin"
sys.path[:0] = [str(HERE), str(OLD), str(REPO)]

import qualify_native_interfaces as base
from deploy_prior import deploy_prior_posterior
from establish_grasp import GRASP_FORCE_N, QUERY_DISPLACEMENT_M, QUERY_FORCE_N, scripted_establish_grasp
from hold_chunk import build_hold_feature
from liftstyle_runtime import LiftstyleFeasibility
from native_visual_mirror import observation_identity
from original_squeeze_inner import OriginalSqueezeInner
from rootlocal_collection_contract import read, sha, verify_runtime, write

import infer_original_rootlocal as original

restore_initial_camera_cache = original.restore_initial_camera_cache


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
        "reused_nominal_sha256": sha(plan / "REUSED_NOMINAL_OUTCOMES.json"),
    }
    if checks != lock:
        raise ValueError("Frozen lift-clone v3 formal plan changed")
    schedule = [read(plan / "SMOKE_CONTEXT.json"), *read(plan / "FORMAL_CONTEXTS.json")]
    if context not in schedule:
        raise ValueError("Unplanned lift-clone v3 formal context")
    if context["root"] != 200002 or context["split"] not in ("SMOKE", "FORMAL_LIFTCLONE_V3"):
        raise ValueError("Wrong lift-clone v3 context scope")
    if context["methods"] != ["ActiveForcing"]:
        raise ValueError("This queue is AF-only")
    verify_runtime(context["runtime_manifest_path"], context["runtime_manifest_sha256"])
    completed = read(models / "TRAINING_COMPLETE.json")
    if not completed.get("completed") or completed.get("test_groups_executed") != 0:
        raise ValueError("Invalid frozen model state")
    for part, key in [("belief", "belief_manifest_sha256"), ("feasibility", "feasibility_manifest_sha256")]:
        manifest = models / part / ("BELIEF_MANIFEST.json" if part == "belief" else "FEASIBILITY_MANIFEST.json")
        if sha(manifest) != completed[key] or sha(manifest) != spec[key]:
            raise ValueError("Deployment checkpoint manifest changed")
        if len(read(models / part / "CHECKPOINT_SELECTION_LOCK.json")["checkpoints"]) != 3:
            raise ValueError("Missing three-member checkpoint lock")
    return spec, context, models


def _seal_decision(parent: Path, context: dict, decision: dict) -> None:
    selected = float(decision["selected_force_N"])
    if not (0.25 - 1e-9 <= selected <= 8.0 + 1e-9):
        raise ValueError(f"Selected force outside v3 support: {selected}")
    executed = min(8.0, max(0.5, selected))
    if context["methods"] != ["ActiveForcing"]:
        raise ValueError("AF-only methods required")
    seal = read(parent / "PREACTION_SELECTION_LOCK.json")
    seal.update(
        branch_methods=["ActiveForcing"],
        forces_N=[executed],
        commanded_forces_N=[executed],
        selected_force_raw_N=selected,
        force_support=[0.25, 8.0],
        planner_grid_step=0.25,
        selector="LiftstyleFeasibility v3 argmax_p on [0.25,8]@0.25",
        inference_source_sha256=sha(Path(__file__)),
        formal_context_spec_sha256=sha(Path(os.environ["AF_FORMAL_CONTEXT"])),
        decision_executed_selector=decision.get("executed_selector"),
        decision_predicted_success=decision.get("predicted_success"),
        decision_eu_counterfactual_selected_N=decision.get("eu_counterfactual_selected_N"),
        feature_source=decision.get("motion_channels"),
        protocol_note="lift clone: 12N established grasp, dump shear probe, last-command handoff, pi0 remainder",
        established_grasp_N=GRASP_FORCE_N,
        query_force_N=QUERY_FORCE_N,
    )
    write_replace(parent / "PREACTION_SELECTION_LOCK.json", seal)
    os.environ["AF_ENGINEERING_FORCES"] = str(executed)


def _gripper_command_m(env, arm: str) -> float:
    """Last gripper command in metres, matching original P4 action slot 6."""
    scale = getattr(env.robot, arm + "_gripper_scale")
    if arm == "left":
        value = float(env.robot.get_left_gripper_val())
    else:
        value = float(env.robot.get_right_gripper_val())
    meters = float(scale[0] + float(np.clip(value, 0.0, 1.0)) * (scale[1] - scale[0]))
    meters = float(np.clip(meters, 0.0, 0.04))
    return meters


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


def locked_query(env, out: Path) -> None:
    spec, context, models = formal_context_and_models()
    if env.af_contact_friction != context["friction"] or env._af_qualification_policy_seed != context["policy_seed"]:
        raise ValueError("Native setup differs from plan")
    out.mkdir(parents=True, exist_ok=False)
    env.af_force_limit_n = GRASP_FORCE_N
    env.activate_activeforcing_candidate_force()
    arm = scripted_establish_grasp(env)
    query = env.run_activeforcing_query(query_force_n=QUERY_FORCE_N, displacement_m=QUERY_DISPLACEMENT_M)
    if not query.get("final_bilateral_contact"):
        raise RuntimeError("Dump shear query lost contact after established grasp")
    write(out / "DUMP_SHEAR_QUERY.json", {k: v for k, v in query.items() if k != "trace"})
    write(out / "DUMP_SHEAR_QUERY_TRACE.json", query.get("trace") or [])
    write(
        out / "ESTABLISHED_GRASP.json",
        {
            "arm": arm,
            "grasp_force_N": GRASP_FORCE_N,
            "query_force_N": QUERY_FORCE_N,
            "query_displacement_m": QUERY_DISPLACEMENT_M,
            "take_action_cnt": int(env.take_action_cnt),
            "source": "v3 scripted_establish_grasp then dump run_activeforcing_query",
        },
    )
    outer = _gripper_command_m(env, arm)
    _inner, inner_trace = _attach_inner_from_last_command(env, arm, outer)
    write(out / "original_squeeze_inner_trace.json", inner_trace)
    action13 = [0.0] * 13
    action13[6] = float(outer)
    write(
        out / "native_controls.json",
        [
            {
                "step": 1,
                "original_action13": action13,
                "source": "last gripper command after dump shear query; lift command_from_probe analog",
            }
        ],
    )
    opening = float(env.robot.get_left_gripper_val())
    contact = env.get_actor_gripper_contact_forces(env.deskbin, arm)
    force = float(contact.get("single_finger_normal_force_n") or 0.0)
    raw = [
        {
            "object_id": "deskbin",
            "gripper_opening": opening,
            "left_fx": 0.0,
            "left_fy": 0.0,
            "left_fz": force,
            "right_fx": 0.0,
            "right_fy": 0.0,
            "right_fz": force,
        }
    ]
    write(out / "original_raw_rows.json", raw)
    joints = [
        float(env.robot.get_left_gripper_val()),
        float(env.robot.get_right_gripper_val()),
    ]
    write(out / "patch_readbacks.json", [{"finger_joints": joints, "step": 1}])
    write(
        out / "qualification.json",
        {
            "completed": True,
            "probe_failure": False,
            "protocol": "lift_clone_established_grasp_then_dump_shear_query",
            "rows": 1,
            "contact_ratio": query.get("contact_ratio"),
            "final_bilateral_contact": query.get("final_bilateral_contact"),
        },
    )
    feature = build_hold_feature(env)
    posterior = deploy_prior_posterior()
    feasibility = LiftstyleFeasibility(
        models / "feasibility/FEASIBILITY_MANIFEST.json",
        manifest_sha256=spec["feasibility_manifest_sha256"],
        device="cpu",
    )
    decision = feasibility.select(feature, posterior)

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
            "evaluation_id": "liftclone-af-preaction-lock",
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
    parent = out.parent
    serial = {k: v for k, v in posterior.items() if k not in ("continuous_posterior", "raw_features", "normalized_features")}
    serial = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in serial.items()}
    write(parent / "PREACTION_POSTERIOR.json", serial)
    write(parent / "PREACTION_FEATURE.json", feature)
    np.save(parent / "PREACTION_PI0_CHUNK.npy", actions)
    cameras = {name: np.asarray(row["rgb"]).copy() for name, row in observation["observation"].items() if "rgb" in row}
    if not cameras:
        raise ValueError("Missing actual initial policy camera frames")
    np.savez_compressed(parent / "COMMON_POSTPROBE_CAMERA_CACHE.npz", **cameras)
    env._af_locked_camera_cache = cameras
    env._af_locked_native_action_count = initial_count
    env._af_locked_observation_identity = observation_identity(observation)
    env._af_locked_state_sha256 = base.digest(initial)
    write(parent / "PREACTION_AF_DECISION.json", decision)
    torch.set_num_threads(1)
    seal = {
        "context_id": context["id"],
        "root": 200002,
        "candidate_actions_executed": 0,
        "labels_read": False,
        "state_sha256": base.digest(initial),
        "first_chunk_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "observation_identity": observation_identity(observation),
        "forces_N": [float(decision["selected_force_N"])],
        "branch_methods": ["ActiveForcing"],
        "model_manifests": {
            "belief": spec["belief_manifest_sha256"],
            "feasibility": spec["feasibility_manifest_sha256"],
        },
        "artifact_hashes": {
            name: sha(parent / name)
            for name in [
                "PREACTION_POSTERIOR.json",
                "PREACTION_FEATURE.json",
                "PREACTION_PI0_CHUNK.npy",
                "PREACTION_AF_DECISION.json",
                "COMMON_POSTPROBE_CAMERA_CACHE.npz",
            ]
        },
        "context": context,
        "inference_source_sha256": sha(Path(__file__)),
        "handoff_command_m": float(outer),
        "handoff_source": "last gripper command after dump shear query",
    }
    write(parent / "PREACTION_SELECTION_LOCK.json", seal)
    env._af_preacton_selection_lock = seal
    _seal_decision(parent, context, decision)
    env._af_preacton_selection_lock = read(parent / "PREACTION_SELECTION_LOCK.json")


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
                node.value = ast.Constant("formal AF lift-clone v3 vs reused Nominal")
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
        raise RuntimeError(f"Unexpected lift-clone harness edits: {edits}")
    if ast.dump(ast.parse(baseline_text)) == ast.dump(tree):
        raise RuntimeError("Harness edits did not change AST")
    module = ModuleType("_formal_af_liftclone_v3_harness")
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    module.inference_binding_receipt = {
        "source_sha256": sha(source),
        "original_AST_recovered_exactly": True,
        "only_harness_changes": [
            "maximum one AF branch",
            "lift-clone descriptive scope",
            "handoff compares joint drive mapped to metres",
        ],
        "force_branch_control_or_physics_changed": False,
        "probe": "dump shear query after established grasp; not original pre-grasp P4",
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
        raise ValueError(f"Incomplete lift-clone v3 context: {summary.get('results')}")
    if summary["first_chunk_hashes"] != [seal["first_chunk_sha256"]]:
        raise ValueError("First action chunk pairing failed")
    if summary["common_handoff_state_sha256"] != seal["state_sha256"]:
        raise ValueError("Common handoff state changed")
    branch = next(out.glob("branch_0_*"))
    result = read(branch / "result.json")
    commanded = seal["commanded_forces_N"][0]
    if result["success"] != result["official_final_check"]:
        raise ValueError("Official success mismatch")
    if float(result["force_setpoint_bilateral_n"]) != float(commanded):
        raise ValueError("Force method setpoint mismatch")
    outcomes = [
        {
            "method": "ActiveForcing",
            "commanded_force_N": commanded,
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
            "af_decision": {
                "selected_force_N": commanded,
                "predicted_success": seal.get("decision_predicted_success"),
                "executed_selector": seal.get("executed_selector") or seal.get("decision_executed_selector"),
                "eu_counterfactual_selected_N": seal.get("decision_eu_counterfactual_selected_N"),
                "feature_source": seal.get("feature_source"),
            },
            "harness_binding": module.inference_binding_receipt,
            "claim_boundary": (
                "same-root AF lift-clone v3 vs reused Nominal; "
                "12N established grasp then dump probe then pi0 remainder"
            ),
        },
    )


if __name__ == "__main__":
    os.environ["AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP"] = "1"
    if "AF_FORMAL_CONTEXT" not in os.environ:
        raise RuntimeError("AF_FORMAL_CONTEXT is required")
    original.context_and_models = formal_context_and_models
    original.NativeOriginalFeasibility = LiftstyleFeasibility
    base.qualify = qualify
    base.main()
