"""Lift-stack dump online: P4 belief, 12N grasp, dump probe, EU, π0 remainder."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
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
from admit_original_query import admit
from establish_grasp import GRASP_FORCE_N, QUERY_DISPLACEMENT_M, QUERY_FORCE_N, scripted_establish_grasp
from liftstyle_runtime import LiftstyleFeasibility
from native_cartesian_kinematics import NativeCartesianKinematics
from native_original_belief_binding import load_native_belief
from native_original_motion_features import build_features
from native_visual_mirror import observation_identity
from original_squeeze_inner import OriginalSqueezeInner
from qualify_original_p4_native import qualify as original_p4_query
from rootlocal_collection_contract import read, sha, verify_runtime, write

import infer_original_rootlocal as original

restore_initial_camera_cache = original.restore_initial_camera_cache

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
        raise ValueError("Frozen lift-clone EU Fmax5 formal plan changed")
    schedule = [read(plan / "SMOKE_CONTEXT.json"), *read(plan / "FORMAL_CONTEXTS.json")]
    if context not in schedule:
        raise ValueError("Unplanned lift-clone EU Fmax5 context")
    if context["root"] != 200002 or context["split"] not in ("SMOKE", "FORMAL_LIFTCLONE_EU_FMAX5"):
        raise ValueError("Wrong lift-clone EU Fmax5 context scope")
    if context["methods"] != ["ActiveForcing"]:
        raise ValueError("This queue is AF-only")
    verify_runtime(context["runtime_manifest_path"], context["runtime_manifest_sha256"])
    completed = read(models / "TRAINING_COMPLETE.json")
    if not completed.get("completed") or completed.get("test_groups_executed") != 0:
        raise ValueError("Invalid frozen model state")
    if completed.get("executed_selector") != "expected_utility":
        raise ValueError("Deployed selector is not expected_utility")
    for part, key in [("belief", "belief_manifest_sha256"), ("feasibility", "feasibility_manifest_sha256")]:
        manifest = models / part / ("BELIEF_MANIFEST.json" if part == "belief" else "FEASIBILITY_MANIFEST.json")
        if sha(manifest) != completed[key] or sha(manifest) != spec[key]:
            raise ValueError("Deployment checkpoint manifest changed")
        if len(read(models / part / "CHECKPOINT_SELECTION_LOCK.json")["checkpoints"]) != 3:
            raise ValueError("Missing three-member checkpoint lock")
    return spec, context, models


def _seal_decision(parent: Path, context: dict, decision: dict) -> None:
    selected = float(decision["selected_force_N"])
    if not (0.25 - 1e-9 <= selected <= 5.0 + 1e-9):
        raise ValueError(f"Selected force outside [0.25, 5]: {selected}")
    if decision.get("executed_selector") != "expected_utility":
        raise ValueError("EU must be the executed selector")
    executed = float(selected)
    if context["methods"] != ["ActiveForcing"]:
        raise ValueError("AF-only methods required")
    seal = read(parent / "PREACTION_SELECTION_LOCK.json")
    seal.update(
        branch_methods=["ActiveForcing"],
        forces_N=[executed],
        commanded_forces_N=[executed],
        selected_force_raw_N=selected,
        force_support=[0.25, 5.0],
        planner_grid_step=0.05,
        selector="LiftstyleFeasibility Fmax5 expected_utility on [0.25,5]@0.05 maxF=5",
        inference_source_sha256=sha(Path(__file__)),
        formal_context_spec_sha256=sha(Path(os.environ["AF_FORMAL_CONTEXT"])),
        decision_executed_selector=decision.get("executed_selector"),
        decision_predicted_success=decision.get("predicted_success"),
        decision_argmax_p_counterfactual_selected_N=decision.get("argmax_p_counterfactual_selected_N"),
        feature_source=decision.get("motion_channels"),
        protocol_note=(
            "P4 belief then 12N grasp then dump shear then last-command handoff "
            "then pi0 remainder; EU executed with no 0.5 clamp"
        ),
        established_grasp_N=GRASP_FORCE_N,
        query_force_N=QUERY_FORCE_N,
        force_clamped_to_0p5=False,
    )
    write_replace(parent / "PREACTION_SELECTION_LOCK.json", seal)
    os.environ["AF_ENGINEERING_FORCES"] = str(executed)


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
            "evaluation_id": "liftclone-eu-fmax5-preaction-lock",
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
    raw = read(out / "original_raw_rows.json")
    patch = read(out / "patch_readbacks.json")
    if not raw or "probe_phase" not in raw[0] or raw[0].get("task_id") != "dump_bin_bigbin":
        raise ValueError("Belief evidence is not original dump P4")
    kin = NativeCartesianKinematics(env, arm)
    joints = np.asarray(patch[-1]["finger_joints"], dtype=np.float64)
    feature = build_features(raw, joints, np.asarray(body.linear_velocity), kin.future_xyz_base(actions))
    torch.set_num_threads(1)
    belief = load_native_belief(
        models / "belief/BELIEF_MANIFEST.json",
        manifest_sha256=spec["belief_manifest_sha256"],
    )
    posterior = belief.rows(raw, patch, runtime_manifest_sha256=context["runtime_manifest_sha256"])
    posterior.update(candidate_actions_executed=0, hidden_friction_used=False)
    feasibility = LiftstyleFeasibility(
        models / "feasibility/FEASIBILITY_MANIFEST.json",
        manifest_sha256=spec["feasibility_manifest_sha256"],
        device="cpu",
    )
    decision = feasibility.select(feature, posterior)
    if base.readback(env) != initial or env.take_action_cnt != initial_count:
        raise ValueError("Decision construction changed physical state/action count")
    _assert_p4_untouched(out, p4_hashes)
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
        "belief_source": "frozen dump v4 on original P4 rows",
        "feature_source": "ONLINE_VLA_ACTION_CHUNK",
        "p4_preserved": read(out / "P4_PRESERVED.json"),
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
    edits = {"max_branches": 0, "scope": 0, "handoff_units": 0, "support": 0, "force_gate": 0}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "max_branches":
                if ast.unparse(node.value) != "8 if collection is not None else 3":
                    raise RuntimeError("Harness branch gate drift")
                node.value = ast.Constant(1)
                edits["max_branches"] += 1
            elif node.targets[0].id == "scope":
                node.value = ast.Constant("formal AF lift-clone EU Fmax5")
                edits["scope"] += 1
            elif node.targets[0].id == "support":
                node.value = ast.parse("(0.25, 5.0)", mode="eval").body
                edits["support"] += 1
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
        if isinstance(node, ast.Compare) and ast.unparse(node).replace(" ", "") in {".5<=x<=8", "0.5<=x<=8"}:
            node.left = ast.Constant(0.25)
            node.comparators = [ast.Name("x", ast.Load()), ast.Constant(5.0)]
            edits["force_gate"] += 1
    if edits != {"max_branches": 1, "scope": 1, "handoff_units": 1, "support": 1, "force_gate": 1}:
        raise RuntimeError(f"Unexpected lift-clone EU harness edits: {edits}")
    if ast.dump(ast.parse(baseline_text)) == ast.dump(tree):
        raise RuntimeError("Harness edits did not change AST")
    module = ModuleType("_formal_af_liftclone_eu_fmax5_harness")
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    module.inference_binding_receipt = {
        "source_sha256": sha(source),
        "original_AST_recovered_exactly": True,
        "only_harness_changes": [
            "maximum one AF branch",
            "lift-clone EU Fmax5 descriptive scope",
            "handoff compares joint drive mapped to metres",
            "declared support [0.25, 5]",
            "force list gate [0.25, 5] so EU is not clamped at 0.5",
        ],
        "force_branch_control_or_physics_changed": False,
        "probe": "original P4 for belief; dump shear after 12N grasp for handoff",
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
        raise ValueError(f"Incomplete lift-clone EU context: {summary.get('results')}")
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
                "executed_selector": seal.get("decision_executed_selector"),
                "argmax_p_counterfactual_selected_N": seal.get("decision_argmax_p_counterfactual_selected_N"),
                "feature_source": seal.get("feature_source"),
            },
            "harness_binding": module.inference_binding_receipt,
            "claim_boundary": (
                "same-root AF lift-clone EU Fmax5; original P4 belief, 12N grasp, "
                "dump shear, pi0 remainder; unmatched old Nominal not the comparator"
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
