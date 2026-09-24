from __future__ import annotations

import ast
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
V4 = HERE.parent / "af_dump_maxf8_20260913"
OLD = HERE.parent / "af_dump_original_restore_20260912"
sys.path[:0] = [str(V4), str(OLD)]

import infer_v4
import qualify_native_interfaces as base
from native_visual_mirror import capture_visual_packet
from rootlocal_collection_contract import read, sha, verify_runtime


def write_replace(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def formal_context_and_models():
    spec = read(os.environ["AF_FORMAL_CONTEXT"])
    context = spec["context"]
    models = Path(spec["models"])
    plan = Path(context["formal_protocol_path"]).parent
    lock = read(plan / "FREEZE_LOCK.json")
    if sha(plan / "PROTOCOL.json") != lock["protocol_sha256"]:
        raise ValueError("VAL policy protocol changed")
    if sha(plan / "SMOKE_CONTEXT.json") != lock["smoke_context_sha256"]:
        raise ValueError("VAL smoke context changed")
    if sha(plan / "VAL_CONTEXTS.json") != lock["val_contexts_sha256"]:
        raise ValueError("VAL contexts changed")
    schedule = [read(plan / "SMOKE_CONTEXT.json"), *read(plan / "VAL_CONTEXTS.json")]
    if context not in schedule:
        raise ValueError("Unplanned VAL policy context")
    if context["root"] != 200002 or context["split"] not in ("SMOKE", "VAL_POLICY"):
        raise ValueError("Wrong VAL policy context scope")
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


v4_namespace = infer_v4.bind()
v4_namespace["context_and_models"] = formal_context_and_models
v4_locked_query = v4_namespace["locked_query"]
restore_initial_camera_cache = v4_namespace["restore_initial_camera_cache"]


def locked_query(env, out: Path) -> None:
    v4_locked_query(env, out)
    spec, context, _ = formal_context_and_models()
    parent = out.parent
    decision = read(parent / "PREACTION_AF_DECISION.json")
    selected = float(decision["selected_force_N"])
    methods = context["methods"]
    commanded = []
    harness = []
    for method in methods:
        if method == "Nominal Frozen VLA":
            commanded.append(None)
            harness.append(8.0)
        elif method == "Fixed-Strong 8N":
            commanded.append(8.0)
            harness.append(8.0)
        elif method == "ActiveForcing":
            commanded.append(selected)
            harness.append(selected)
        else:
            raise ValueError("Unknown formal method")
    seal = read(parent / "PREACTION_SELECTION_LOCK.json")
    seal.update(
        branch_methods=methods,
        forces_N=harness,
        commanded_forces_N=commanded,
        nominal_harness_force_sentinel_N=8.0,
        nominal_harness_force_sentinel_used_for_branch_naming_only=True,
        inference_source_sha256=sha(__file__),
        formal_context_spec_sha256=sha(Path(os.environ["AF_FORMAL_CONTEXT"])),
    )
    write_replace(parent / "PREACTION_SELECTION_LOCK.json", seal)
    env._af_preacton_selection_lock = seal
    os.environ["AF_ENGINEERING_FORCES"] = ",".join(str(value) for value in harness)


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
                            "raw_vla_arm_command": action[offset:offset + 6].tolist(),
                            "final_arm_command": action[offset:offset + 6].tolist(),
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
    tree = ast.parse(source.read_text())
    baseline = ast.dump(tree)
    changes = []
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qualify")
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id == "max_branches":
            if ast.unparse(node.value) != "8 if collection is not None else 3":
                raise RuntimeError("Harness branch gate drift")
            changes.append((node, node.value))
            node.value = ast.Constant(3)
        elif node.targets[0].id == "scope":
            changes.append((node, node.value))
            node.value = ast.Constant("VAL policy paired same-root evaluation with native nominal comparator")
    if len(changes) != 2:
        raise RuntimeError("Unexpected formal harness binding count")
    executable = deepcopy(tree)
    for node, previous in changes:
        node.value = previous
    if ast.dump(tree) != baseline:
        raise RuntimeError("Undeclared original harness change")
    module = ModuleType("_formal_single_root_harness")
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(executable), str(source), "exec"), module.__dict__)
    force_child_run = module.child_run

    def dispatch(env, pipe, out, force, handoff, stock_drives, support, scope):
        context = read(os.environ["AF_FORMAL_CONTEXT"])["context"]
        match = re.match(r"branch_(\d+)_", out.name)
        if not match:
            raise ValueError("Cannot resolve branch index")
        method = context["methods"][int(match.group(1))]
        if method == "Nominal Frozen VLA":
            return nominal_child_run(env, pipe, out, force, handoff, stock_drives, support, scope)
        return force_child_run(env, pipe, out, force, handoff, stock_drives, support, scope)

    module.child_run = dispatch
    module.inference_binding_receipt = {
        "source_sha256": sha(source),
        "original_AST_recovered_exactly": True,
        "only_harness_changes": ["maximum three paired branches", "VAL descriptive scope", "native nominal child dispatch"],
        "force_branch_control_or_physics_changed": False,
    }
    return module


def qualify(env, out: Path) -> None:
    module = bound_execution_module()
    module.qualify_query = locked_query
    live_renderer = module.native_observation_from_packet
    module.native_observation_from_packet = lambda current_env, packet: restore_initial_camera_cache(current_env, packet, live_renderer)
    module.qualify(env, out)
    _, context, _ = formal_context_and_models()
    seal = read(out / "PREACTION_SELECTION_LOCK.json")
    summary = read(out / "online_qualification.json")
    methods = context["methods"]
    if len(summary["results"]) != len(methods) or any(row["kind"] != "done" for row in summary["results"]):
        raise ValueError("Incomplete VAL paired context")
    if summary["first_chunk_hashes"] != [seal["first_chunk_sha256"]] * len(methods):
        raise ValueError("First action chunk pairing failed")
    if summary["common_handoff_state_sha256"] != seal["state_sha256"]:
        raise ValueError("Common handoff state changed")
    outcomes = []
    for index, method in enumerate(methods):
        branch = next(out.glob(f"branch_{index}_*"))
        result = read(branch / "result.json")
        commanded = seal["commanded_forces_N"][index]
        if result["success"] != result["official_final_check"]:
            raise ValueError("Official success mismatch")
        if method == "Nominal Frozen VLA":
            if result["force_setpoint_bilateral_n"] is not None or result["force_controller_installed"] or result["gripper_force_override"]:
                raise ValueError("Nominal branch used a force override")
        elif float(result["force_setpoint_bilateral_n"]) != float(commanded):
            raise ValueError("Force method setpoint mismatch")
        outcomes.append(
            {
                "method": method,
                "commanded_force_N": commanded,
                "success": int(result["success"]),
                "measured_mean_squeeze_N": result["measured_mean_squeeze_n"],
                "measured_max_squeeze_N": result["measured_max_squeeze_n"],
                "native_actions": result["native_actions"],
                "result_sha256": sha(branch / "result.json"),
            }
        )
    write_replace(
        out / "FORMAL_CONTEXT_RESULT.json",
        {
            "completed": True,
            "context": context,
            "outcomes": outcomes,
            "paired_rollouts": len(methods),
            "selection_lock_sha256": sha(out / "PREACTION_SELECTION_LOCK.json"),
            "first_chunk_sha256": seal["first_chunk_sha256"],
            "common_handoff_state_sha256": seal["state_sha256"],
            "harness_binding": module.inference_binding_receipt,
            "claim_boundary": "feasibility-VAL motion/friction policy sanity only",
        },
    )


if __name__ == "__main__":
    os.environ["AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP"] = "1"
    if "AF_FORMAL_CONTEXT" not in os.environ:
        raise RuntimeError("AF_FORMAL_CONTEXT is required")
    from rim20_formal_binding import install

    install()
    base.qualify = qualify
    base.main()

