from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import ModuleType

HERE = Path(__file__).resolve().parent
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
OLD = BASE / "experiments/af_dump_original_restore_20260912"
sys.path[:0] = [str(HERE), str(OLD)]

import infer_original_rootlocal as original
import infer_liftstyle
import qualify_native_interfaces as base
from liftstyle_runtime import LiftstyleFeasibility
from rootlocal_collection_contract import read, sha, verify_runtime

if sys.path[0] != str(HERE):
    sys.path.insert(0, str(HERE))


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
        raise ValueError("Frozen lift-style formal plan changed")
    schedule = [read(plan / "SMOKE_CONTEXT.json"), *read(plan / "FORMAL_CONTEXTS.json")]
    if context not in schedule:
        raise ValueError("Unplanned lift-style formal context")
    if context["root"] != 200002 or context["split"] not in ("SMOKE", "FORMAL_LIFTSTYLE"):
        raise ValueError("Wrong lift-style context scope")
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


original.context_and_models = formal_context_and_models
original.NativeOriginalFeasibility = LiftstyleFeasibility
bound = infer_liftstyle.bind()
bound["context_and_models"] = formal_context_and_models
bound["NativeOriginalFeasibility"] = LiftstyleFeasibility
restore_initial_camera_cache = original.restore_initial_camera_cache
original_locked_query = original.locked_query


def locked_query(env, out: Path) -> None:
    original_locked_query(env, out)
    _, context, _ = formal_context_and_models()
    parent = out.parent
    decision = read(parent / "PREACTION_AF_DECISION.json")
    selected = float(decision["selected_force_N"])
    if not (1.0 - 1e-9 <= selected <= 5.0 + 1e-9):
        raise ValueError(f"Selected force outside lift-style support: {selected}")
    if context["methods"] != ["ActiveForcing"]:
        raise ValueError("AF-only methods required")
    seal = read(parent / "PREACTION_SELECTION_LOCK.json")
    seal.update(
        branch_methods=["ActiveForcing"],
        forces_N=[selected],
        commanded_forces_N=[selected],
        force_support=[1.0, 5.0],
        planner_grid_step=0.5,
        selector="LiftstyleFeasibility argmax_p on [1,5]@0.5",
        inference_source_sha256=sha(__file__),
        formal_context_spec_sha256=sha(Path(os.environ["AF_FORMAL_CONTEXT"])),
        decision_executed_selector=decision.get("executed_selector"),
        decision_predicted_success=decision.get("predicted_success"),
        decision_eu_counterfactual_selected_N=decision.get("eu_counterfactual_selected_N"),
    )
    write_replace(parent / "PREACTION_SELECTION_LOCK.json", seal)
    env._af_preacton_selection_lock = seal
    os.environ["AF_ENGINEERING_FORCES"] = str(selected)


def bound_execution_module():
    source = OLD / "qualify_original_online_forks.py"
    baseline_text = source.read_text()
    tree = ast.parse(baseline_text)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qualify")
    edits = {"max_branches": 0, "scope": 0}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "max_branches":
                if ast.unparse(node.value) != "8 if collection is not None else 3":
                    raise RuntimeError("Harness branch gate drift")
                node.value = ast.Constant(1)
                edits["max_branches"] += 1
            elif node.targets[0].id == "scope":
                node.value = ast.Constant("formal AF-liftstyle only vs reused Nominal")
                edits["scope"] += 1
    if edits != {"max_branches": 1, "scope": 1}:
        raise RuntimeError(f"Unexpected lift-style harness edits: {edits}")
    if ast.dump(ast.parse(baseline_text)) == ast.dump(tree):
        raise RuntimeError("Harness edits did not change AST")
    module = ModuleType("_formal_af_liftstyle_harness")
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    module.inference_binding_receipt = {
        "source_sha256": sha(source),
        "original_AST_recovered_exactly": True,
        "only_harness_changes": ["maximum one AF branch", "lift-style descriptive scope"],
        "force_branch_control_or_physics_changed": False,
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
    _, context, _ = formal_context_and_models()
    seal = read(out / "PREACTION_SELECTION_LOCK.json")
    summary = read(out / "online_qualification.json")
    if len(summary["results"]) != 1 or summary["results"][0]["kind"] != "done":
        raise ValueError(f"Incomplete lift-style context: {summary.get('results')}")
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
                "eu_counterfactual_selected_N": seal.get("decision_eu_counterfactual_selected_N"),
            },
            "harness_binding": module.inference_binding_receipt,
            "claim_boundary": "same-root AF-liftstyle vs reused Nominal; not a lift paper transfer claim",
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
