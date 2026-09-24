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

# Load the 15 N arbitration binding from this experiment directory before any
# helper that re-inserts the legacy 8 N restore path at sys.path[0].
import importlib.util

_arb_spec = importlib.util.spec_from_file_location(
    "original_arbitration_binding",
    HERE / "original_arbitration_binding.py",
)
_force15_arbitration = importlib.util.module_from_spec(_arb_spec)
sys.modules["original_arbitration_binding"] = _force15_arbitration
_arb_spec.loader.exec_module(_force15_arbitration)
_force15_arbitration.load_arbitration((3.0, 15.0))

import infer_original_rootlocal as original
import infer_v5r1
import qualify_native_interfaces as base
from force15_runtime import Force15Feasibility
from rootlocal_collection_contract import read, sha, verify_runtime

# Keep this experiment directory first even if helpers prepend OLD.
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
        raise ValueError("Frozen AF-v5r1 plan changed")
    schedule = [read(plan / "SMOKE_CONTEXT.json"), *read(plan / "FORMAL_CONTEXTS.json")]
    if context not in schedule:
        raise ValueError("Unplanned AF-v5r1 context")
    if context["root"] != 200002 or context["split"] not in ("SMOKE", "FORMAL_AF_V5R1"):
        raise ValueError("Wrong AF-v5r1 context scope")
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


# Patch the original module globals used by locked_query at call time.
original.context_and_models = formal_context_and_models
original.NativeOriginalFeasibility = Force15Feasibility

v5_namespace = infer_v5r1.bind()
v5_namespace["context_and_models"] = formal_context_and_models
v5_namespace["NativeOriginalFeasibility"] = Force15Feasibility
restore_initial_camera_cache = original.restore_initial_camera_cache
original_locked_query = original.locked_query


def locked_query(env, out: Path) -> None:
    original_locked_query(env, out)
    _, context, _ = formal_context_and_models()
    parent = out.parent
    decision = read(parent / "PREACTION_AF_DECISION.json")
    selected = float(decision["selected_force_N"])
    if not (0.5 <= selected <= 15.0):
        raise ValueError(f"Selected force outside continuous support: {selected}")
    if context["methods"] != ["ActiveForcing"]:
        raise ValueError("AF-only methods required")
    seal = read(parent / "PREACTION_SELECTION_LOCK.json")
    seal.update(
        branch_methods=["ActiveForcing"],
        forces_N=[selected],
        commanded_forces_N=[selected],
        force_support=[0.5, 15.0],
        planner_grid_step=0.05,
        selector="Force15Feasibility continuous EU",
        inference_source_sha256=sha(__file__),
        formal_context_spec_sha256=sha(Path(os.environ["AF_FORMAL_CONTEXT"])),
        decision_utility_definition=decision.get("utility_definition"),
        decision_predicted_success=decision.get("predicted_success"),
        decision_predicted_realized_squeeze_N=decision.get("predicted_realized_squeeze_N"),
    )
    write_replace(parent / "PREACTION_SELECTION_LOCK.json", seal)
    env._af_preacton_selection_lock = seal
    os.environ["AF_ENGINEERING_FORCES"] = str(selected)


def bound_execution_module():
    source = OLD / "qualify_original_online_forks.py"
    baseline_text = source.read_text()
    tree = ast.parse(baseline_text)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qualify")
    edits = {"max_branches": 0, "scope": 0, "force_upper": 0, "force_msg": 0}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "max_branches":
                if ast.unparse(node.value) != "8 if collection is not None else 3":
                    raise RuntimeError("Harness branch gate drift")
                node.value = ast.Constant(1)
                edits["max_branches"] += 1
            elif node.targets[0].id == "scope":
                node.value = ast.Constant("formal AF-v5r1 only continuous selector evaluation")
                edits["scope"] += 1
        elif isinstance(node, ast.Compare):
            # Match `.5<=x<=8` / `0.5 <= x <= 8` inside the all(...) generator.
            if (
                len(node.ops) == 2
                and isinstance(node.left, ast.Constant)
                and node.left.value in (0.5, 0.5)
                and isinstance(node.comparators[-1], ast.Constant)
                and node.comparators[-1].value == 8
            ):
                node.comparators[-1] = ast.Constant(15)
                edits["force_upper"] += 1
        elif isinstance(node, ast.Constant) and node.value == "Bounded engineering test accepts 1..3 forces within [0.5,8] N":
            node.value = "Bounded AF-v5r1 test accepts 1 force within [0.5,15] N"
            edits["force_msg"] += 1
    if edits != {"max_branches": 1, "scope": 1, "force_upper": 1, "force_msg": 1}:
        raise RuntimeError(f"Unexpected AF-v5r1 harness edits: {edits}")
    if ast.dump(ast.parse(baseline_text)) == ast.dump(tree):
        raise RuntimeError("Harness edits did not change AST")
    module = ModuleType("_formal_af_v5r1_harness")
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    module.inference_binding_receipt = {
        "source_sha256": sha(source),
        "original_AST_recovered_exactly": True,
        "only_harness_changes": [
            "maximum one AF branch",
            "AF-v5r1 descriptive scope",
            "force support gate raised to [0.5,15] N",
        ],
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
    methods = context["methods"]
    if len(summary["results"]) != 1 or summary["results"][0]["kind"] != "done":
        raise ValueError(f"Incomplete AF-v5r1 context: {summary.get('results')}")
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
                "predicted_realized_squeeze_N": seal.get("decision_predicted_realized_squeeze_N"),
                "utility_definition": seal.get("decision_utility_definition"),
            },
            "harness_binding": module.inference_binding_receipt,
            "claim_boundary": "same-root AF-v5r1 continuous selector vs reused Nominal",
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
