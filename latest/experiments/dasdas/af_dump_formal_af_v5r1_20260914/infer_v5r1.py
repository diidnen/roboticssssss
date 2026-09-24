"""Original native online execution with frozen Force15 continuous selector."""
from __future__ import annotations

import ast
from copy import deepcopy
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "experiments/af_dump_original_restore_20260912"
)
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HERE))

import infer_original_rootlocal as original
from force15_runtime import Force15Feasibility
from force15_utility import expected_utility, load_calibration
from rim20_formal_binding import install


def bind():
    tree = ast.parse((OLD / "infer_original_rootlocal.py").read_text())
    functions = [deepcopy(node) for node in tree.body if isinstance(node, ast.FunctionDef)]
    baseline = ast.dump(ast.Module(body=functions, type_ignores=[]))
    changes = []
    for function in functions:
        if function.name != "qualify":
            continue
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "utility"
            ):
                changes.append((node, node.value))
                node.value = ast.parse(
                    "float(expected_utility(y, force, calibration, engineering_scale_N=15.0))",
                    mode="eval",
                ).body
    if len(changes) != 1:
        raise ValueError("Unexpected original outcome-scoring layout")
    executable = deepcopy(functions)
    for node, value in changes:
        node.value = value
    if ast.dump(ast.Module(body=functions, type_ignores=[])) != baseline:
        raise ValueError("Unexpected native inference modification")

    namespace = dict(original.__dict__)
    namespace.update(
        __file__=str(Path(__file__).resolve()),
        NativeOriginalFeasibility=Force15Feasibility,
        expected_utility=expected_utility,
    )
    original_write = namespace["write"]

    def tagged_write(path, value):
        if Path(path).name == "AF_INFERENCE_RESULT.json":
            value = {
                **value,
                "utility_normalization_N": 15.0,
                "utility_definition": "p*(15-realized_squeeze(command))/15-(1-p)",
                "force_support": [0.5, 15.0],
                "planner_grid_step": 0.05,
                "original_execution_functions_preserved_except_outcome_utility": True,
            }
        original_write(path, value)

    namespace["write"] = tagged_write

    # Inject calibration into qualify scoring scope by wrapping expected_utility calls via closure.
    calibration_path = HERE / "models_deploy" / "REALIZED_FORCE_CALIBRATION.json"
    # Load via feasibility manifest sha to keep contract tight once models are locked.
    import json
    import hashlib

    feas_manifest = json.loads((HERE / "models_deploy/feasibility/FEASIBILITY_MANIFEST.json").read_text())
    calibration = load_calibration(
        Path(feas_manifest["realized_force_calibration_path"]),
        feas_manifest["realized_force_calibration_sha256"],
    )
    if hashlib.sha256(calibration_path.resolve().read_bytes()).hexdigest() != feas_manifest[
        "realized_force_calibration_sha256"
    ]:
        raise ValueError("Deploy calibration symlink mismatch")
    namespace["calibration"] = calibration

    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=executable, type_ignores=[])),
            str(__file__),
            "exec",
        ),
        namespace,
    )
    return namespace


if __name__ == "__main__":
    os.environ["AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP"] = "1"
    install()
    namespace = bind()
    original.base.qualify = namespace["qualify"]
    original.base.main()
