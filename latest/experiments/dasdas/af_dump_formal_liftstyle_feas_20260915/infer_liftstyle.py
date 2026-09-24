"""Original native online execution with lift-style dump feasibility."""
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OLD))

import infer_original_rootlocal as original
from liftstyle_runtime import LiftstyleFeasibility
from max_force_utility import expected_utility
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
                node.value = ast.parse("float(expected_utility(y, force, 8.0))", mode="eval").body
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
        NativeOriginalFeasibility=LiftstyleFeasibility,
        expected_utility=expected_utility,
    )
    original_write = namespace["write"]

    def tagged_write(path, value):
        if Path(path).name == "AF_INFERENCE_RESULT.json":
            value = {
                **value,
                "utility_normalization_N": 8.0,
                "executed_selector": "argmax_p",
                "force_support": [1.0, 5.0],
                "planner_grid_step": 0.5,
                "original_execution_functions_preserved_except_outcome_utility": True,
            }
        original_write(path, value)

    namespace["write"] = tagged_write
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
