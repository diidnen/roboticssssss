"""Execute two frozen cross-force branches using the original V4 locked-query body."""
import ast
from copy import deepcopy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "af_dump_maxf8_20260913"
sys.path.insert(0, str(V4))
import infer_v4


def bind():
    namespace = infer_v4.bind()
    source = Path(infer_v4.original.__file__)
    function = deepcopy(
        next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "locked_query"
        )
    )
    baseline = ast.dump(function)
    edits = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "forces"
        ):
            edits.append((node, "value", node.value))
            node.value = ast.parse("_declared_forces(spec)", mode="eval").body
        if isinstance(node, ast.Dict):
            for index, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == "branch_methods":
                    original = list(node.values)
                    edits.append((node, "values", original))
                    node.values = list(original)
                    node.values[index] = ast.parse("spec['crossforce']['methods']", mode="eval").body
    if len(edits) != 2:
        raise ValueError("Original locked-query layout changed")
    executable = deepcopy(function)
    for node, attribute, value in edits:
        setattr(node, attribute, value)
    if ast.dump(function) != baseline:
        raise ValueError("Unexpected execution edit")

    def declared_forces(spec):
        case = spec["crossforce"]
        protocol = namespace["read"](HERE / "CROSSFORCE_PROTOCOL.json")
        if case not in protocol["cases"]:
            raise ValueError("Undeclared cross-force case")
        if namespace["sha"](HERE / "CROSSFORCE_PROTOCOL.json") != spec["crossforce_protocol_sha256"]:
            raise ValueError("Protocol drift")
        for path, expected in protocol["source_hashes"].items():
            if namespace["sha"](path) != expected:
                raise ValueError("Source drift: " + path)
        values = case["forces_N"]
        if len(values) != 2 or len(case["methods"]) != 2 or not all(0.5 <= force <= 8 for force in values):
            raise ValueError("Wrong branch schedule")
        return values

    namespace["_declared_forces"] = declared_forces
    original_write = namespace["write"]

    def write_result(path, value):
        if Path(path).name == "AF_INFERENCE_RESULT.json":
            value = {
                **value,
                "formal_AF_inference": False,
                "scientific_stage": "POSTHOC_DEVELOPMENT_CROSS_FORCE",
                "historical_outcomes_informed_selection": True,
                "original_models_unchanged": True,
                "original_execution_unchanged": True,
                "scope": "two selected same-state mu=.675 actions; not unbiased confirmation",
                "diagnostic_source_sha256": namespace["sha"](__file__),
            }
            path = Path(path).with_name("CROSSFORCE_RESULT.json")
        original_write(path, value)

    namespace["write"] = write_result
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[executable], type_ignores=[])), __file__, "exec"),
        namespace,
    )
    return namespace


if __name__ == "__main__":
    os.environ["AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP"] = "1"
    infer_v4.install()
    ns = bind()
    infer_v4.original.base.qualify = ns["qualify"]
    infer_v4.original.base.main()
