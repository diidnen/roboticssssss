"""Audited extension of the original arbitration constructor support to 15 N.

Only the constructor's accepted force interval is rebound.  The utility,
release rule, feedback loop, and action equations are executed from the
unchanged original source AST.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np


SOURCE = Path(
    "/media/volume/newdata/exouser/online_vla_activeforcing_20260907/"
    "final_continuous_friction_generalization_v1/SOURCE_SNAPSHOT/arbitration.py"
)


def load_arbitration(support=(3.0, 5.0)):
    lo, hi = map(float, support)
    if not (0 < lo <= hi <= 15):
        raise ValueError("Engineering support must remain positive and at most 15 N")
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    original = ast.dump(tree)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Arbitration")
    ctor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    guard = ctor.body[0]
    if not isinstance(guard, ast.If) or ast.unparse(guard.test) != "not 3 <= force <= 5":
        raise RuntimeError("Original arbitration force guard drifted")
    old = guard.test
    guard.test = ast.parse("not SUPPORT[0] <= force <= SUPPORT[1]", mode="eval").body
    executable = deepcopy(tree)
    guard.test = old
    if ast.dump(tree) != original:
        raise RuntimeError("Undeclared arbitration change")
    namespace = {"SUPPORT": (lo, hi)}
    exec(compile(ast.fix_missing_locations(executable), str(SOURCE), "exec"), namespace)
    receipt = {
        "original_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "force_support_bilateral_N": [lo, hi],
        "only_force_guard_binding_changed": True,
        "original_AST_recovered_exactly": True,
        "utility_changed": False,
        "release_and_feedback_rules_changed": False,
        "engineering_extension_above_training_support": True,
    }
    return namespace["Arbitration"], receipt


if __name__ == "__main__":
    cls, receipt = load_arbitration((0.5, 15.0))
    action = np.zeros(13, np.float32)
    for force in (0.5, 5.0, 8.0, 10.0, 12.0, 15.0):
        value, decision = cls(force, 0.006).action(action)
        assert value[9] == value[12] == force / 2
        assert not decision["vla_release_intent"]
    print("ORIGINAL_ARBITRATION_15N_GUARD_ONLY_BINDING_PASSED", receipt)
