"""One open-loop high-force context: record π₀ once, replay across forces."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "af_dump_original_restore_20260912"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(OLD))

import importlib.util

_arb_spec = importlib.util.spec_from_file_location(
    "original_arbitration_binding",
    HERE / "original_arbitration_binding.py",
)
_arb = importlib.util.module_from_spec(_arb_spec)
sys.modules["original_arbitration_binding"] = _arb
_arb_spec.loader.exec_module(_arb)
_arb.load_arbitration((0.5, 20.0))

import qualify_native_interfaces as base
from openloop_execution import execute_openloop_context
from rootlocal_collection_contract import read, sha, verify_runtime

if sys.path[0] != str(HERE):
    sys.path.insert(0, str(HERE))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def context_spec():
    spec = read(Path(os.environ["AF_OPENLOOP_CONTEXT"]))
    context = spec["context"]
    plan = Path(context["protocol_path"]).parent
    lock = read(plan / "FREEZE_LOCK.json")
    checks = {
        "protocol_sha256": sha(plan / "PROTOCOL.json"),
        "smoke_context_sha256": sha(plan / "SMOKE_CONTEXT.json"),
        "main_contexts_sha256": sha(plan / "MAIN_CONTEXTS.json"),
    }
    if checks != lock:
        raise ValueError("Frozen open-loop plan changed")
    schedule = [read(plan / "SMOKE_CONTEXT.json"), *read(plan / "MAIN_CONTEXTS.json")]
    if context not in schedule:
        raise ValueError("Unplanned open-loop context")
    if context["root"] != 200002 or context["split"] not in ("ENGINEERING_SMOKE", "MAIN_OPENLOOP"):
        raise ValueError("Wrong open-loop scope")
    verify_runtime(context["runtime_manifest_path"], context["runtime_manifest_sha256"])
    if float(context["friction"]) != float(spec["friction"]):
        raise ValueError("SPEC friction mismatch")
    return spec, context


def qualify(env, out: Path) -> None:
    _, context = context_spec()
    if float(env.af_contact_friction) != float(context["friction"]):
        raise ValueError("Runtime friction differs from frozen context")
    if int(env._af_qualification_policy_seed) != int(context["policy_seed"]):
        raise ValueError("Runtime policy seed differs from frozen context")
    if os.environ.get("AF_P4_PHYSICAL_SURFACE_CAMERA") != "1" or os.environ.get("AF_ORIGINAL_SQUEEZE_INNER") != "1":
        raise ValueError("Physical surface camera and full original squeeze loop are required")
    outcomes = execute_openloop_context(
        env,
        out,
        forces=context["forces_N"],
        record_force=context["record_force_N"],
        support=tuple(context["force_support_N"]),
        policy_seed=int(context["policy_seed"]),
        prior_branch=context.get("prior_successful_branch"),
    )
    result = read(out / "OPENLOOP_CONTEXT_RESULT.json")
    result["context"] = context
    write_json(out / "OPENLOOP_CONTEXT_RESULT.json", result)
    write_json(out / "FIXED_SWEEP_CONTEXT_RESULT.json", result)  # reuse audit naming


if __name__ == "__main__":
    os.environ["AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP"] = "1"
    if "AF_OPENLOOP_CONTEXT" not in os.environ:
        raise RuntimeError("AF_OPENLOOP_CONTEXT is required")
    from rim20_formal_binding import install

    install()
    base.qualify = qualify
    base.main()
