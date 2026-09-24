"""Resume one task of the predeclared 24-context No-Query comparison.

This adapter imports the frozen E5 runner unchanged, replaces only its
synthetic plan with the existing 24-context online-ablation plan, and limits
execution to the two scientifically required arms: ActiveForcing and
No-Query/Training-Prior. Both arms retain the frozen E5 rollout code. Running
one task per process avoids the simulator hang observed while switching tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path("/home/exouser/FORTE")
SOURCE = ROOT / "analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006/run_e5_fresh_utility.py"
PLAN = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_ablation_confirmatory_v1/FINAL_ABLATION_CONTEXT_PLAN.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("noquery24_e5", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def contexts() -> list[dict]:
    data = json.loads(PLAN.read_text())
    rows = data["contexts"]
    expected = {(root, task, band) for root in (170048, 170049) for task in (0, 1, 5, 6) for band in ("LOW", "MID", "HIGH")}
    got = {(r["root"], r["task"], r["band"]) for r in rows}
    if len(rows) != 24 or got != expected:
        raise RuntimeError("The predeclared 24-context plan is incomplete or changed")
    return [
        {
            "phase": "noquery24", "task": r["task"], "object": r["object"],
            "root_index": (0 if r["root"] == 170048 else 1), "root_seed": r["root"],
            "root_id": f"noquery24_t{r['task']}_r{r['root']}",
            "friction_band": r["band"], "friction": r["mu"],
            "tuple_id": f"noquery24_{r['id']}",
        }
        for r in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=(1, 5, 6))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    module = load_source()
    plan = contexts()
    module.ARM_NAMES = ["NO_QUERY_TRAINING_PRIOR_UTILITY", "ACTIVEFORCING_1Q_UTILITY"]
    module.ARM_EXECUTION_ORDER = ["active", "prior"]
    module.PROTOCOL_VERSION = "ACTIVEFORCING_NOQUERY24_FRESH_RESET_V1"
    # The frozen validator compares rows against its module-level protocol
    # constant. This adapter intentionally declares a new protocol because it
    # changes the context plan and arm set, so validate against that declared
    # protocol while leaving every other provenance check unchanged.
    module.provenance.EXPECTED_PROTOCOL = module.PROTOCOL_VERSION
    module.planned_subset = lambda phase, tasks, tuple_start=0, max_tuples=None: [r for r in plan if r["task"] in tasks]
    original_main = module.main
    sys.argv = [
        str(SOURCE), "--phase", "full", "--task", str(args.task),
        "--out", str(args.out),
    ]
    result = original_main()
    out = args.out
    (out / "NOQUERY24_ADAPTER_MANIFEST.json").write_text(json.dumps({
        "adapter": str(Path(__file__).resolve()), "adapter_sha256": sha(Path(__file__)),
        "frozen_e5_runner": str(SOURCE), "frozen_e5_runner_sha256": sha(SOURCE),
        "context_plan": str(PLAN), "context_plan_sha256": sha(PLAN),
        "resume_reason": "task-isolated restart after cross-task simulator hang",
        "task": args.task, "contexts": 6, "arms": module.ARM_NAMES,
        "noquery_semantics": "fresh root reset; query count 0; training-prior expected-utility selection",
    }, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
