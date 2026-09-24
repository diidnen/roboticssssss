#!/usr/bin/env python3
"""Independent audit of the frozen 24-pair AF/strict No-Query result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


GROUPS = ("environment", "joint_targets", "materials", "objects", "rng", "state")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def expected_seed(context_id: str, step: int) -> int:
    return int(hashlib.sha256(f"v1:{context_id}:{step}".encode()).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = read(args.plan)["contexts"]
    results_path = args.root / "PAIRED_RESULTS.json"
    results = read(results_path)
    status = read(args.root / "QUEUE_STATUS.json")
    assert status["status"] == "COMPLETE" and status["completed"] == status["planned"] == 24
    assert status["results_sha256"] == sha(results_path)
    assert len(plan) == results["n"] == results["valid_pairs"] == 24

    rows = []
    task_counts = defaultdict(lambda: {"n": 0, "strict": 0, "af": 0, "af_only": 0, "strict_only": 0})
    total_rpc = 0
    for context in plan:
        context_id = context["id"]
        pair = args.root / "contexts" / context_id
        assert read(pair / "WORKER_COMPLETION.json")["logical_success"] is True
        pair_result = read(pair / "PAIR_RESULT.json")
        gate = read(pair / "STRICT_NOQUERY" / "STRICT_INFORMATION_GATE.json")
        assert gate == {
            "belief_loaded": False,
            "feasibility_loaded": False,
            "forbidden_modules_loaded": [],
            "force_source": "TRAIN_ONLY_PER_TASK_FIXED_FREEZE",
            "passed": True,
            "physical_probe_actions_executed": 0,
            "probe_phases_executed": [],
            "selected_force_N": pair_result["strict_force_N"],
        }
        with (pair / "PREPROBE_REFERENCE" / "ESTABLISHED_PREFIX.csv").open(newline="") as handle:
            prefix = list(csv.DictReader(handle))
        assert len(prefix) == 190
        assert {row["probe_phase"] for row in prefix} == {"approach", "descend", "close", "hold"}
        assert all(row["contact_state"] == "bilateral" for row in prefix[-10:])

        restore = read(pair / "PREPROBE_REFERENCE" / "PREPROBE_RESTORE_GATE.json")
        assert restore["passed"] is True
        assert tuple(sorted(restore["checks"])) == GROUPS
        assert all(restore["checks"].values())
        with (pair / "ACTIVEFORCING" / "RAW_PROBE.csv").open(newline="") as handle:
            af_probe = list(csv.DictReader(handle))
        assert any(row["probe_phase"] == "probe_out" for row in af_probe)

        branch_rpc = {}
        for branch in ("STRICT_NOQUERY", "ACTIVEFORCING"):
            receipts = {}
            for path in sorted((pair / branch / "RPC").glob("*.json")):
                receipt = read(path)
                step = int(receipt["step"])
                assert receipt["context_id"] == context_id
                assert receipt["noise_seed"] == expected_seed(context_id, step)
                receipts[step] = receipt
            assert len(receipts) == 35
            branch_rpc[branch] = receipts
            total_rpc += len(receipts)
        assert branch_rpc["STRICT_NOQUERY"].keys() == branch_rpc["ACTIVEFORCING"].keys()
        for step in branch_rpc["STRICT_NOQUERY"]:
            strict_rpc = branch_rpc["STRICT_NOQUERY"][step]
            af_rpc = branch_rpc["ACTIVEFORCING"][step]
            assert strict_rpc["noise_seed"] == af_rpc["noise_seed"]
            assert strict_rpc["noise_sha256"] == af_rpc["noise_sha256"]
            assert strict_rpc["checkpoint_sha256"] == af_rpc["checkpoint_sha256"]

        strict = int(pair_result["strict_success"])
        af = int(pair_result["af_success"])
        rows.append((context_id, context["task"], strict, af))
        count = task_counts[str(context["task"])]
        count["n"] += 1
        count["strict"] += strict
        count["af"] += af
        count["af_only"] += int(af == 1 and strict == 0)
        count["strict_only"] += int(strict == 1 and af == 0)

    strict_successes = sum(row[2] for row in rows)
    af_successes = sum(row[3] for row in rows)
    af_only = [row[0] for row in rows if row[2] == 0 and row[3] == 1]
    strict_only = [row[0] for row in rows if row[2] == 1 and row[3] == 0]
    assert strict_successes == results["strict_successes"] == 19
    assert af_successes == results["af_successes"] == 18
    assert len(af_only) == results["af_only_successes"] == 1
    assert len(strict_only) == results["strict_only_successes"] == 2
    assert dict(task_counts) == results["task_counts"]

    audit = {
        "schema": "INDEPENDENT_PAIRED_AF_STRICT_NOQUERY_AUDIT_V1",
        "passed": True,
        "valid_pairs": 24,
        "strict_successes": strict_successes,
        "af_successes": af_successes,
        "strict_rate": strict_successes / 24,
        "af_rate": af_successes / 24,
        "af_minus_strict_rate": (af_successes - strict_successes) / 24,
        "af_only_contexts": af_only,
        "strict_only_contexts": strict_only,
        "discordant_pairs": len(af_only) + len(strict_only),
        "exact_two_sided_mcnemar_p": 1.0,
        "task_counts": dict(task_counts),
        "strict_information_gates": "24/24",
        "preprobe_restore_gates": "24/24",
        "paired_vla_seed_and_noise_hash_gates": "840/840 common RPC steps",
        "total_rpc_receipts_audited": total_rpc,
        "exploratory_18_of_24_included": False,
        "results_sha256": sha(results_path),
        "source_manifest_sha256": sha(args.root / "FORMAL_SOURCE_MANIFEST.json"),
        "declared_limit": "unexposed PhysX contact/solver warm-start state is outside snapshot API",
    }
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
