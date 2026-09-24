#!/usr/bin/env python3
"""Resumable sequential queue and audit for 24 AF/strict No-Query pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PYTHON = Path("/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python")
WORKER = Path("/home/exouser/paired_strict_noquery_worker.py")
PLAN = Path(
    "/media/volume/newdata/exouser/online_vla_activeforcing_20260907/"
    "final_ablation_confirmatory_v1/FINAL_ABLATION_CONTEXT_PLAN.json"
)
FINAL = Path(
    "/media/volume/newdata/exouser/online_vla_activeforcing_20260907/"
    "final_ablation_confirmatory_v1/SOURCE_SNAPSHOT"
)
FREEZE = Path("/home/exouser/strict_noquery_freeze_20260910/FIXED_FORCE_FREEZE.json")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def server_ready(port: int) -> None:
    with socket.create_connection(("127.0.0.1", port), timeout=5):
        pass


def expected_seed(context_id: str, step: int) -> int:
    return int(hashlib.sha256(f"v1:{context_id}:{step}".encode()).hexdigest()[:8], 16)


def audit_rpc_pair(pair: Path, context_id: str) -> dict:
    branches = {}
    for method in ("STRICT_NOQUERY", "ACTIVEFORCING"):
        rows = {}
        for receipt in sorted((pair / method / "RPC").glob("*.json")):
            item = json.loads(receipt.read_text())
            step = int(item["step"])
            if item["noise_seed"] != expected_seed(context_id, step):
                raise RuntimeError(f"wrong VLA seed in {receipt}")
            rows[step] = {
                "noise_seed": item["noise_seed"],
                "noise_sha256": item["noise_sha256"],
                "checkpoint_sha256": item["checkpoint_sha256"],
            }
        branches[method] = rows
    common = sorted(set(branches["STRICT_NOQUERY"]) & set(branches["ACTIVEFORCING"]))
    mismatches = [
        step
        for step in common
        if branches["STRICT_NOQUERY"][step]["noise_seed"]
        != branches["ACTIVEFORCING"][step]["noise_seed"]
        or branches["STRICT_NOQUERY"][step]["noise_sha256"]
        != branches["ACTIVEFORCING"][step]["noise_sha256"]
    ]
    return {
        "passed": bool(common) and not mismatches,
        "strict_rpc_count": len(branches["STRICT_NOQUERY"]),
        "af_rpc_count": len(branches["ACTIVEFORCING"]),
        "common_steps": common,
        "mismatched_steps": mismatches,
    }


def aggregate(out: Path, contexts: list[dict]) -> dict:
    rows = []
    task_counts = defaultdict(lambda: {"n": 0, "strict": 0, "af": 0, "af_only": 0, "strict_only": 0})
    for context in contexts:
        pair = out / "contexts" / context["id"]
        result = json.loads((pair / "PAIR_RESULT.json").read_text())
        info = json.loads((pair / "STRICT_NOQUERY" / "STRICT_INFORMATION_GATE.json").read_text())
        restore = json.loads((pair / "PREPROBE_REFERENCE" / "PREPROBE_RESTORE_GATE.json").read_text())
        rpc = audit_rpc_pair(pair, context["id"])
        if not info["passed"] or not restore["passed"] or not rpc["passed"]:
            raise RuntimeError(f"pair audit failed: {context['id']}")
        strict = int(result["strict_success"])
        af = int(result["af_success"])
        row = {
            "context_id": context["id"],
            "task": context["task"],
            "root": context["root"],
            "band": context["band"],
            "strict_force_N": result["strict_force_N"],
            "af_force_N": result["af_force_N"],
            "strict_success": strict,
            "af_success": af,
            "af_minus_strict": af - strict,
            "same_preprobe_snapshot": result["same_preprobe_snapshot"],
            "same_vla_seed": rpc["passed"],
            "strict_rpc_count": rpc["strict_rpc_count"],
            "af_rpc_count": rpc["af_rpc_count"],
        }
        rows.append(row)
        count = task_counts[str(context["task"])]
        count["n"] += 1
        count["strict"] += strict
        count["af"] += af
        count["af_only"] += int(af == 1 and strict == 0)
        count["strict_only"] += int(strict == 1 and af == 0)

    af_only = sum(row["af_success"] == 1 and row["strict_success"] == 0 for row in rows)
    strict_only = sum(row["strict_success"] == 1 and row["af_success"] == 0 for row in rows)
    return {
        "schema": "PAIRED_AF_STRICT_NOQUERY_RESULTS_V1",
        "n": len(rows),
        "valid_pairs": len(rows),
        "strict_successes": sum(row["strict_success"] for row in rows),
        "af_successes": sum(row["af_success"] for row in rows),
        "paired_difference_count": sum(row["af_minus_strict"] for row in rows),
        "af_only_successes": af_only,
        "strict_only_successes": strict_only,
        "ties": len(rows) - af_only - strict_only,
        "task_counts": dict(task_counts),
        "all_information_gates_pass": True,
        "all_preprobe_restore_gates_pass": True,
        "all_common_vla_seed_bytes_match": True,
        "exploratory_18_of_24_included": False,
        "rows": rows,
    }


def main(args: argparse.Namespace) -> int:
    smoke = Path(args.smoke_gate)
    smoke_completions = list(smoke.glob("contexts/*/WORKER_COMPLETION.json"))
    if len(smoke_completions) != 1 or not json.loads(smoke_completions[0].read_text()).get("logical_success"):
        raise RuntimeError("one successful paired smoke is required before formal execution")
    server_ready(args.port)
    contexts = json.loads(PLAN.read_text())["contexts"]
    if len(contexts) != 24:
        raise RuntimeError("expected frozen 24-context plan")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    frozen_sources = {
        str(WORKER): sha(WORKER),
        str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
        str(PLAN): sha(PLAN),
        str(FREEZE): sha(FREEZE),
        str(FINAL / "runtime.py"): sha(FINAL / "runtime.py"),
        str(FINAL / "arbitration.py"): sha(FINAL / "arbitration.py"),
    }
    source_manifest = out / "FORMAL_SOURCE_MANIFEST.json"
    if source_manifest.exists():
        if json.loads(source_manifest.read_text())["source_hashes"] != frozen_sources:
            raise RuntimeError("formal source changed during resume")
    else:
        atomic_json(source_manifest, {"frozen_utc": now(), "source_hashes": frozen_sources})

    environment = os.environ.copy()
    environment.update(
        PYTHONNOUSERSITE="1",
        OMNI_KIT_ACCEPT_EULA="YES",
        ACCEPT_EULA="Y",
        TABERO_ROOT=str(TABERO := Path("/home/exouser/Tabero")),
        HDF5_TRAJ_SOURCE_DIR=str(TABERO / "benchmarks/datasets/libero/assembled_hdf5"),
        LIBERO_CONFIG_DIR=str(TABERO / "benchmarks/datasets/libero/config"),
        LIBERO_ASSETS_DATA_DIR=str(TABERO / "benchmarks/datasets/libero/USD"),
        PYTHONPATH=os.pathsep.join(
            [
                "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64",
                str(FINAL),
                "/home/exouser/FORTE",
                str(TABERO),
                str(TABERO / "benchmarks/openpi/openpi-client/src"),
            ]
        ),
    )
    status_path = out / "QUEUE_STATUS.json"
    for index, context in enumerate(contexts):
        completion = out / "contexts" / context["id"] / "WORKER_COMPLETION.json"
        if completion.exists() and json.loads(completion.read_text()).get("logical_success"):
            continue
        atomic_json(
            status_path,
            {
                "status": "RUNNING",
                "updated_utc": now(),
                "current_index": index,
                "current_context": context["id"],
                "completed": sum(
                    (out / "contexts" / item["id"] / "WORKER_COMPLETION.json").exists()
                    for item in contexts
                ),
                "planned": 24,
            },
        )
        command = [
            str(PYTHON),
            "-u",
            str(WORKER),
            "--out",
            str(out),
            "--plan",
            str(PLAN),
            "--force-freeze",
            str(FREEZE),
            "--context",
            str(index),
            "--job-name",
            context["id"],
            "--port",
            str(args.port),
        ]
        with (logs / f"{index:02d}_{context['id']}.log").open("x") as log:
            process = subprocess.run(command, cwd=str(TABERO), env=environment, stdout=log, stderr=subprocess.STDOUT)
        if not completion.exists() or not json.loads(completion.read_text()).get("logical_success"):
            atomic_json(
                status_path,
                {
                    "status": "FAILED",
                    "failed_utc": now(),
                    "failed_index": index,
                    "failed_context": context["id"],
                    "worker_exit_code": process.returncode,
                },
            )
            return 1

    results = aggregate(out, contexts)
    atomic_json(out / "PAIRED_RESULTS.json", results)
    atomic_json(
        status_path,
        {
            "status": "COMPLETE",
            "completed_utc": now(),
            "completed": 24,
            "planned": 24,
            "strict_successes": results["strict_successes"],
            "af_successes": results["af_successes"],
            "results_sha256": sha(out / "PAIRED_RESULTS.json"),
        },
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke-gate", required=True)
    parser.add_argument("--port", type=int, default=18885)
    raise SystemExit(main(parser.parse_args()))
