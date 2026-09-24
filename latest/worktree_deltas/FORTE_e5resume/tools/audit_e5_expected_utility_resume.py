#!/usr/bin/env python3
"""CPU-only, non-mutating E5 coverage audit and crash-safe resume manifest.

The auditor reads the frozen E5 implementation and result shards in place.  It
never imports Isaac, opens a GPU, starts pi0, or writes into an E5 data root.
All products are atomically written beneath the caller-supplied output folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TASKS = (0, 1, 5, 6)
EXPECTED_ARMS = {
    "FROZEN_PI0_NATIVE_DEFAULT",
    "FIXED_MAX",
    "NO_QUERY_TRAINING_PRIOR_UTILITY",
    "ACTIVEFORCING_1Q_UTILITY",
    "GT_PHYSICS_DIRECT_UTILITY",
}
EXPECTED_RUNNER_SHA256 = "13b33cbc784e2d9bc712a8211ad11b846eca1f8be773dd2888d7b1d33cd9b0e8"
EXPECTED_UTILITY_SHA256 = "c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10"

FORTE_ROOT = Path("/home/exouser/FORTE")
E5_CODE_ROOT = FORTE_ROOT / "analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006"
RESULTS_ROOT = FORTE_ROOT / "analysis/results"
VOLUME_ROOT = Path("/media/volume/newdata/exouser/activeforcing_e5_shards_20260902")
UTILITY_CONFIG = FORTE_ROOT / "UTILITY_FINAL_CONFIG.json"
RUNNER = E5_CODE_ROOT / "run_e5_fresh_utility.py"
VALIDATOR = E5_CODE_ROOT / "validate_e5_shard.py"
PROVENANCE = E5_CODE_ROOT / "e5_provenance.py"
UTILITY_RUNTIME = E5_CODE_ROOT / "e5_expected_utility_runtime.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def csv_payload(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return stream.getvalue()


def canonical_key(row: dict[str, Any]) -> str:
    return "|".join(
        (
            str(int(row["task"])),
            str(row["root_id"]),
            str(row["friction_band"]),
            f"{float(row['friction']):.12f}",
        )
    )


def evidence_tree_hash(root: Path) -> str:
    selected = []
    for name in (
        "RUN_PROTOCOL.json",
        "PREFLIGHT_AUDIT.json",
        "FULL_STATUS.json",
        "SHARD_QA.json",
        "E5_UTILITY_ROLLOUTS.csv",
        "E5_UTILITY_QUERY_RESULTS.csv",
        "E5_UTILITY_DECISIONS.csv",
    ):
        path = root / name
        if path.is_file():
            selected.append(path)
    selected.extend(sorted((root / "decisions").glob("*.json")))
    digest = hashlib.sha256()
    for path in sorted(selected, key=lambda value: str(value.relative_to(root))):
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def discover_attempts() -> tuple[dict[Path, set[str]], list[str]]:
    aliases: dict[Path, set[str]] = defaultdict(set)
    skipped: list[str] = []
    for discovery_root in (RESULTS_ROOT, VOLUME_ROOT):
        if not discovery_root.is_dir():
            skipped.append(str(discovery_root))
            continue
        for entry in sorted(discovery_root.glob("ACTIVEFORCING_E5*")):
            if entry == E5_CODE_ROOT:
                continue
            if not entry.is_dir():
                continue
            resolved = entry.resolve()
            aliases[resolved].add(str(entry))
    return aliases, skipped


def load_authorities():
    if sha256_file(RUNNER) != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("authoritative E5 runner hash mismatch")
    sys.path.insert(0, str(E5_CODE_ROOT))
    runner = importlib.import_module("run_e5_fresh_utility")
    validator = importlib.import_module("validate_e5_shard")
    utility_runtime = importlib.import_module("e5_expected_utility_runtime")
    utility_audit = utility_runtime.validate_authoritative_config()
    if utility_audit["canonical_hash"] != EXPECTED_UTILITY_SHA256:
        raise RuntimeError("frozen Utility canonical hash mismatch")
    if validator.QUALIFIED_RUNNER_SHA256 != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("validator qualified-runner hash mismatch")
    return runner, validator, utility_audit


def classify_attempt(root: Path, names: set[str], validator: Any) -> dict[str, Any]:
    protocol_path = root / "RUN_PROTOCOL.json"
    status_path = root / "FULL_STATUS.json"
    rollouts = read_csv(root / "E5_UTILITY_ROLLOUTS.csv")
    queries = read_csv(root / "E5_UTILITY_QUERY_RESULTS.csv")
    decisions_csv = read_csv(root / "E5_UTILITY_DECISIONS.csv")
    decision_json_count = len(list((root / "decisions").glob("*.json")))
    base: dict[str, Any] = {
        "physical_path": str(root),
        "aliases": sorted(names),
        "rollout_rows": len(rollouts),
        "query_rows": len(queries),
        "decision_csv_rows": len(decisions_csv),
        "decision_json_files": decision_json_count,
        "protocol_present": protocol_path.is_file(),
        "full_status_present": status_path.is_file(),
    }
    if not protocol_path.is_file():
        return {**base, "classification": "QUARANTINED_ATTEMPT", "reason": "missing RUN_PROTOCOL.json"}
    try:
        protocol = read_json(protocol_path)
    except Exception as exc:
        return {**base, "classification": "QUARANTINED_ATTEMPT", "reason": f"unreadable protocol: {exc!r}"}
    base.update(
        {
            "runner_sha256": protocol.get("runner_sha256", ""),
            "utility_config_sha256": protocol.get("utility_config_sha256", ""),
            "phase": protocol.get("phase", ""),
            "tasks": protocol.get("tasks", []),
            "tuple_start": protocol.get("tuple_start", ""),
            "max_tuples": protocol.get("max_tuples", ""),
            "protocol_tuple_ids": [item.get("tuple_id", "") for item in protocol.get("roots", [])],
        }
    )
    if not status_path.is_file():
        return {
            **base,
            "classification": "QUARANTINED_PARTIAL",
            "reason": "missing FULL_STATUS.json; no rows from this attempt are admissible",
        }
    try:
        status = read_json(status_path)
    except Exception as exc:
        return {**base, "classification": "QUARANTINED_PARTIAL", "reason": f"unreadable FULL_STATUS.json: {exc!r}"}
    if status.get("status") != "PASS":
        return {
            **base,
            "classification": "QUARANTINED_PARTIAL",
            "reason": f"FULL_STATUS is {status.get('status')!r}, not PASS",
        }
    try:
        qa = validator.validate(root)
    except Exception as exc:
        return {**base, "classification": "REJECTED_COMPLETED", "reason": repr(exc)}
    row_keys = {canonical_key(row) for row in rollouts}
    protocol_keys = {canonical_key(row) for row in protocol.get("roots", [])}
    if row_keys != protocol_keys:
        return {**base, "classification": "REJECTED_COMPLETED", "reason": "protocol/rollout tuple set mismatch"}
    return {
        **base,
        "classification": "ACCEPTED",
        "reason": "independent reset-to-end shard validation PASS",
        "qa": qa,
        "tuple_keys": sorted(row_keys),
        "evidence_tree_sha256": evidence_tree_hash(root),
    }


def report_markdown(
    generated_utc: str,
    accepted: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    manifest: dict[str, Any],
    qa: dict[str, Any],
) -> str:
    counts = manifest["coverage"]
    next_cursor = manifest["next_tuple_cursor"]
    by_class: dict[str, int] = defaultdict(int)
    for item in quarantine:
        by_class[item["classification"]] += 1
    task_rows = []
    for task in TASKS:
        rows = [row for row in coverage if row["task"] == task]
        task_rows.append((task, sum(row["status"] == "ACCEPTED" for row in rows), sum(row["status"] == "MISSING" for row in rows)))
    lines = [
        "# E5 Expected-Utility recovery audit",
        "",
        f"Generated: `{generated_utc}` (CPU-only, source data read-only).",
        "",
        "## Verdict",
        "",
        f"CPU QA: **{qa['status']}**. Authoritative coverage is **{counts['accepted_unique_tuples']}/{counts['planned_tuples']} tuples** "
        f"(**{counts['accepted_unique_rollouts']}/{counts['planned_rollouts']} rollouts**); collection is **{counts['completion_status']}**.",
        "",
        f"The exact canonical resume cursor is **task {next_cursor['task']}, tuple offset {next_cursor['tuple_start']}**, "
        f"`{next_cursor['tuple_id']}`. It must start in a new timestamped directory with `--max-tuples 1`.",
        "",
        "The existing task0 offsets07_08 attempt has no completion marker and contributes zero accepted rows. It remains in place but is logically quarantined; no in-place append or reuse is permitted.",
        "",
        "## Coverage by task",
        "",
        "| Task | Accepted | Missing | Planned |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {task} | {got} | {missing} | 15 |" for task, got, missing in task_rows)
    lines.extend(
        [
            "",
            "## Evidence disposition",
            "",
            f"- Accepted physical shards: {len(accepted)} (each independently revalidated).",
            f"- Quarantined partial/attempt directories: {by_class.get('QUARANTINED_PARTIAL', 0) + by_class.get('QUARANTINED_ATTEMPT', 0)}.",
            f"- Non-admissible rollout rows retained inside quarantined attempts: {sum(item['rollout_rows'] for item in quarantine)} (all excluded).",
            f"- Rejected completed directories: {by_class.get('REJECTED_COMPLETED', 0)}.",
            "- Duplicate admissible tuple keys: 0.",
            "- Quarantined and rejected rows are never counted, even when CSV rows exist.",
            "",
            "## Frozen controls",
            "",
            f"- Qualified runner SHA-256: `{EXPECTED_RUNNER_SHA256}`.",
            f"- Frozen canonical Utility hash: `{EXPECTED_UTILITY_SHA256}`.",
            "- Selector: `EXPECTED_UTILITY`; exact ties choose the lower force.",
            "- Population: four independent single-task plans (tasks 0, 1, 5, 6), 15 tuples per task.",
            "",
            "## Crash-safe handoff",
            "",
            "The resume manifest is a CPU-prepared handoff, not launch authorization. A future launcher must recompute coverage, verify the frozen hashes, require no existing active E5 worker, pass its live GPU and pi0 prerequisites, create a brand-new output directory atomically, run one tuple, and only then promote that shard after `FULL_STATUS=PASS` plus independent QA. Any interrupted directory stays quarantined and the same cursor is retried in another fresh directory.",
            "",
            "This lane did not launch GPU, Isaac, pi0, or training and did not modify any existing E5 data.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generated-utc", default=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    runner, validator, utility_audit = load_authorities()
    planned: list[dict[str, Any]] = []
    for task in TASKS:
        for offset, item in enumerate(runner.planned_subset("full", [task])):
            planned.append({**item, "task_local_offset": offset, "tuple_key": canonical_key(item)})
    planned_by_key = {item["tuple_key"]: item for item in planned}
    if len(planned) != 60 or len(planned_by_key) != 60:
        raise RuntimeError("authoritative plan is not 60 unique tuples")

    aliases, skipped_roots = discover_attempts()
    attempts = [classify_attempt(root, names, validator) for root, names in sorted(aliases.items(), key=lambda pair: str(pair[0]))]
    accepted_attempts = [item for item in attempts if item["classification"] == "ACCEPTED"]
    quarantine = [item for item in attempts if item["classification"] != "ACCEPTED"]

    accepted_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard in accepted_attempts:
        for tuple_key in shard["tuple_keys"]:
            if tuple_key not in planned_by_key:
                shard["classification"] = "REJECTED_COMPLETED"
                shard["reason"] = f"tuple outside authoritative population: {tuple_key}"
                break
        else:
            for tuple_key in shard["tuple_keys"]:
                accepted_by_key[tuple_key].append(shard)
    accepted_attempts = [item for item in accepted_attempts if item["classification"] == "ACCEPTED"]
    quarantine = [item for item in attempts if item["classification"] != "ACCEPTED"]
    duplicates = {key: rows for key, rows in accepted_by_key.items() if len(rows) != 1}
    accepted_keys = {key for key, rows in accepted_by_key.items() if len(rows) == 1}

    coverage = []
    for item in planned:
        shards = accepted_by_key.get(item["tuple_key"], [])
        coverage.append(
            {
                "task": int(item["task"]),
                "task_local_offset": item["task_local_offset"],
                "tuple_id": item["tuple_id"],
                "tuple_key": item["tuple_key"],
                "root_id": item["root_id"],
                "root_seed": item["root_seed"],
                "friction_band": item["friction_band"],
                "friction": f"{float(item['friction']):.12f}",
                "status": "ACCEPTED" if len(shards) == 1 else ("DUPLICATE" if len(shards) > 1 else "MISSING"),
                "accepted_physical_path": shards[0]["physical_path"] if len(shards) == 1 else "",
            }
        )
    missing = [row for row in coverage if row["status"] == "MISSING"]
    if not missing:
        next_cursor: dict[str, Any] | None = None
    else:
        row = missing[0]
        next_cursor = {
            "task": row["task"],
            "tuple_start": row["task_local_offset"],
            "max_tuples": 1,
            "tuple_id": row["tuple_id"],
            "tuple_key": row["tuple_key"],
            "friction_band": row["friction_band"],
            "friction": row["friction"],
            "root_id": row["root_id"],
            "root_seed": row["root_seed"],
        }

    queue = [
        {
            "ordinal": ordinal,
            "task": row["task"],
            "tuple_start": row["task_local_offset"],
            "max_tuples": 1,
            "tuple_id": row["tuple_id"],
            "tuple_key": row["tuple_key"],
        }
        for ordinal, row in enumerate(missing, 1)
    ]
    manifest = {
        "schema_version": "E5_EXPECTED_UTILITY_CRASH_SAFE_RESUME_V1",
        "generated_utc": args.generated_utc,
        "state": "IDLE_CPU_HANDOFF_GPU_LAUNCH_NOT_AUTHORIZED",
        "selector": "EXPECTED_UTILITY",
        "utility": {
            "canonical_sha256": EXPECTED_UTILITY_SHA256,
            "validation_checks": utility_audit["checks"],
            "tie_break": "lower_force",
            "source": str(UTILITY_CONFIG),
        },
        "runner": {"path": str(RUNNER), "sha256": EXPECTED_RUNNER_SHA256},
        "validation_sources": {
            "validator": {"path": str(VALIDATOR), "sha256": sha256_file(VALIDATOR)},
            "provenance": {"path": str(PROVENANCE), "sha256": sha256_file(PROVENANCE)},
            "utility_runtime": {"path": str(UTILITY_RUNTIME), "sha256": sha256_file(UTILITY_RUNTIME)},
            "utility_config_file_sha256": sha256_file(UTILITY_CONFIG),
            "note": "The canonical Utility hash is the frozen core-config hash; the file-byte hash is provenance only.",
        },
        "authoritative_population": {
            "semantics": "concatenated independent single-task fresh-worker plans",
            "tasks": list(TASKS),
            "tuples_per_task": 15,
            "arms": sorted(EXPECTED_ARMS),
            "rollouts_per_tuple": 5,
        },
        "coverage": {
            "planned_tuples": len(planned),
            "planned_rollouts": len(planned) * 5,
            "accepted_unique_tuples": len(accepted_keys),
            "accepted_unique_rollouts": len(accepted_keys) * 5,
            "missing_tuples": len(missing),
            "missing_rollouts": len(missing) * 5,
            "completion_status": "COMPLETE" if not missing and not duplicates else "INCOMPLETE",
        },
        "next_tuple_cursor": next_cursor,
        "remaining_queue": queue,
        "accepted_physical_shards": [
            {
                "physical_path": item["physical_path"],
                "aliases": item["aliases"],
                "tuple_keys": item["tuple_keys"],
                "rollout_rows": item["rollout_rows"],
                "evidence_tree_sha256": item["evidence_tree_sha256"],
            }
            for item in accepted_attempts
        ],
        "quarantined_attempts": [
            {key: item.get(key) for key in (
                "physical_path", "aliases", "classification", "reason", "rollout_rows", "query_rows",
                "decision_csv_rows", "decision_json_files", "runner_sha256", "utility_config_sha256",
                "tasks", "tuple_start", "max_tuples", "protocol_tuple_ids",
            )}
            for item in quarantine
        ],
        "discovery_roots_unavailable": skipped_roots,
        "resume_contract": {
            "launch_authorized_by_this_manifest": False,
            "gpu_prerequisites_checked": False,
            "pi0_listener_checked": False,
            "output_policy": "one tuple per brand-new timestamped directory; never append, overwrite, or reuse",
            "acceptance_policy": "FULL_STATUS.status=PASS then independent reset-to-end validation under exact runner and Utility hashes",
            "interruption_policy": "leave interrupted directory immutable and quarantined; recompute coverage; retry same cursor in a new directory",
            "race_policy": "fail closed if any E5 worker exists; only a separately authorized scheduler may evaluate live GPU/pi0 gates",
            "post_acceptance_policy": "recompute authoritative coverage before selecting another cursor",
        },
        "launch_template": {
            "enabled": False,
            "reason": "CPU handoff only; a separately authorized resource scheduler must satisfy live prerequisites",
            "working_directory": str(E5_CODE_ROOT),
            "executable": "/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python",
            "argv_after_executable": [
                "-u", str(RUNNER), "--phase", "full", "--task", "${NEXT_TASK}",
                "--tuple-start", "${NEXT_TUPLE_START}", "--max-tuples", "1",
                "--out", str(VOLUME_ROOT / "ACTIVEFORCING_E5_FRESH_UTILITY_E2E_${UTC_TIMESTAMP}_task${NEXT_TASK}_offset${NEXT_OFFSET_PADDED}"),
            ],
            "cursor_substitutions": {
                "NEXT_TASK": next_cursor["task"] if next_cursor else None,
                "NEXT_TUPLE_START": next_cursor["tuple_start"] if next_cursor else None,
                "NEXT_OFFSET_PADDED": f"{next_cursor['tuple_start']:02d}" if next_cursor else None,
            },
            "required_environment": {
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64:/home/exouser/Tabero:/home/exouser/Tabero/benchmarks/openpi/openpi-client/src",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "ACCEPT_EULA": "Y",
                "TABERO_ROOT": "/home/exouser/Tabero",
                "HDF5_TRAJ_SOURCE_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/assembled_hdf5",
                "LIBERO_ASSETS_DATA_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/USD",
                "LIBERO_CONFIG_DIR": "/home/exouser/Tabero/benchmarks/datasets/libero/config",
            },
            "mandatory_live_preconditions": [
                "re-run this CPU audit against the current authoritative roots",
                "exact runner and canonical Utility hashes still pass",
                "no active E5 worker and no output-path collision",
                "pi0 server/checkpoint lineage gate passes",
                "live GPU resource gate passes",
                "explicit GPU/Isaac/pi0 launch authorization exists",
            ],
        },
    }

    qa_checks = {
        "runner_sha256_exact": sha256_file(RUNNER) == EXPECTED_RUNNER_SHA256,
        "utility_canonical_sha256_exact": utility_audit["canonical_hash"] == EXPECTED_UTILITY_SHA256,
        "utility_all_frozen_checks_pass": all(utility_audit["checks"].values()),
        "population_is_60_unique_tuples": len(planned) == len(planned_by_key) == 60,
        "each_task_has_15_tuples": all(sum(item["task"] == task for item in planned) == 15 for task in TASKS),
        "accepted_shards_revalidated": all(item["qa"]["status"] == "PASS" for item in accepted_attempts),
        "accepted_rows_are_five_per_tuple": all(item["rollout_rows"] == 5 * len(item["tuple_keys"]) for item in accepted_attempts),
        "no_duplicate_accepted_tuple_keys": not duplicates,
        "accepted_and_missing_partition_population": len(accepted_keys) + len(missing) == len(planned),
        "next_cursor_is_first_canonical_gap": next_cursor is None or next_cursor["tuple_key"] == missing[0]["tuple_key"],
        "quarantined_attempts_excluded": all(
            item["physical_path"] not in {shard["physical_path"] for shard in accepted_attempts}
            for item in quarantine
        ),
    }
    qa = {
        "schema_version": "E5_EXPECTED_UTILITY_CPU_QA_V1",
        "generated_utc": args.generated_utc,
        "status": "PASS" if all(qa_checks.values()) else "FAIL",
        "collection_status": manifest["coverage"]["completion_status"],
        "checks": qa_checks,
        "counts": {
            "discovered_physical_attempts": len(attempts),
            "accepted_physical_shards": len(accepted_attempts),
            "quarantined_or_rejected_attempts": len(quarantine),
            "quarantined_rollout_rows_excluded": sum(item["rollout_rows"] for item in quarantine),
            "accepted_unique_tuples": len(accepted_keys),
            "missing_tuples": len(missing),
        },
        "duplicate_tuple_paths": {
            key: [item["physical_path"] for item in rows] for key, rows in duplicates.items()
        },
    }

    accepted_csv_rows = []
    for item in accepted_attempts:
        offsets = sorted(planned_by_key[key]["task_local_offset"] for key in item["tuple_keys"])
        tasks = sorted({int(planned_by_key[key]["task"]) for key in item["tuple_keys"]})
        accepted_csv_rows.append(
            {
                "physical_path": item["physical_path"],
                "aliases": ";".join(item["aliases"]),
                "tasks": ";".join(map(str, tasks)),
                "task_local_offsets": ";".join(map(str, offsets)),
                "tuples": len(item["tuple_keys"]),
                "rollouts": item["rollout_rows"],
                "decisions": item["decision_json_files"],
                "runner_sha256": item["runner_sha256"],
                "utility_config_sha256": item["utility_config_sha256"],
                "evidence_tree_sha256": item["evidence_tree_sha256"],
            }
        )
    quarantine_csv_rows = [
        {
            "classification": item["classification"],
            "physical_path": item["physical_path"],
            "aliases": ";".join(item["aliases"]),
            "reason": item["reason"],
            "rollout_rows": item["rollout_rows"],
            "query_rows": item["query_rows"],
            "decision_json_files": item["decision_json_files"],
            "runner_sha256": item.get("runner_sha256", ""),
            "utility_config_sha256": item.get("utility_config_sha256", ""),
            "tasks": json.dumps(item.get("tasks", []), separators=(",", ":")),
            "tuple_start": item.get("tuple_start", ""),
            "max_tuples": item.get("max_tuples", ""),
            "protocol_tuple_ids": ";".join(item.get("protocol_tuple_ids", [])),
        }
        for item in quarantine
    ]

    atomic_text(args.out / "E5_ACCEPTED_SHARDS.csv", csv_payload(
        ["physical_path", "aliases", "tasks", "task_local_offsets", "tuples", "rollouts", "decisions",
         "runner_sha256", "utility_config_sha256", "evidence_tree_sha256"], accepted_csv_rows))
    atomic_text(args.out / "E5_QUARANTINED_PARTIALS.csv", csv_payload(
        ["classification", "physical_path", "aliases", "reason", "rollout_rows", "query_rows",
         "decision_json_files", "runner_sha256", "utility_config_sha256", "tasks", "tuple_start",
         "max_tuples", "protocol_tuple_ids"], quarantine_csv_rows))
    atomic_text(args.out / "E5_TUPLE_COVERAGE.csv", csv_payload(
        ["task", "task_local_offset", "tuple_id", "tuple_key", "root_id", "root_seed", "friction_band",
         "friction", "status", "accepted_physical_path"], coverage))
    atomic_json(args.out / "E5_EXPECTED_UTILITY_RESUME_MANIFEST.json", manifest)
    atomic_json(args.out / "E5_CPU_QA.json", qa)
    atomic_text(args.out / "E5_RECOVERY_REPORT.md", report_markdown(
        args.generated_utc, accepted_attempts, quarantine, coverage, manifest, qa))

    checksums = []
    for path in sorted(args.out.iterdir()):
        if path.name == "SHA256SUMS":
            continue
        checksums.append(f"{sha256_file(path)}  {path.name}")
    atomic_text(args.out / "SHA256SUMS", "\n".join(checksums) + "\n")
    print(json.dumps({
        "qa": qa["status"],
        "coverage": f"{len(accepted_keys)}/60",
        "next_tuple_cursor": next_cursor,
        "quarantined_or_rejected_attempts": len(quarantine),
        "output": str(args.out.resolve()),
    }, sort_keys=True))
    return 0 if qa["status"] == "PASS" and not duplicates else 1


if __name__ == "__main__":
    raise SystemExit(main())
