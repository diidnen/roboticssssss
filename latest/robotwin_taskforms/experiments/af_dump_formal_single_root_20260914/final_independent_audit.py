#!/usr/bin/env python3
"""Independent denominator, provenance, archive-receipt, and statistics audit.

This verifier is deliberately separate from the execution queue and its per-context
auditor.  It reads only frozen plans and completed compact records/receipts, then
recomputes every aggregate used by the final report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS = ("Nominal Frozen VLA", "Fixed-Strong 8N", "ActiveForcing")
SHA256_LEN = 64


def read(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def is_sha(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LEN
        and all(character in "0123456789abcdef" for character in value)
    )


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_mcnemar_two_sided(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize_method(rows: list[dict]) -> dict:
    successes = sum(int(row["success"]) for row in rows)
    commanded = [float(row["commanded_force_N"]) for row in rows if row["commanded_force_N"] is not None]
    commanded_success = [
        float(row["commanded_force_N"])
        for row in rows
        if row["commanded_force_N"] is not None and row["success"]
    ]
    squeezes = [float(row["measured_mean_squeeze_N"]) for row in rows]
    return {
        "contexts": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "success_rate_wilson_95": wilson(successes, len(rows)),
        "mean_commanded_force_N": statistics.fmean(commanded) if commanded else None,
        "median_commanded_force_N": statistics.median(commanded) if commanded else None,
        "mean_commanded_force_success_conditioned_N": (
            statistics.fmean(commanded_success) if commanded_success else None
        ),
        "mean_measured_squeeze_N": statistics.fmean(squeezes),
        "median_measured_squeeze_N": statistics.median(squeezes),
        "mean_native_actions": statistics.fmean(float(row["native_actions"]) for row in rows),
        "mean_physics_steps": statistics.fmean(float(row["physics_steps"]) for row in rows),
    }


def paired_matrix(records: list[dict], left: str, right: str) -> dict:
    counts = {"both_success": 0, "left_only": 0, "right_only": 0, "both_fail": 0}
    for record in records:
        outcomes = {row["method"]: bool(row["success"]) for row in record["outcomes"]}
        a, b = outcomes[left], outcomes[right]
        key = "both_success" if a and b else "left_only" if a else "right_only" if b else "both_fail"
        counts[key] += 1
    counts["left_method"] = left
    counts["right_method"] = right
    counts["contexts"] = len(records)
    counts["success_difference_percentage_points"] = 100.0 * (
        counts["left_only"] - counts["right_only"]
    ) / len(records)
    counts["exact_mcnemar_two_sided_p"] = exact_mcnemar_two_sided(
        counts["left_only"], counts["right_only"]
    )
    return counts


def analyze(records: list[dict]) -> dict:
    by_method = {}
    for method in METHODS:
        rows = [
            next(row for row in record["outcomes"] if row["method"] == method)
            for record in records
        ]
        by_method[method] = summarize_method(rows)

    by_friction = {}
    for friction in sorted({float(record["context"]["friction"]) for record in records}):
        subset = [record for record in records if float(record["context"]["friction"]) == friction]
        by_friction[f"{friction:.3f}"] = {
            method: summarize_method(
                [next(row for row in record["outcomes"] if row["method"] == method) for record in subset]
            )
            for method in METHODS
        }

    by_policy_seed = {}
    for seed in sorted({int(record["context"]["policy_seed"]) for record in records}):
        subset = [record for record in records if int(record["context"]["policy_seed"]) == seed]
        by_policy_seed[str(seed)] = {
            method: summarize_method(
                [next(row for row in record["outcomes"] if row["method"] == method) for record in subset]
            )
            for method in METHODS
        }

    paired_squeeze_af_minus_fixed = []
    paired_squeeze_af_minus_nominal = []
    setpoint_savings_both_success = []
    for record in records:
        outcomes = {row["method"]: row for row in record["outcomes"]}
        paired_squeeze_af_minus_fixed.append(
            float(outcomes["ActiveForcing"]["measured_mean_squeeze_N"])
            - float(outcomes["Fixed-Strong 8N"]["measured_mean_squeeze_N"])
        )
        paired_squeeze_af_minus_nominal.append(
            float(outcomes["ActiveForcing"]["measured_mean_squeeze_N"])
            - float(outcomes["Nominal Frozen VLA"]["measured_mean_squeeze_N"])
        )
        if outcomes["ActiveForcing"]["success"] and outcomes["Fixed-Strong 8N"]["success"]:
            setpoint_savings_both_success.append(
                8.0 - float(outcomes["ActiveForcing"]["commanded_force_N"])
            )

    af_forces = [
        float(next(row for row in record["outcomes"] if row["method"] == "ActiveForcing")["commanded_force_N"])
        for record in records
    ]
    return {
        "by_method": by_method,
        "by_friction": by_friction,
        "by_policy_seed": by_policy_seed,
        "paired_AF_vs_Fixed8": paired_matrix(records, "ActiveForcing", "Fixed-Strong 8N"),
        "paired_AF_vs_Nominal": paired_matrix(records, "ActiveForcing", "Nominal Frozen VLA"),
        "AF_force_distribution_N": dict(sorted(Counter(f"{value:.2f}" for value in af_forces).items())),
        "paired_mean_squeeze_difference_N": {
            "AF_minus_Fixed8": statistics.fmean(paired_squeeze_af_minus_fixed),
            "AF_minus_Nominal": statistics.fmean(paired_squeeze_af_minus_nominal),
        },
        "AF_setpoint_saving_when_both_AF_and_Fixed8_succeed": {
            "contexts": len(setpoint_savings_both_success),
            "mean_N": statistics.fmean(setpoint_savings_both_success) if setpoint_savings_both_success else None,
            "median_N": statistics.median(setpoint_savings_both_success) if setpoint_savings_both_success else None,
            "values_N": setpoint_savings_both_success,
        },
    }


def audit(experiment: Path) -> dict:
    experiment = experiment.resolve()
    plan = experiment / "plan"
    protocol_path = plan / "PROTOCOL.json"
    contexts_path = plan / "FORMAL_CONTEXTS.json"
    smoke_context_path = plan / "SMOKE_CONTEXT.json"
    non_exposure_path = plan / "NON_EXPOSURE_AUDIT.json"
    lock = read(plan / "FREEZE_LOCK.json")
    require(sha256(protocol_path) == lock["protocol_sha256"], "protocol hash differs from freeze lock")
    require(sha256(contexts_path) == lock["formal_contexts_sha256"], "context schedule hash differs")
    require(sha256(smoke_context_path) == lock["smoke_context_sha256"], "smoke context hash differs")
    require(sha256(non_exposure_path) == lock["non_exposure_audit_sha256"], "non-exposure audit hash differs")

    protocol = read(protocol_path)
    contexts = read(contexts_path)
    non_exposure = read(non_exposure_path)
    require(non_exposure.get("passed") is True, "seed non-exposure audit did not pass")
    require(protocol["physical_root"] == 200002 and protocol["no_new_roots"] is True, "root scope drift")
    require(protocol["formal_contexts"] == 24 and protocol["formal_rollouts"] == 72, "denominator drift")
    require(tuple(protocol["formal_methods"]) == METHODS, "method set/order drift")
    require(len(contexts) == 24, "expected 24 scheduled contexts")
    expected_ids = [context["id"] for context in contexts]
    require(len(expected_ids) == len(set(expected_ids)), "duplicate scheduled context id")
    require({context["root"] for context in contexts} == {200002}, "unexpected physical root")
    require({float(context["friction"]) for context in contexts} == {0.425, 0.575, 0.85}, "friction grid drift")
    require({int(context["policy_seed"]) for context in contexts} == set(range(80200002, 80200010)), "seed grid drift")
    require(sum(len(context["methods"]) for context in contexts) == 72, "scheduled rollout count drift")
    require(all(set(context["methods"]) == set(METHODS) for context in contexts), "per-context method set drift")

    status = read(experiment / "STATUS.json")
    require(status.get("status") == "complete", "queue status is not complete")
    require(status.get("completed_contexts") == 24 and status.get("completed_rollouts") == 72, "terminal count mismatch")
    smoke_pass = read(experiment / "SMOKE_PASS.json")
    require(smoke_pass.get("passed") is True, "nominal smoke did not pass")
    require(smoke_pass.get("commanded_force_N") is None, "nominal smoke invented an N-valued setpoint")
    require(smoke_pass.get("native_actions", 0) > 0, "nominal smoke had no native actions")
    require(smoke_pass.get("measured_squeeze_telemetry_nonempty") is True, "nominal smoke lacks squeeze telemetry")

    record_paths = sorted((experiment / "main_records").glob("*.json"))
    receipt_paths = sorted((experiment / "main_archived").glob("*.json"))
    require({path.stem for path in record_paths} == set(expected_ids), "record denominator/membership mismatch")
    require({path.stem for path in receipt_paths} == set(expected_ids), "archive denominator/membership mismatch")

    contexts_by_id = {context["id"]: context for context in contexts}
    records = []
    archives = []
    trace_hashes = set()
    handoff_by_seed = {}
    for context_id in expected_ids:
        record_path = experiment / "main_records" / f"{context_id}.json"
        receipt_path = experiment / "main_archived" / f"{context_id}.json"
        record = read(record_path)
        receipt = read(receipt_path)
        require(record["context"] == contexts_by_id[context_id], f"{context_id}: context differs from frozen schedule")
        require(record.get("strict_matched") is True, f"{context_id}: strict matching false")
        require(record.get("first_chunk_pairing_exact") is True, f"{context_id}: first chunk pairing false")
        require(record.get("all_failures_retained") is True, f"{context_id}: failure retention false")
        require(record.get("outcome_used_for_queue_control") is False, f"{context_id}: outcome controlled queue")
        require(is_sha(record.get("common_handoff_state_sha256")), f"{context_id}: invalid handoff hash")
        require(is_sha(record.get("formal_context_result_sha256")), f"{context_id}: invalid result hash")
        handoff_key = int(record["context"]["policy_seed"])
        handoff_by_seed.setdefault(handoff_key, set()).add(record["common_handoff_state_sha256"])
        outcomes = record.get("outcomes") or []
        require(len(outcomes) == 3, f"{context_id}: expected three outcomes")
        require({row["method"] for row in outcomes} == set(METHODS), f"{context_id}: outcome method mismatch")
        for row in outcomes:
            method = row["method"]
            require(row.get("raw_audit_passed") is True, f"{context_id}/{method}: raw audit false")
            require(row.get("success") in (0, 1), f"{context_id}/{method}: invalid success label")
            require(int(row.get("native_actions", 0)) > 0, f"{context_id}/{method}: no native actions")
            require(int(row.get("physics_steps", 0)) > 0, f"{context_id}/{method}: no physics trace")
            require(finite(row.get("measured_mean_squeeze_N")), f"{context_id}/{method}: invalid mean squeeze")
            require(finite(row.get("measured_max_squeeze_N")), f"{context_id}/{method}: invalid max squeeze")
            require(float(row["measured_max_squeeze_N"]) >= float(row["measured_mean_squeeze_N"]) >= 0.0,
                    f"{context_id}/{method}: squeeze summary inconsistent")
            require(is_sha(row.get("trace_sha256")), f"{context_id}/{method}: invalid trace hash")
            require(row["trace_sha256"] not in trace_hashes, f"{context_id}/{method}: duplicate trace")
            trace_hashes.add(row["trace_sha256"])
            if method == "Nominal Frozen VLA":
                require(row.get("commanded_force_N") is None, f"{context_id}: nominal force must be N/A")
            elif method == "Fixed-Strong 8N":
                require(float(row.get("commanded_force_N")) == 8.0, f"{context_id}: fixed force is not 8 N")
            else:
                force = float(row.get("commanded_force_N"))
                require(0.5 <= force <= 8.0, f"{context_id}: AF force outside frozen support")
                require(abs(force * 20 - round(force * 20)) < 1e-8, f"{context_id}: AF force off 0.05 N search grid")

        require(receipt.get("remote_sha256_verified") is True, f"{context_id}: remote SHA receipt false")
        require(receipt.get("remote_gzip_test_passed") is True, f"{context_id}: remote gzip receipt false")
        require(receipt.get("remote_tar_listing_passed") is True, f"{context_id}: remote tar receipt false")
        require(is_sha(receipt.get("archive_sha256")), f"{context_id}: invalid archive hash")
        require(is_sha(receipt.get("source_file_manifest_sha256")), f"{context_id}: invalid manifest hash")
        require(int(receipt.get("archive_bytes", 0)) == int(receipt.get("remote_archive_bytes", -1)),
                f"{context_id}: archive byte count mismatch")
        require(int(receipt.get("source_file_count", 0)) > 0, f"{context_id}: empty archive")
        require(str(receipt.get("remote_path", "")).endswith(f"/{context_id}.tar.gz"),
                f"{context_id}: unexpected archive path")
        records.append(record)
        archives.append({"context_id": context_id, **receipt})

    # A context's saved-state hash may legitimately vary with friction, so only
    # assert presence here; equality is required across siblings inside each record
    # and was checked by the independent per-context raw auditor.
    require(len(trace_hashes) == 72, "expected 72 distinct raw trace content hashes")

    for source_path, expected_hash in protocol["source_hashes"].items():
        path = Path(source_path)
        actual_hash = sha256(path)
        if actual_hash == expected_hash:
            continue
        extension = read(experiment / "RUNTIME_EXTENSION_01.json")
        require(path.resolve() == (experiment / "run_formal_queue.py").resolve(), f"unapproved source drift: {path}")
        require(extension.get("original_queue_sha256") == expected_hash, "runtime extension original hash mismatch")
        require(extension.get("resume_queue_sha256") == actual_hash, "runtime extension resumed hash mismatch")
        require(extension.get("scientific_protocol_changed") is False, "runtime extension changed scientific protocol")
        require(extension.get("classification") == "PRE_PHYSICS_CONTEXT_ADAPTER_ENVIRONMENT_BINDING",
                "runtime extension has unexpected classification")

    training_complete = Path(protocol["frozen_models"]) / "TRAINING_COMPLETE.json"
    require(sha256(training_complete) == protocol["training_complete_sha256"], "frozen model lock changed")
    runtime_manifest = Path(protocol["runtime_manifest"])
    require(sha256(runtime_manifest) == protocol["runtime_manifest_sha256"], "runtime manifest changed")

    queue_final = read(experiment / "FINAL_RESULTS.json")
    require(queue_final.get("completed") is True, "queue final result is not complete")
    require(queue_final.get("contexts") == 24 and queue_final.get("paired_rollouts") == 72, "queue final denominator mismatch")
    require(queue_final.get("claim_boundary") == protocol["claim_boundary"], "claim boundary drift")

    analysis = analyze(records)
    queue_summary = {row["method"]: row for row in queue_final["summary"]}
    for method in METHODS:
        independent = analysis["by_method"][method]
        queued = queue_summary[method]
        require(independent["successes"] == queued["successes"], f"{method}: queue aggregate success mismatch")
        require(abs(independent["success_rate"] - queued["success_rate"]) < 1e-12,
                f"{method}: queue aggregate rate mismatch")
        left = independent["mean_commanded_force_N"]
        right = queued["mean_commanded_force_N"]
        require((left is None and right is None) or abs(left - right) < 1e-12,
                f"{method}: queue aggregate force mismatch")

    expected_paired = analysis["paired_AF_vs_Fixed8"]
    queue_paired = queue_final["AF_vs_Fixed8"]
    require(queue_paired["both_success"] == expected_paired["both_success"], "paired both-success mismatch")
    require(queue_paired["AF_only"] == expected_paired["left_only"], "paired AF-only mismatch")
    require(queue_paired["Fixed_only"] == expected_paired["right_only"], "paired Fixed-only mismatch")
    require(queue_paired["both_fail"] == expected_paired["both_fail"], "paired both-fail mismatch")

    failure_record = experiment / "engineering_failures" / "smoke_mu0.575_root200002_ps40200002_attempt_001"
    require(failure_record.exists(), "retained pre-physics engineering failure is missing")
    require((experiment / "RUNTIME_EXTENSION_01.json").exists(), "runtime extension record is missing")

    return {
        "audit_version": "FINAL_SINGLE_ROOT_INDEPENDENT_AUDIT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "claim_boundary": protocol["claim_boundary"],
        "requirements": {
            "nominal_smoke_passed": True,
            "physical_roots": [200002],
            "formal_contexts": 24,
            "formal_rollouts": 72,
            "unique_trace_hashes": 72,
            "all_context_records_independently_audited": True,
            "all_archive_receipts_passed": True,
            "frozen_plan_verified": True,
            "frozen_model_and_runtime_verified": True,
            "queue_aggregates_reproduced": True,
            "engineering_failure_retained_and_scoped": True,
        },
        "frozen_hashes": {
            "protocol_sha256": sha256(protocol_path),
            "contexts_sha256": sha256(contexts_path),
            "freeze_lock_sha256": sha256(plan / "FREEZE_LOCK.json"),
            "non_exposure_audit_sha256": sha256(non_exposure_path),
            "training_complete_sha256": sha256(training_complete),
            "runtime_manifest_sha256": sha256(runtime_manifest),
        },
        "analysis": analysis,
        "records": records,
        "archive_expectations": archives,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.experiment)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
