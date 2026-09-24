#!/usr/bin/env python3
"""Read-only formal trace audit for force realization and pre-action inputs.

Runs where the verified formal tarballs live.  It does not modify models,
labels, rollouts, or selection locks.  Each physics trace is re-hashed against
the prior deep audit while contact-conditioned squeeze statistics are derived.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile
from typing import Optional


METHODS = ("Nominal Frozen VLA", "Fixed-Strong 8N", "ActiveForcing")


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_target_contact(row: dict) -> bool:
    for finger in ((row.get("contact") or {}).get("fingers") or []):
        for point in finger.get("points") or []:
            if not point.get("is_target"):
                continue
            separation = point.get("separation_m")
            impulse = point.get("impulse_ns") or [0.0, 0.0, 0.0]
            if (separation is not None and float(separation) < 0.002) or any(
                abs(float(value)) > 1e-8 for value in impulse
            ):
                return True
    return False


def quantile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": quantile(values, 0.10),
        "p90": quantile(values, 0.90),
        "max": max(values),
    }


def trace_metrics(stream, command: Optional[float]) -> dict:
    content_hash = hashlib.sha256()
    all_values: list[float] = []
    contact_values: list[float] = []
    no_target_values: list[float] = []
    logged_steps = 0
    target_contact_steps = 0
    within_1n = 0
    within_20pct = 0
    with gzip.GzipFile(fileobj=stream, mode="rb") as trace:
        for raw_line in trace:
            encoded = raw_line.rstrip(b"\n")
            content_hash.update(encoded)
            row = json.loads(encoded)
            logged_steps += 1
            if int(row.get("physics_step")) != logged_steps:
                raise AssertionError("non-contiguous physics trace")
            value = (row.get("contact") or {}).get("measured_squeeze_n")
            if value is None:
                value = (row.get("original_squeeze_inner") or {}).get("measured_filtered_N")
            if value is None:
                raise AssertionError("missing squeeze telemetry")
            value = float(value)
            all_values.append(value)
            if has_target_contact(row):
                target_contact_steps += 1
                contact_values.append(value)
                if command is not None:
                    within_1n += abs(value - command) <= 1.0
                    within_20pct += abs(value - command) <= 0.20 * command
            else:
                no_target_values.append(value)
    if not all_values:
        raise AssertionError("empty trace")
    contact = describe(contact_values)
    result = {
        "logged_steps": logged_steps,
        "trace_content_sha256": content_hash.hexdigest(),
        "target_contact_steps": target_contact_steps,
        "target_contact_fraction": target_contact_steps / logged_steps,
        "all_step_squeeze_N": describe(all_values),
        "target_contact_squeeze_N": contact,
        "no_target_contact_squeeze_N": describe(no_target_values),
        "commanded_force_N": command,
        "contact_conditioned_tracking_error_N": None,
        "contact_conditioned_tracking_ratio": None,
        "contact_steps_within_1N_fraction": None,
        "contact_steps_within_20pct_fraction": None,
    }
    if command is not None and contact_values:
        result.update(
            contact_conditioned_tracking_error_N=contact["mean"] - command,
            contact_conditioned_tracking_ratio=contact["mean"] / command,
            contact_steps_within_1N_fraction=within_1n / len(contact_values),
            contact_steps_within_20pct_fraction=within_20pct / len(contact_values),
        )
    return result


def summarize(rows: list[dict]) -> dict:
    result = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        all_sum = sum(row["all_step_squeeze_N"]["mean"] * row["all_step_squeeze_N"]["n"] for row in group)
        all_n = sum(row["all_step_squeeze_N"]["n"] for row in group)
        contact_sum = sum(
            row["target_contact_squeeze_N"]["mean"] * row["target_contact_squeeze_N"]["n"]
            for row in group if row["target_contact_squeeze_N"]["n"]
        )
        contact_n = sum(row["target_contact_squeeze_N"]["n"] for row in group)
        tracked = [row for row in group if row["commanded_force_N"] is not None]
        result[method] = {
            "contexts": len(group),
            "successes": sum(row["success"] for row in group),
            "mean_commanded_force_N": (
                statistics.fmean(row["commanded_force_N"] for row in tracked) if tracked else None
            ),
            "context_mean_all_step_squeeze_N": statistics.fmean(
                row["all_step_squeeze_N"]["mean"] for row in group
            ),
            "step_weighted_all_step_squeeze_N": all_sum / all_n,
            "context_mean_target_contact_squeeze_N": statistics.fmean(
                row["target_contact_squeeze_N"]["mean"] for row in group
                if row["target_contact_squeeze_N"]["n"]
            ),
            "step_weighted_target_contact_squeeze_N": contact_sum / contact_n,
            "mean_target_contact_fraction": statistics.fmean(row["target_contact_fraction"] for row in group),
            "context_mean_tracking_ratio": (
                statistics.fmean(row["contact_conditioned_tracking_ratio"] for row in tracked) if tracked else None
            ),
            "context_mean_tracking_error_N": (
                statistics.fmean(row["contact_conditioned_tracking_error_N"] for row in tracked) if tracked else None
            ),
            "contact_step_weighted_within_1N_fraction": (
                sum(row["contact_steps_within_1N_fraction"] * row["target_contact_steps"] for row in tracked)
                / sum(row["target_contact_steps"] for row in tracked) if tracked else None
            ),
            "contact_step_weighted_within_20pct_fraction": (
                sum(row["contact_steps_within_20pct_fraction"] * row["target_contact_steps"] for row in tracked)
                / sum(row["target_contact_steps"] for row in tracked) if tracked else None
            ),
        }
        for label, selected in (("success", [r for r in group if r["success"]]),
                                ("failure", [r for r in group if not r["success"]])):
            result[method][label] = {
                "n": len(selected),
                "mean_commanded_force_N": statistics.fmean(
                    r["commanded_force_N"] for r in selected if r["commanded_force_N"] is not None
                ) if selected and selected[0]["commanded_force_N"] is not None else None,
                "mean_all_step_squeeze_N": statistics.fmean(r["all_step_squeeze_N"]["mean"] for r in selected) if selected else None,
                "mean_target_contact_squeeze_N": statistics.fmean(
                    r["target_contact_squeeze_N"]["mean"] for r in selected
                ) if selected else None,
                "mean_target_contact_fraction": statistics.fmean(r["target_contact_fraction"] for r in selected) if selected else None,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--deep-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs-output", type=Path, required=True)
    args = parser.parse_args()

    core = read_json(args.core)
    deep = read_json(args.deep_audit)
    if not core.get("passed") or not deep.get("passed"):
        raise AssertionError("upstream audit did not pass")
    deep_by_context = {row["context_id"]: row for row in deep["contexts"]}
    record_by_context = {row["context"]["id"]: row for row in core["records"]}
    rows = []
    inputs = []
    for expectation in core["archive_expectations"]:
        context_id = expectation["context_id"]
        record = record_by_context[context_id]
        methods = record["context"]["methods"]
        outcomes = {row["method"]: row for row in record["outcomes"]}
        prior = {row["method"]: row for row in deep_by_context[context_id]["branch_summaries"]}
        archive = Path(expectation["remote_path"])
        if not archive.is_file() or archive.stat().st_size != int(expectation["archive_bytes"]):
            raise AssertionError(f"archive unavailable or wrong size: {archive}")
        preaction = {}
        trace_count = 0
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                relative = str(PurePosixPath(member.name).relative_to(context_id))
                stream = tf.extractfile(member)
                if stream is None:
                    raise AssertionError(member.name)
                if relative in {
                    "job/PREACTION_FEATURE.json",
                    "job/PREACTION_POSTERIOR.json",
                    "job/PREACTION_AF_DECISION.json",
                }:
                    payload = stream.read()
                    preaction[PurePosixPath(relative).stem] = {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "value": json.loads(payload),
                    }
                elif relative.endswith("/physics_trace.jsonl.gz"):
                    branch_name = PurePosixPath(relative).parts[1]
                    branch_index = int(branch_name.split("_")[1])
                    method = methods[branch_index]
                    command = outcomes[method].get("commanded_force_N")
                    if command is None and method == "Fixed-Strong 8N":
                        command = 8.0
                    command = None if command is None else float(command)
                    derived = trace_metrics(stream, command)
                    expected = prior[method]
                    if derived["trace_content_sha256"] != expected["trace_content_sha256"]:
                        raise AssertionError(f"trace hash mismatch: {context_id}/{method}")
                    if not math.isclose(
                        derived["all_step_squeeze_N"]["mean"],
                        float(expected["measured_mean_squeeze_N"]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise AssertionError(f"mean squeeze mismatch: {context_id}/{method}")
                    rows.append({
                        "context_id": context_id,
                        "friction": float(record["context"]["friction"]),
                        "policy_seed": int(record["context"]["policy_seed"]),
                        "method": method,
                        "success": int(bool(outcomes[method]["success"])),
                        **derived,
                    })
                    trace_count += 1
        if trace_count != 3 or set(preaction) != {
            "PREACTION_FEATURE", "PREACTION_POSTERIOR", "PREACTION_AF_DECISION"
        }:
            raise AssertionError(f"missing members: {context_id}")
        decision = preaction["PREACTION_AF_DECISION"]["value"]
        inputs.append({
            "context_id": context_id,
            "friction": float(record["context"]["friction"]),
            "policy_seed": int(record["context"]["policy_seed"]),
            "feature": preaction["PREACTION_FEATURE"]["value"],
            "posterior": preaction["PREACTION_POSTERIOR"]["value"],
            "original_decision": decision,
            "member_sha256": {key: value["sha256"] for key, value in preaction.items()},
        })

    if len(rows) != 72 or len(inputs) != 24:
        raise AssertionError((len(rows), len(inputs)))
    output = {
        "version": "FORMAL_COMMAND_SQUEEZE_AUDIT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "scope": "read-only post-hoc derived audit of 24 formal contexts / 72 verified traces",
        "definitions": {
            "all_step_squeeze": "mean over every logged physics step, including steps without qualifying target contact",
            "target_contact_squeeze": "mean over steps satisfying the frozen target-contact predicate",
            "tracking_ratio": "target-contact-conditioned mean squeeze divided by commanded bilateral force",
            "causal_limit": "tracking conditional on retained contact cannot determine why contact was lost",
        },
        "source_hashes": {
            str(args.core): sha256(args.core),
            str(args.deep_audit): sha256(args.deep_audit),
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
        "archive_expectation_sha256": {
            row["context_id"]: row["archive_sha256"] for row in core["archive_expectations"]
        },
        "summary": summarize(rows),
        "rows": sorted(rows, key=lambda row: (row["policy_seed"], row["friction"], METHODS.index(row["method"]))),
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    input_output = {
        "version": "FORMAL_PREACTION_INPUTS_FOR_OFFLINE_DIAGNOSIS_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "scope": "pre-action feature/posterior/decision only; no outcomes embedded",
        "source_core_sha256": sha256(args.core),
        "contexts": sorted(inputs, key=lambda row: (row["policy_seed"], row["friction"])),
    }
    args.inputs_output.write_text(json.dumps(input_output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "inputs_output": str(args.inputs_output), "rows": len(rows)}))


if __name__ == "__main__":
    main()
