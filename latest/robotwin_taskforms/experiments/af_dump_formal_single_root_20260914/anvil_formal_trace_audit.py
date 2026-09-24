#!/usr/bin/env python3
"""Deep, one-pass audit of formal tarballs on Anvil.

The script recomputes each compressed archive SHA256, verifies every archived
member against ARCHIVE_FILE_MANIFEST.json, replays all 72 physics traces, and
recomputes outcome/squeeze/contact-loss/intermediate-task summaries from raw state.
It is meant to run on Anvil where the tarballs reside, avoiding bulk data transfer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile


METHODS = ("Nominal Frozen VLA", "Fixed-Strong 8N", "ActiveForcing")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        if block:
            self.digest.update(block)
            self.bytes += len(block)
        return block


def read_all_and_hash(stream) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    blocks = []
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        blocks.append(block)
    return b"".join(blocks), digest.hexdigest()


def hash_only(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def actor_positions(row: dict) -> tuple[float | None, list[float]]:
    actors = ((row.get("state") or {}).get("actors") or [])
    deskbin_z = None
    garbage_z = []
    for actor in actors:
        pose = actor.get("pose") or []
        if len(pose) < 3:
            continue
        if actor.get("name") == "063_tabletrashbin":
            deskbin_z = float(pose[2])
        elif actor.get("name") == "garbage":
            garbage_z.append(float(pose[2]))
    return deskbin_z, garbage_z


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


def trace_summary(compressed: bytes, sustained_window: int = 50) -> dict:
    content_digest = hashlib.sha256()
    measured = []
    logged_steps = 0
    contact_steps = 0
    no_contact_run = 0
    first_sustained_loss = None
    initial_bin_z = None
    max_bin_z = None
    final_bin_z = None
    max_garbage_in_band = 0
    final_garbage_in_band = None
    final_garbage_count = None
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
        for raw_line in stream:
            encoded = raw_line.rstrip(b"\n")
            content_digest.update(encoded)
            row = json.loads(encoded)
            logged_steps += 1
            require(int(row.get("physics_step")) == logged_steps, "non-contiguous physics trace")
            value = (row.get("contact") or {}).get("measured_squeeze_n")
            if value is None:
                value = (row.get("original_squeeze_inner") or {}).get("measured_filtered_N")
            if value is not None:
                measured.append(float(value))
            if has_target_contact(row):
                contact_steps += 1
                no_contact_run = 0
            else:
                no_contact_run += 1
                if no_contact_run >= sustained_window and first_sustained_loss is None:
                    first_sustained_loss = logged_steps - sustained_window + 1
            deskbin_z, garbage_z = actor_positions(row)
            if deskbin_z is not None:
                if initial_bin_z is None:
                    initial_bin_z = deskbin_z
                max_bin_z = deskbin_z if max_bin_z is None else max(max_bin_z, deskbin_z)
                final_bin_z = deskbin_z
            if garbage_z:
                count_in_band = sum(0.13 <= value <= 0.25 for value in garbage_z)
                max_garbage_in_band = max(max_garbage_in_band, count_in_band)
                final_garbage_in_band = count_in_band
                final_garbage_count = len(garbage_z)
    require(logged_steps > 0 and measured, "empty trace or squeeze telemetry")
    require(initial_bin_z is not None and max_bin_z is not None and final_bin_z is not None,
            "deskbin state missing from trace")
    require(final_garbage_count == 5, "expected five garbage actors in final trace state")
    official_final_check = bool(final_bin_z > 1.0 and final_garbage_in_band == 5)
    return {
        "logged_steps": logged_steps,
        "trace_content_sha256": content_digest.hexdigest(),
        "measured_mean_squeeze_N": statistics.fmean(measured),
        "measured_max_squeeze_N": max(measured),
        "target_contact_fraction": contact_steps / logged_steps,
        "sustained_contact_loss_50_steps": first_sustained_loss is not None,
        "first_sustained_contact_loss_step": first_sustained_loss,
        "initial_bin_z_m": initial_bin_z,
        "max_bin_z_m": max_bin_z,
        "final_bin_z_m": final_bin_z,
        "ever_lifted_4cm": max_bin_z >= initial_bin_z + 0.04,
        "ever_bin_above_1m": max_bin_z > 1.0,
        "max_garbage_count_in_success_band": max_garbage_in_band,
        "final_garbage_count_in_success_band": final_garbage_in_band,
        "official_final_check_recomputed": official_final_check,
    }


def audit_archive(expectation: dict, expected_record: dict) -> dict:
    context_id = expectation["context_id"]
    archive_path = Path(expectation["remote_path"])
    require(archive_path.exists(), f"{context_id}: archive missing")
    require(archive_path.stat().st_size == int(expectation["archive_bytes"]), f"{context_id}: byte count changed")
    root_prefix = f"{context_id}/"
    member_hashes = {}
    json_payloads = {}
    traces = {}
    regular_members = 0

    with archive_path.open("rb") as raw:
        hashing_raw = HashingReader(raw)
        with tarfile.open(fileobj=hashing_raw, mode="r|gz") as archive:
            for member in archive:
                require(member.name == context_id or member.name.startswith(root_prefix),
                        f"{context_id}: tar member escapes context root")
                if not member.isfile():
                    continue
                regular_members += 1
                relative = str(PurePosixPath(member.name).relative_to(context_id))
                stream = archive.extractfile(member)
                require(stream is not None, f"{context_id}: cannot extract {relative}")
                keep_json = (
                    relative == "ARCHIVE_FILE_MANIFEST.json"
                    or relative == "INDEPENDENT_CONTEXT_AUDIT.json"
                    or relative == "CASE_COMPACT.json"
                    or relative == "job/FORMAL_CONTEXT_RESULT.json"
                    or relative.endswith("/result.json")
                )
                if relative.endswith("/physics_trace.jsonl.gz") or keep_json:
                    data, digest = read_all_and_hash(stream)
                    member_hashes[relative] = digest
                    if relative.endswith("/physics_trace.jsonl.gz"):
                        traces[relative] = trace_summary(data)
                    else:
                        json_payloads[relative] = json.loads(data)
                else:
                    member_hashes[relative] = hash_only(stream)
        # Ensure any bytes after the tar end marker/trailer also enter the raw hash.
        for _ in iter(lambda: hashing_raw.read(1024 * 1024), b""):
            pass
        archive_digest = hashing_raw.digest.hexdigest()
        archive_bytes = hashing_raw.bytes

    require(archive_digest == expectation["archive_sha256"], f"{context_id}: archive SHA256 mismatch")
    require(archive_bytes == int(expectation["archive_bytes"]), f"{context_id}: hashed byte count mismatch")
    manifest = json_payloads["ARCHIVE_FILE_MANIFEST.json"]
    manifest_bytes_hash = member_hashes["ARCHIVE_FILE_MANIFEST.json"]
    require(manifest_bytes_hash == expectation["source_file_manifest_sha256"],
            f"{context_id}: source manifest hash mismatch")
    require(manifest["file_count"] + 1 == int(expectation["source_file_count"]),
            f"{context_id}: receipt source file count mismatch")
    require(regular_members == int(expectation["source_file_count"]), f"{context_id}: tar member count mismatch")
    require(set(manifest["files"]) == set(member_hashes) - {"ARCHIVE_FILE_MANIFEST.json"},
            f"{context_id}: archive member membership differs from manifest")
    for relative, metadata in manifest["files"].items():
        require(member_hashes[relative] == metadata["sha256"], f"{context_id}: member hash mismatch: {relative}")

    archived_compact = json_payloads["CASE_COMPACT.json"]
    require(archived_compact == expected_record, f"{context_id}: archived compact record differs")
    raw_audit = json_payloads["INDEPENDENT_CONTEXT_AUDIT.json"]
    require(raw_audit.get("passed") is True, f"{context_id}: archived independent context audit false")
    require(raw_audit.get("first_chunk_pairing_exact") is True, f"{context_id}: archived pairing false")
    require(raw_audit.get("common_handoff_state_sha256") == expected_record["common_handoff_state_sha256"],
            f"{context_id}: handoff hash differs")
    require(raw_audit.get("formal_context_result_sha256") == expected_record["formal_context_result_sha256"],
            f"{context_id}: raw formal-result hash differs")

    outcomes = {row["method"]: row for row in expected_record["outcomes"]}
    branch_summaries = []
    for index, method in enumerate(expected_record["context"]["methods"]):
        prefix = f"job/branch_{index}_"
        result_names = [name for name in json_payloads if name.startswith(prefix) and name.endswith("/result.json")]
        trace_names = [name for name in traces if name.startswith(prefix) and name.endswith("/physics_trace.jsonl.gz")]
        require(len(result_names) == len(trace_names) == 1, f"{context_id}/{method}: branch artifacts missing")
        result = json_payloads[result_names[0]]
        trace = traces[trace_names[0]]
        compact = outcomes[method]
        require(result["method"] == method, f"{context_id}/{method}: result method mismatch")
        require(bool(result["completed"]), f"{context_id}/{method}: branch incomplete")
        require(bool(result["success"]) == bool(result["official_final_check"]),
                f"{context_id}/{method}: result label inconsistent")
        require(bool(result["success"]) == bool(compact["success"]), f"{context_id}/{method}: compact label differs")
        require(bool(result["success"]) == trace["official_final_check_recomputed"],
                f"{context_id}/{method}: raw final-state success differs")
        require(int(result["physics_steps"]) == trace["logged_steps"] == int(compact["physics_steps"]),
                f"{context_id}/{method}: physics-step count differs")
        require(result["trace_sha256"] == trace["trace_content_sha256"] == compact["trace_sha256"],
                f"{context_id}/{method}: trace content hash differs")
        require(math.isclose(float(result["measured_mean_squeeze_n"]), trace["measured_mean_squeeze_N"],
                             rel_tol=0.0, abs_tol=1e-12), f"{context_id}/{method}: mean squeeze differs")
        require(math.isclose(float(result["measured_max_squeeze_n"]), trace["measured_max_squeeze_N"],
                             rel_tol=0.0, abs_tol=1e-12), f"{context_id}/{method}: max squeeze differs")
        branch_audit = raw_audit["branches"][index]
        require(branch_audit["method"] == method and branch_audit["audit"].get("passed") is True,
                f"{context_id}/{method}: archived raw branch audit false")
        branch_summaries.append({
            "method": method,
            "success": int(bool(result["success"])),
            "descriptive_drop_failure": bool(not result["success"] and trace["sustained_contact_loss_50_steps"]),
            **trace,
        })

    return {
        "context_id": context_id,
        "archive_path": str(archive_path),
        "archive_sha256_recomputed": archive_digest,
        "archive_bytes_recomputed": archive_bytes,
        "all_member_hashes_verified": True,
        "raw_context_audit_verified": True,
        "branch_summaries": branch_summaries,
    }


def summarize(results: list[dict]) -> dict:
    by_method = {}
    for method in METHODS:
        rows = [
            next(row for row in result["branch_summaries"] if row["method"] == method)
            for result in results
        ]
        by_method[method] = {
            "rollouts": len(rows),
            "successes_recomputed": sum(row["success"] for row in rows),
            "sustained_contact_loss_50_steps": sum(row["sustained_contact_loss_50_steps"] for row in rows),
            "descriptive_drop_failures": sum(row["descriptive_drop_failure"] for row in rows),
            "ever_lifted_4cm": sum(row["ever_lifted_4cm"] for row in rows),
            "ever_bin_above_1m": sum(row["ever_bin_above_1m"] for row in rows),
            "ever_all_five_garbage_in_success_band": sum(
                row["max_garbage_count_in_success_band"] == 5 for row in rows
            ),
            "mean_target_contact_fraction": statistics.fmean(row["target_contact_fraction"] for row in rows),
            "mean_max_bin_z_m": statistics.fmean(row["max_bin_z_m"] for row in rows),
        }
    return by_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expectations", type=Path, help="JSON emitted by final_independent_audit.py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    core = json.loads(args.expectations.read_text())
    require(core.get("passed") is True, "core independent audit did not pass")
    records = {record["context"]["id"]: record for record in core["records"]}
    expectations = core["archive_expectations"]
    require(len(records) == len(expectations) == 24, "expected 24 archive expectations")
    results = [audit_archive(item, records[item["context_id"]]) for item in expectations]
    output = {
        "audit_version": "ANVIL_FORMAL_TRACE_AUDIT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "archive_count": len(results),
        "rollout_count": sum(len(result["branch_summaries"]) for result in results),
        "all_archive_sha256_recomputed": True,
        "all_archive_members_verified_against_manifests": True,
        "all_72_trace_hashes_and_outcomes_recomputed": True,
        "contact_loss_definition": "at least 50 consecutive logged physics steps without qualifying target contact",
        "drop_failure_definition": "descriptive only: official task failure plus sustained-contact-loss flag",
        "intermediate_metrics_posthoc_descriptive": True,
        "summary": summarize(results),
        "contexts": results,
    }
    encoded = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
