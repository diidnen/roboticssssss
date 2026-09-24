#!/usr/bin/env python3
"""Freeze and audit the accepted prefix of an interrupted Mass formal run.

The source run is treated as immutable.  A snapshot is assembled in a sibling
staging directory, verified against the CSV-defined grain, and atomically
renamed into place only after all hashes and cross-file checks pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


FORCES = (0.5, 1.0, 1.5, 2.5, 4.0)
MASS_BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}
ACCEPTED_CONTEXT_IDS = (
    "mass_structured_train_t2_root8100_low",
    "mass_structured_train_t2_root8100_mid",
    "mass_structured_train_t2_root8100_high",
    "mass_structured_train_t2_root8101_low",
    "mass_structured_train_t2_root8101_mid",
    "mass_structured_train_t2_root8101_high",
    "mass_structured_train_t2_root8102_low",
    "mass_structured_train_t2_root8102_mid",
    "mass_structured_train_t2_root8102_high",
    "mass_structured_train_t2_root8103_low",
)
TELEMETRY_RE = re.compile(
    r"^(?P<context>.+)_F(?P<force>0\.5|1|1\.5|2\.5|4)_R(?P<repeat>[01])\.csv$"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_fields(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_copy(source: Path, destination: Path) -> dict:
    before = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied = sha256(destination)
    after = sha256(source)
    if before != copied or before != after:
        raise RuntimeError(f"source changed during copy: {source}")
    return {
        "source": str(source),
        "snapshot": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": copied,
    }


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def telemetry_name(row: dict[str, str]) -> str:
    return (
        f"{row['context_id']}_F{float(row['requested_force_N']):g}"
        f"_R{as_int(row, 'repeat')}.csv"
    )


def corrected_full(row: dict[str, str]) -> int:
    return int(
        as_int(row, "lift_success") == 1
        and as_int(row, "transport_retention") == 1
        and as_int(row, "place_success") == 1
        and as_int(row, "dropped") == 0
    )


def rate(rows: list[dict[str, str]], key: str) -> float | None:
    return sum(as_int(row, key) for row in rows) / len(rows) if rows else None


def summarize(label: str, rows: list[dict[str, str]]) -> dict:
    corrected = [corrected_full(row) for row in rows]
    delayed = [
        int(as_int(row, "lift_success") == 1 and corrected_full(row) == 0)
        for row in rows
    ]
    return {
        "group": label,
        "branches": len(rows),
        "contexts": len({row["context_id"] for row in rows}),
        "root_families": len({row["root_family"] for row in rows}),
        "lift_success_rate": rate(rows, "lift_success"),
        "transport_retention_rate": rate(rows, "transport_retention"),
        "placement_success_rate": rate(rows, "place_success"),
        "corrected_full_task_success_rate": sum(corrected) / len(rows) if rows else None,
        "post_lift_failure_rate": sum(delayed) / len(rows) if rows else None,
        "mean_requested_force_N": (
            sum(as_float(row, "requested_force_N") for row in rows) / len(rows)
            if rows
            else None
        ),
    }


def audit_source(source: Path) -> dict:
    contexts_path = next(source.glob("M3_TASK*_STRUCTURED_FORMAL_CONTEXTS.csv"))
    branches_path = next(source.glob("M3_TASK*_STRUCTURED_FORMAL_BRANCHES.csv"))
    steps_path = next(source.glob("M3_TASK*_STRUCTURED_FORMAL_QUERY_TIMESTEPS.csv"))
    observations_path = next(source.glob("M3_TASK*_STRUCTURED_FORMAL_QUERY_OBSERVATIONS.json"))
    protocol_path = next(source.glob("M3_TASK*_STRUCTURED_FORMAL_PROTOCOL.json"))
    controlling_paths = {
        "contexts": contexts_path,
        "branches": branches_path,
        "steps": steps_path,
        "observations": observations_path,
        "protocol": protocol_path,
    }
    controlling_hashes = {key: sha256(path) for key, path in controlling_paths.items()}

    source_contexts = read_csv(contexts_path)
    source_branches = read_csv(branches_path)
    source_steps = read_csv(steps_path)
    source_observations = json.loads(observations_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    accepted_set = set(ACCEPTED_CONTEXT_IDS)
    contexts = [row for row in source_contexts if row["context_id"] in accepted_set]
    branches = [row for row in source_branches if row["context_id"] in accepted_set]
    steps = [row for row in source_steps if row["context_id"] in accepted_set]
    observations = [
        row for row in source_observations if str(row["context_id"]) in accepted_set
    ]
    if len(contexts) != 10 or len(branches) != 100:
        raise RuntimeError(
            f"refusing unexpected accepted prefix: {len(contexts)} contexts, "
            f"{len(branches)} branches"
        )
    context_ids = [row["context_id"] for row in contexts]
    accepted = set(context_ids)
    if tuple(context_ids) != ACCEPTED_CONTEXT_IDS:
        raise RuntimeError("source no longer contains the exact accepted context order")
    if len(accepted) != len(context_ids):
        raise RuntimeError("duplicate accepted context_id")
    if any(as_int(row, "query_valid") != 1 for row in contexts):
        raise RuntimeError("accepted prefix contains an invalid query context")

    branch_key_counts = Counter(
        (
            row["context_id"],
            as_float(row, "requested_force_N"),
            as_int(row, "repeat"),
        )
        for row in branches
    )
    if any(count != 1 for count in branch_key_counts.values()):
        raise RuntimeError("duplicate branch composite key")
    if {row["context_id"] for row in branches} != accepted:
        raise RuntimeError("branch/context referential integrity failed")

    by_context: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in branches:
        by_context[row["context_id"]].append(row)
    expected_grid = {(force, repeat) for force in FORCES for repeat in (0, 1)}
    for context in contexts:
        cid = context["context_id"]
        rows = by_context[cid]
        grid = {(as_float(row, "requested_force_N"), as_int(row, "repeat")) for row in rows}
        if len(rows) != 10 or grid != expected_grid:
            raise RuntimeError(f"incomplete force/repeat grid: {cid}")
        for row in rows:
            for field in (
                "task_id",
                "root_seed",
                "split",
                "mass_band",
                "mass_kg",
                "friction",
                "initial_state_hash",
                "post_query_state_hash",
                "query_valid",
                "path_structure",
            ):
                if str(row[field]) != str(context[field]):
                    raise RuntimeError(f"context/branch mismatch {cid}: {field}")

    steps_by_context = Counter(row["context_id"] for row in steps)
    expected_step_counts = {row["context_id"]: as_int(row, "query_history_rows") for row in contexts}
    if set(steps_by_context) != accepted or dict(steps_by_context) != expected_step_counts:
        raise RuntimeError("query timestep/context completeness failed")

    observation_ids = [str(row["context_id"]) for row in observations]
    if len(observation_ids) != 10 or set(observation_ids) != accepted or len(set(observation_ids)) != 10:
        raise RuntimeError("query observation/context completeness failed")

    telemetry_checks = []
    accepted_telemetry = set()
    for branch in branches:
        path = source / "telemetry" / telemetry_name(branch)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing accepted telemetry: {path.name}")
        rows = read_csv(path)
        if len(rows) != as_int(branch, "steps"):
            raise RuntimeError(f"telemetry/branch step mismatch: {path.name}")
        expected = {
            "context_id": branch["context_id"],
            "task": branch["task_id"],
            "seed": branch["root_seed"],
            "mass_band": branch["mass_band"],
        }
        for field, value in expected.items():
            if any(str(row[field]) != str(value) for row in rows):
                raise RuntimeError(f"telemetry identity mismatch {path.name}: {field}")
        if any(abs(float(row["force"]) - as_float(branch, "requested_force_N")) > 1e-9 for row in rows):
            raise RuntimeError(f"telemetry force mismatch: {path.name}")
        if [as_int(row, "step") for row in rows] != list(range(1, len(rows) + 1)):
            raise RuntimeError(f"telemetry steps are not contiguous: {path.name}")
        accepted_telemetry.add(path)
        telemetry_checks.append({"file": path.name, "rows": len(rows)})

    all_telemetry = set((source / "telemetry").glob("*.csv"))
    orphan_telemetry = sorted(all_telemetry - accepted_telemetry)
    all_rgb = set((source / "query_rgb").glob("*.npy"))
    accepted_rgb = {
        path
        for path in all_rgb
        if any(path.name.startswith(f"{context_id}_") for context_id in accepted)
    }
    expected_rgb_names = {
        f"{context_id}_{camera}.npy"
        for context_id in accepted
        for camera in ("agentview_cam", "eye_in_hand_cam")
    }
    if {path.name for path in accepted_rgb} != expected_rgb_names:
        raise RuntimeError("accepted RGB/context completeness failed")
    orphan_rgb = sorted(all_rgb - accepted_rgb)

    orphan_contexts = set()
    for path in orphan_telemetry:
        match = TELEMETRY_RE.match(path.name)
        if not match:
            raise RuntimeError(f"unparseable orphan telemetry name: {path.name}")
        orphan_contexts.add(match.group("context"))
    for path in orphan_rgb:
        for suffix in ("_agentview_cam.npy", "_eye_in_hand_cam.npy"):
            if path.name.endswith(suffix):
                orphan_contexts.add(path.name[: -len(suffix)])
                break
        else:
            raise RuntimeError(f"unparseable orphan RGB name: {path.name}")
    allowed_unaccepted_contexts = {
        "mass_structured_train_t2_root8103_mid",
        "mass_structured_train_t2_root8103_high",
        "mass_structured_test_t2_root8200_low",
        "mass_structured_test_t2_root8200_mid",
        "mass_structured_test_t2_root8200_high",
        "mass_structured_test_t2_root8201_low",
        "mass_structured_test_t2_root8201_mid",
        "mass_structured_test_t2_root8201_high",
    }
    if (
        "mass_structured_train_t2_root8103_mid" not in orphan_contexts
        or not orphan_contexts <= allowed_unaccepted_contexts
    ):
        raise RuntimeError(f"unexpected orphan contexts: {sorted(orphan_contexts)}")

    quarantine_contexts = [
        row
        for row in source_contexts
        if row["context_id"] in orphan_contexts
    ]
    quarantine_branches = [
        row
        for row in source_branches
        if row["context_id"] in orphan_contexts
    ]
    quarantine_steps = [
        row
        for row in source_steps
        if row["context_id"] in orphan_contexts
    ]
    quarantine_observations = [
        row
        for row in source_observations
        if str(row["context_id"]) in orphan_contexts
    ]

    expected_contexts = [
        f"mass_structured_train_t2_root{root}_{band.lower()}"
        for root in protocol["train_roots"]
        for band in MASS_BANDS
    ] + [
        f"mass_structured_test_t2_root{root}_{band.lower()}"
        for root in protocol["test_roots"]
        for band in MASS_BANDS
    ]
    remaining = [context_id for context_id in expected_contexts if context_id not in accepted]
    if len(expected_contexts) != 18 or len(remaining) != 8:
        raise RuntimeError("unexpected formal protocol coverage")
    for key, path in controlling_paths.items():
        if sha256(path) != controlling_hashes[key]:
            raise RuntimeError(f"controlling source changed during audit: {path}")

    return {
        "paths": controlling_paths,
        "controlling_hashes": controlling_hashes,
        "contexts": contexts,
        "branches": branches,
        "steps": steps,
        "observations": observations,
        "quarantine_contexts": quarantine_contexts,
        "quarantine_branches": quarantine_branches,
        "quarantine_steps": quarantine_steps,
        "quarantine_observations": quarantine_observations,
        "protocol": protocol,
        "accepted_context_ids": context_ids,
        "accepted_telemetry": sorted(accepted_telemetry),
        "accepted_rgb": sorted(accepted_rgb),
        "orphan_telemetry": orphan_telemetry,
        "orphan_rgb": orphan_rgb,
        "orphan_context_ids": sorted(orphan_contexts),
        "remaining_context_ids": remaining,
        "telemetry_checks": telemetry_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--collector-python",
        default="/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)

    audit = audit_source(source)
    accepted = staging / "accepted_snapshot"
    quarantine = staging / "quarantine" / "unaccepted_after_100_branch_cutoff"
    copied = []
    try:
        copied.append(
            stable_copy(audit["paths"]["protocol"], accepted / "SOURCE_PROTOCOL.json")
        )
        write_csv(
            accepted / audit["paths"]["contexts"].name,
            audit["contexts"],
            csv_fields(audit["paths"]["contexts"]),
        )
        write_csv(
            accepted / audit["paths"]["branches"].name,
            audit["branches"],
            csv_fields(audit["paths"]["branches"]),
        )
        write_csv(
            accepted / audit["paths"]["steps"].name,
            audit["steps"],
            csv_fields(audit["paths"]["steps"]),
        )

        observations = audit["observations"]
        for observation in observations:
            for frame in observation.get("rgb_observation", {}).get("frames", {}).values():
                original = Path(frame["path"])
                remapped = accepted / "query_rgb" / original.name
                frame["source_path"] = str(original)
                frame["path"] = str(remapped)
        observations_path = accepted / audit["paths"]["observations"].name
        write_json(observations_path, observations)

        for src in audit["accepted_telemetry"]:
            copied.append(stable_copy(src, accepted / "telemetry" / src.name))
        for src in audit["accepted_rgb"]:
            copied.append(stable_copy(src, accepted / "query_rgb" / src.name))
        for src in audit["orphan_telemetry"]:
            copied.append(stable_copy(src, quarantine / "telemetry" / src.name))
        for src in audit["orphan_rgb"]:
            copied.append(stable_copy(src, quarantine / "query_rgb" / src.name))
        if audit["quarantine_contexts"]:
            write_csv(
                quarantine / "M3_TASK2_STRUCTURED_FORMAL_CONTEXTS.csv",
                audit["quarantine_contexts"],
                csv_fields(audit["paths"]["contexts"]),
            )
        if audit["quarantine_branches"]:
            write_csv(
                quarantine / "M3_TASK2_STRUCTURED_FORMAL_BRANCHES.csv",
                audit["quarantine_branches"],
                csv_fields(audit["paths"]["branches"]),
            )
        if audit["quarantine_steps"]:
            write_csv(
                quarantine / "M3_TASK2_STRUCTURED_FORMAL_QUERY_TIMESTEPS.csv",
                audit["quarantine_steps"],
                csv_fields(audit["paths"]["steps"]),
            )
        if audit["quarantine_observations"]:
            write_json(
                quarantine / "M3_TASK2_STRUCTURED_FORMAL_QUERY_OBSERVATIONS.json",
                audit["quarantine_observations"],
            )

        # Resume-safe protocol is separate from the stale source RUNNING marker.
        recovered_protocol = dict(audit["protocol"])
        recovered_protocol.update(
            {
                "status": "RECOVERED_ACCEPTED_PREFIX_READY_FOR_RESUME",
                "contexts": 10,
                "branches": 100,
                "query_timesteps": len(audit["steps"]),
                "source_protocol_status": audit["protocol"].get("status"),
                "source_directory": str(source),
                "orphan_material_excluded": True,
            }
        )
        write_json(accepted / "M3_TASK2_STRUCTURED_FORMAL_PROTOCOL.json", recovered_protocol)

        resume_context_rows = []
        for ordinal, context_id in enumerate(audit["remaining_context_ids"], 1):
            split = "TEST" if "_test_" in context_id else "TRAIN"
            root = int(re.search(r"root(\d+)", context_id).group(1))
            band = context_id.rsplit("_", 1)[1].upper()
            resume_context_rows.append(
                {
                    "resume_order": ordinal,
                    "context_id": context_id,
                    "split": split,
                    "root_seed": root,
                    "mass_band": band,
                    "mass_kg": MASS_BANDS[band],
                    "branches_expected": 10,
                }
            )
        write_csv(
            staging / "RESUME_CONTEXTS.csv",
            resume_context_rows,
            ["resume_order", "context_id", "split", "root_seed", "mass_band", "mass_kg", "branches_expected"],
        )

        resume_output = output / "resume_output"
        resume_staging = output / ".resume_output.staging"
        preparation = [
            [
                "cp",
                "-a",
                str(output / "accepted_snapshot"),
                str(resume_staging),
            ],
            ["mv", str(resume_staging), str(resume_output)],
        ]
        command = [
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            args.collector_python,
            "-u",
            "/home/exouser/FORTE_mass/collect_mass_structured_formal.py",
            "--task",
            "2",
            "--out",
            str(resume_output),
            "--train-roots",
            "8100",
            "8101",
            "8102",
            "8103",
            "--test-roots",
            "8200",
            "8201",
            "--repeats",
            "2",
            "--resume-from",
            str(output / "accepted_snapshot"),
        ]
        post_resume_cpu_command = [
            args.collector_python,
            "/home/exouser/FORTE_mass/analyze_mass_formal.py",
            "--inputs",
            str(resume_output),
            "--out",
            str(output / "post_resume_cpu_analysis"),
        ]
        resume_manifest = {
            "status": "READY_NOT_LAUNCHED_GPU_PREREQUISITE_REQUIRED",
            "source_is_protected_and_untouched": True,
            "source_directory": str(source),
            "accepted_snapshot": str(output / "accepted_snapshot"),
            "resume_output_must_be_new": str(resume_output),
            "resume_staging_must_be_new": str(resume_staging),
            "accepted_context_count": 10,
            "accepted_branch_count": 100,
            "expected_final_context_count": 18,
            "expected_final_branch_count": 180,
            "remaining_context_count": 8,
            "remaining_branch_count": 80,
            "remaining_contexts_in_collector_order": resume_context_rows,
            "quarantined_contexts": audit["orphan_context_ids"],
            "quarantined_telemetry_files": len(audit["orphan_telemetry"]),
            "quarantined_rgb_files": len(audit["orphan_rgb"]),
            "atomic_preseed_argv": preparation,
            "exact_argv": command,
            "deferred_post_resume_cpu_argv": post_resume_cpu_command,
            "working_directory": "/home/exouser/FORTE_massresume",
            "launch_policy": "DO_NOT_LAUNCH_UNTIL_GPU_AUTHORIZED_AND_AVAILABLE",
            "post_resume_acceptance": {
                "protocol_status": "COMPLETED",
                "contexts": 18,
                "branches": 180,
                "query_timesteps": 3708,
                "telemetry_files": 180,
                "rgb_files": 36,
                "all_contexts_query_valid": True,
                "each_context_force_repeat_grid": "5 forces x 2 repeats",
            },
        }
        write_json(staging / "EXACT_RESUME_MANIFEST.json", resume_manifest)

        quarantine_manifest = {
            "status": "QUARANTINED_COPY_SOURCE_UNTOUCHED",
            "reason": (
                "context was partial at the explicit 10-context/100-branch recovery cutoff; "
                "late source commits are excluded from the accepted snapshot"
            ),
            "context_ids": audit["orphan_context_ids"],
            "telemetry_files": [path.name for path in audit["orphan_telemetry"]],
            "rgb_files": [path.name for path in audit["orphan_rgb"]],
            "late_committed_context_rows": len(audit["quarantine_contexts"]),
            "late_committed_branch_rows": len(audit["quarantine_branches"]),
            "late_committed_query_timestep_rows": len(audit["quarantine_steps"]),
            "scientific_use": "NONE",
            "resume_behavior": "regenerate entire context from its query",
        }
        write_json(quarantine / "QUARANTINE_MANIFEST.json", quarantine_manifest)

        analysis_rows = [summarize("OVERALL_ACCEPTED_TRAIN_PREFIX", audit["branches"])]
        for band in MASS_BANDS:
            analysis_rows.append(
                summarize(
                    f"MASS_{band}",
                    [row for row in audit["branches"] if row["mass_band"] == band],
                )
            )
        for force in FORCES:
            analysis_rows.append(
                summarize(
                    f"FORCE_{force:g}N",
                    [
                        row
                        for row in audit["branches"]
                        if abs(as_float(row, "requested_force_N") - force) < 1e-9
                    ],
                )
            )
        write_csv(
            staging / "cpu_analysis" / "TRAIN_PREFIX_OUTCOME_SUMMARY.csv",
            analysis_rows,
            list(analysis_rows[0]),
        )
        full_vs_lift = []
        for row in audit["branches"]:
            full_vs_lift.append(
                {
                    "context_id": row["context_id"],
                    "root_seed": row["root_seed"],
                    "mass_band": row["mass_band"],
                    "requested_force_N": row["requested_force_N"],
                    "repeat": row["repeat"],
                    "lift_success": as_int(row, "lift_success"),
                    "transport_retention": as_int(row, "transport_retention"),
                    "placement_success": as_int(row, "place_success"),
                    "corrected_full_task_success": corrected_full(row),
                    "post_lift_failure": int(
                        as_int(row, "lift_success") == 1 and corrected_full(row) == 0
                    ),
                }
            )
        write_csv(
            staging / "cpu_analysis" / "TRAIN_PREFIX_FULLTASK_VS_LOCALLIFT.csv",
            full_vs_lift,
            list(full_vs_lift[0]),
        )
        analysis = {
            "status": "COMPLETED_TRAIN_PREFIX_ONLY",
            "formal_test_analysis_status": "BLOCKED_NO_TEST_CONTEXTS_ACCEPTED",
            "accepted_train_contexts": 10,
            "accepted_train_branches": 100,
            "accepted_test_contexts": 0,
            "accepted_test_branches": 0,
            "interpretation_boundary": (
                "descriptive CPU analysis only; no formal policy comparison, "
                "root-heldout TEST claim, or final mass-adaptation claim"
            ),
            "summaries": analysis_rows,
        }
        write_json(staging / "cpu_analysis" / "DOWNSTREAM_CPU_ANALYSIS.json", analysis)

        overall = analysis_rows[0]
        report_lines = [
            "# Mass formal recovery CPU analysis",
            "",
            "Status: **COMPLETED_TRAIN_PREFIX_ONLY**.",
            "",
            "The accepted prefix contains 10 TRAIN contexts and 100 paired force branches. "
            "There are no accepted TEST contexts, so formal policy adaptation and locked-test "
            "claims remain blocked until the exact resume completes.",
            "",
            "| scope | branches | contexts | lift SR | transport SR | placement SR | corrected full SR | post-lift failure |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in analysis_rows:
            report_lines.append(
                f"| {row['group']} | {row['branches']} | {row['contexts']} | "
                f"{row['lift_success_rate']:.3f} | {row['transport_retention_rate']:.3f} | "
                f"{row['placement_success_rate']:.3f} | "
                f"{row['corrected_full_task_success_rate']:.3f} | "
                f"{row['post_lift_failure_rate']:.3f} |"
            )
        report_lines += [
            "",
            "The accepted CSVs, their 100 telemetry files, 10 query histories, and 20 RGB "
            "arrays passed uniqueness, completeness, identity, contiguous-step, and hash checks.",
            "",
            f"Overall corrected full-task SR is {overall['corrected_full_task_success_rate']:.3f}; "
            f"post-lift failures occur on {overall['post_lift_failure_rate']:.3f} of branches. "
            "These are TRAIN-prefix descriptions only.",
        ]
        (staging / "cpu_analysis" / "DOWNSTREAM_CPU_ANALYSIS.md").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )

        recovery_status = {
            "status": "CPU_RECOVERY_COMPLETE_GPU_RESUME_NOT_LAUNCHED",
            "atomic_snapshot": True,
            "accepted_contexts": 10,
            "accepted_branches": 100,
            "accepted_query_timesteps": len(audit["steps"]),
            "accepted_telemetry_files": len(audit["accepted_telemetry"]),
            "accepted_rgb_files": len(audit["accepted_rgb"]),
            "quarantined_contexts": audit["orphan_context_ids"],
            "quarantined_telemetry_files": len(audit["orphan_telemetry"]),
            "quarantined_rgb_files": len(audit["orphan_rgb"]),
            "source_protocol_status": audit["protocol"].get("status"),
            "source_untouched": True,
            "formal_test_analysis": "BLOCKED_PENDING_8_CONTEXT_GPU_RESUME",
            "cpu_train_prefix_analysis": "COMPLETED",
        }
        write_json(staging / "RECOVERY_STATUS.json", recovery_status)

        # Hash every durable evidence file except the hash manifest itself.
        evidence_files = sorted(
            path
            for path in staging.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS.json"
        )
        hash_manifest = {
            str(path.relative_to(staging)): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in evidence_files
        }
        write_json(staging / "SHA256SUMS.json", hash_manifest)

        # Recheck controlling source files immediately before publication.
        for key, src in audit["paths"].items():
            if sha256(src) != audit["controlling_hashes"][key]:
                raise RuntimeError(f"controlling source changed before commit: {src}")
        if set((source / "telemetry").glob("*.csv")) != set(audit["accepted_telemetry"]) | set(audit["orphan_telemetry"]):
            raise RuntimeError("source telemetry inventory changed before commit")
        if set((source / "query_rgb").glob("*.npy")) != set(audit["accepted_rgb"]) | set(audit["orphan_rgb"]):
            raise RuntimeError("source RGB inventory changed before commit")
        if sha256(audit["paths"]["observations"]) != audit["controlling_hashes"]["observations"]:
            raise RuntimeError("source observations changed before commit")

        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(json.dumps({"status": "COMPLETED", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
