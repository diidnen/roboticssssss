#!/usr/bin/env python3
"""Finalize the fail-closed old720 execution-contract restoration audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path


OUT = Path("/home/exouser/FORTE/analysis/results/old720_exact_execution_restoration_20260905")
HIST_ROOT = Path("/home/exouser/FORTE/gnp_style_continuous_20260830_125107/collection_long2")
FRESH_ROOT = OUT / "parity_run_2"
CID = "p5s0c_train_t0_r00_s5100_low_mu0.293710"
HIST_PREPROBE_HASH = "18bde4e4fc96af4e36561c7170328947a96c9685799131871039e013de43e899"


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def metric(values: list[float]) -> dict:
    k = max(1, math.ceil(0.05 * len(values)))
    return {
        "samples": len(values),
        "mean_N": statistics.fmean(values),
        "top5_mean_N": statistics.fmean(sorted(values)[-k:]),
        "peak_N": max(values),
        "exposure_Ns": sum(values) * 0.05,
    }


def trace_metric(path: Path) -> dict:
    rows = read_csv(path)
    common = [r for r in rows if r["phase"] in {"branch_hold", "lift"}]
    native = [float(r["measured_force_N"]) for r in common]
    true_sq = [2.0 * min(float(r["left_normal_force_N"]), float(r["right_normal_force_N"])) for r in common]
    return {
        "path": str(path),
        "row_count": len(rows),
        "common_window": "branch_hold+lift (70 environment steps, 3.5s)",
        "native_measured_squeeze": metric(native),
        "object_filtered_true_squeeze": metric(true_sq),
    }


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        average = 0.5 * (i + 1 + j)
        for k in range(i, j):
            result[order[k]] = average
        i = j
    return result


def pearson(x: list[float], y: list[float]) -> float:
    xm, ym = statistics.fmean(x), statistics.fmean(y)
    num = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    den = math.sqrt(sum((a - xm) ** 2 for a in x) * sum((b - ym) ** 2 for b in y))
    return num / den if den else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rank(x), rank(y))


def main() -> int:
    fresh_trace = next((FRESH_ROOT / "P5S0C_BRANCH_TELEMETRY").glob("*.csv"))
    shutil.copyfile(fresh_trace, OUT / "OLD720_HISTORICAL_BRANCH_REPLAY_TRACE.csv")
    fresh_capture = read_csv(FRESH_ROOT / "task0/strict_preprobe_capture.csv")[0]
    fresh_branch = read_csv(FRESH_ROOT / "task0/branches.csv")[0]
    hist_paths = sorted((HIST_ROOT / "P5S0C_BRANCH_TELEMETRY").glob(f"{CID}_GNP_*.csv"))
    selected_hist = [p for p in hist_paths if "GNP_S3_F4p19512689_R" in p.name]
    historical_selected = [trace_metric(p) for p in selected_hist]
    fresh_selected = trace_metric(fresh_trace)
    provenance_path = OUT / "OLD720_EXECUTION_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update({
        "launch_config": {
            "isaac_python": "/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python",
            "env_id": "Isaac-Libero-Franka-Hybrid-Tactile-v0",
            "physics_hz": 60.0,
            "environment_hz": 20.0,
            "original_pythonpath_contract": "Tabero-VTLA + /home/exouser/Tabero + openpi/src",
        },
        "action_adapter": "P4._make_action: 13D absolute EEF pose, d_pred in slot6, requested squeeze split equally into local-z force slots9/12",
        "reset_contract": (
            "env.reset(seed=root_seed); P4 run_probe_episode resets same seed and applies object friction; capture scene at "
            "P4 step190; finish probe; reset_to captured strict-preprobe state twice; capture branch snapshot; reset_to it per branch"
        ),
        "initial_state_source": "seeded scripted P4-B scene reset; no HDF5 demo state",
        "branch_start_point": "P4-B final hold step190 immediately before probe_out",
        "arm_trace_source": "scripted P5 Cartesian interpolation, not a VLA trajectory",
        "complete_runtime_image_pinned": False,
        "physical_replay_parity": False,
        "provenance_limit": (
            "80ab is a post-collection archive commit. It preserves the exact hashed P5/P4 sources, but the Aug-30 run "
            "did not archive a restorable scene snapshot or a complete Isaac/contact-runtime image."
        ),
    })
    write_json(provenance_path, provenance)
    parity = {
        "status": "FAIL",
        "branch_replay_parity": False,
        "gate_consequence": "setpoint sweep and root7703 integration were not run",
        "reference_branch": "GNP_S3_F4p19512689_R1",
        "requested_setpoint": 4.1951268916184405,
        "source_hash_gate": "PASS",
        "root_initial_observable_hash_parity": True,
        "historical_strict_preprobe_hash": HIST_PREPROBE_HASH,
        "fresh_strict_preprobe_hash": fresh_capture["preprobe_state_hash"],
        "strict_preprobe_hash_parity": fresh_capture["preprobe_state_hash"] == HIST_PREPROBE_HASH,
        "historical_probe_rows": 215,
        "fresh_probe_rows": len(read_csv(FRESH_ROOT / f"P5S0C_PROBE_TELEMETRY/{CID}_probe_timesteps.csv")),
        "historical_probe_stop": "max_displacement_cap",
        "fresh_probe_stop": "shear_ratio_cap",
        "historical_branch": historical_selected,
        "fresh_branch": fresh_selected,
        "historical_full_task_outcomes": [1, 1],
        "fresh_full_task_outcome": int(fresh_branch["full_task_success_y"]),
        "outcome_parity_only": int(fresh_branch["full_task_success_y"]) == 1,
        "diagnosis": (
            "Runner/P4/controller source hashes, task seed, initial observable hash, and branch logic were recovered, "
            "but the contact trajectory diverges during P4 close. The archive did not pin or preserve a restorable "
            "preprobe scene snapshot or a complete Isaac/contact-runtime dependency image, so exact physical replay is not recovered."
        ),
        "parity_rule": (
            "PASS requires source/config parity, strict preprobe state parity, matched force telemetry within the historical "
            "two-repeat envelope, and outcome parity. Outcome parity alone is insufficient."
        ),
    }
    write_json(OUT / "OLD720_HISTORICAL_BRANCH_REPLAY_PARITY.json", parity)

    by_force: dict[float, list[dict]] = defaultdict(list)
    for path in hist_paths:
        rows = read_csv(path)
        force = float(rows[0]["requested_force_N"])
        entry = trace_metric(path)
        entry["branch_label"] = rows[0]["branch_label"]
        by_force[force].append(entry)
    dataset_rows = read_csv(Path("/home/exouser/FORTE/gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv"))
    outcomes = defaultdict(list)
    for row in dataset_rows:
        if row["context_id"] == CID:
            outcomes[float(row["requested_force_N"])].append(int(float(row["full_task_success_y"])))
    groups = []
    for force in sorted(by_force):
        reps = by_force[force]
        matched_outcomes = [
            value
            for candidate, values in outcomes.items()
            if abs(candidate - force) < 1e-9
            for value in values
        ]
        groups.append({
            "setpoint": force,
            "repeat_count": len(reps),
            "historical_native_mean_force_N": statistics.fmean(r["native_measured_squeeze"]["mean_N"] for r in reps),
            "historical_native_top5_force_N": statistics.fmean(r["native_measured_squeeze"]["top5_mean_N"] for r in reps),
            "historical_native_force_exposure_Ns": statistics.fmean(r["native_measured_squeeze"]["exposure_Ns"] for r in reps),
            "historical_object_filtered_mean_force_N": statistics.fmean(r["object_filtered_true_squeeze"]["mean_N"] for r in reps),
            "historical_success_rate": statistics.fmean(matched_outcomes),
            "repeats": reps,
        })
    x = [g["setpoint"] for g in groups]
    native_mean = [g["historical_native_mean_force_N"] for g in groups]
    native_top5 = [g["historical_native_top5_force_N"] for g in groups]
    native_exp = [g["historical_native_force_exposure_Ns"] for g in groups]
    violations = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if native_mean[j] < native_mean[i]:
                violations.append([groups[i]["setpoint"], groups[j]["setpoint"]])
    mapping = {
        "evaluation_status": "NOT_RUN_BECAUSE_BRANCH_REPLAY_PARITY_FAILED",
        "historical_archive_mapping_ordered": True,
        "historical_archive_metrics_are_not_fresh_validation": True,
        "historical_setpoint_levels": x,
        "common_metric_window": "branch_hold+lift (70 environment steps, 3.5s)",
        "historical_group_summary": groups,
        "historical_native_setpoint_mean_force_spearman": spearman(x, native_mean),
        "historical_native_setpoint_top5_spearman": spearman(x, native_top5),
        "historical_native_setpoint_exposure_spearman": spearman(x, native_exp),
        "historical_pairwise_group_mean_monotonic_violations": violations,
        "reproduced_repeat_count": 0,
        "old720_continuous_setpoint_mapping_valid": False,
        "reason": "Archived traces are ordered, but the exact historical physical execution contract failed replay parity; no fresh sweep was authorized.",
    }
    write_json(OUT / "OLD720_SETPOINT_MAPPING.json", mapping)
    with (OUT / "OLD720_SETPOINT_SWEEP_TRACE_NOT_RUN.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["status", "reason"])
        writer.writeheader()
        writer.writerow({"status": "NOT_RUN", "reason": "OLD720_BRANCH_REPLAY_PARITY_FAILED"})

    mapping_report = f"""# old720 setpoint mapping report

## Answer

The archived reference context itself has an ordered five-level relationship: all three native-force summaries have Spearman rho `1.0`, and the five group means have zero pairwise inversions. This is historical evidence only. It is not a fresh validation because the single-branch replay gate failed before the sweep.

## Why the fresh sweep was stopped

- All hash-locked Python sources and the task's initial observable state matched.
- The P4 close trajectory diverged at contact: historical preload ended near `3.3131 N` at aperture `0.024370 m`; fresh ended near `2.8696 N` at `0.021999 m`.
- The strict-preprobe state hash changed from `{HIST_PREPROBE_HASH}` to `{fresh_capture['preprobe_state_hash']}`.
- For setpoint `4.1951268916`, historical branch-hold+lift native mean force was `14.8214/14.8468 N`; fresh was `{fresh_selected['native_measured_squeeze']['mean_N']:.4f} N`.
- Full-task success matched, but outcome parity cannot substitute for state and force-trajectory parity.

## Historical archive-only group trend

| Setpoint | Mean force (N) | Top-5% (N) | Exposure (N·s) | Success |
|---:|---:|---:|---:|---:|
"""
    for g in groups:
        mapping_report += f"| {g['setpoint']:.6f} | {g['historical_native_mean_force_N']:.4f} | {g['historical_native_top5_force_N']:.4f} | {g['historical_native_force_exposure_Ns']:.4f} | {g['historical_success_rate']:.2f} |\n"
    mapping_report += "\nThe required next action is not controller tuning. It is recovery of a restorable Aug-30 preprobe snapshot or a complete time-pinned Isaac/contact-runtime image. Without one of those, the archived mapping cannot be promoted to a reproduced final execution contract.\n"
    (OUT / "OLD720_SETPOINT_MAPPING_REPORT.md").write_text(mapping_report, encoding="utf-8")

    decision = {
        "final_status": "BLOCKED_EXACT_OLD720_PHYSICAL_REPLAY_PARITY_FAILED",
        "old720_exact_execution_contract_restored": False,
        "final_low_level_execution": "NONE",
        "final_execution_frozen": False,
        "old720_training_execution_match": False,
        "root7703_activeforcing_run": False,
        "final_force_semantics": "MEASURED_FORCE_BUDGET",
        "measured_force_supervision_pivot_needed": True,
        "current_blocker": "No restorable old720 preprobe snapshot or complete time-pinned Isaac/contact-runtime image; source-identical fresh contact dynamics do not reproduce the archived preprobe/force trace.",
        "next_action": "Recover an Aug-30 runtime image or saved scene snapshot. If unavailable, accept the fail-closed pivot to measured-force/exposure supervision; do not tune or invent another controller.",
    }
    write_json(OUT / "FINAL_EXECUTION_DECISION.json", decision)
    final_report = f"""# Final low-controller decision report

## Decision

`OLD720_EXECUTION_CONTRACT_RESTORED_EXACTLY = NO` and `OLD720_BRANCH_REPLAY_PARITY = NO`.

The code contract was recovered with strong provenance: detached commit `80ab3be09ce884f86cfc2037d3af30bc28061426`, exact P5 runner hash, exact P4-B hash, exact `ForcePositionAction` source, 20 Hz outer `d_pred` servo, 60 Hz native force loop, feed-forward `k=0.9`, and contact override off. The historical physical state was not recovered. Contact dynamics diverged during P4 close despite an identical initial observable hash, producing a different strict-preprobe state and a radically different branch force trace.

The archived five-level traces are internally ordered (rho `1.0` for native mean/top-5%/exposure), but the mandatory fresh reproduction sweep was not authorized after parity failed. Therefore the historical contract cannot be frozen as the final execution layer, old720 is not currently execution-matched to a reproducible final controller, and root7703 was not run.

## Method constraints preserved

No controller, posterior, Expected Utility rule, probe, or VLA arm trajectory was changed. No setpoint sweep or root7703 integration was run after the parity gate failed.

## Required next step

Recover either a restorable Aug-30 strict-preprobe scene snapshot or a complete time-pinned Isaac/contact-runtime image. If neither exists, move to the already-defined measured-force/exposure supervision pivot rather than continuing low-level controller work.
"""
    (OUT / "FINAL_EXECUTION_DECISION_REPORT.md").write_text(final_report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
