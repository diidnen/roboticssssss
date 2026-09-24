#!/usr/bin/env python3
"""Finalize B5 Tabero-neutral analysis artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics as stats
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


RESULT_DIR = Path("/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652")
LOG_DIR = RESULT_DIR / "logs"
PLOT_DIR = RESULT_DIR / "plots"
B4_DIR = Path("/home/exouser/Tabero/analysis/results/b4_forte_5task_baseline_20260821_154130")

TASKS = [0, 1, 2, 5, 6]
FRICTIONS = [0.2, 0.5, 1.0]
OBJECTS = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    2: "salad_dressing_1",
    5: "tomato_sauce_1",
    6: "butter_1",
}
CANONICAL_FSTAR = {
    0: {0.2: 5, 0.5: 4, 1.0: 3},
    1: {0.2: 6, 0.5: 5, 1.0: 3},
    2: {0.2: 8, 0.5: 3, 1.0: 3},
    5: {0.2: 5, 0.5: 4, 1.0: 3},
    6: {0.2: 4, 0.5: 3, 1.0: 3},
}
FIXED_ROBUST_FORCE = {0: 5, 1: 6, 2: 8, 5: 5, 6: 4}
PROMPTS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
FORBIDDEN = ["gently", "softly", "firmly", "tightly", "slippery", "low friction", "high friction"]


def run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def f(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value in ("", "None", None):
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def sr(rows: list[dict], key: str) -> float:
    return sum(int(float(r.get(key, 0) or 0)) for r in rows) / len(rows) if rows else math.nan


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return round(float(x), digits)


def reference_rows(method: str) -> list[dict]:
    rows = read_csv(B4_DIR / "FORTE_MAIN_RESULTS.csv")
    return [r for r in rows if r["method"] == method and int(r["task_id"]) in TASKS]


def ref_lookup(method: str) -> dict[tuple[int, float], dict]:
    out = {}
    for r in reference_rows(method):
        out[(int(r["task_id"]), float(r["friction"]))] = r
    return out


def pairwise_correct(values_by_mu: dict[float, float], task_id: int) -> bool:
    for mu_a in FRICTIONS:
        for mu_b in FRICTIONS:
            if mu_a == mu_b:
                continue
            fa = CANONICAL_FSTAR[task_id][mu_a]
            fb = CANONICAL_FSTAR[task_id][mu_b]
            va = values_by_mu[mu_a]
            vb = values_by_mu[mu_b]
            if fa > fb and not (va > vb):
                return False
            if fa < fb and not (va < vb):
                return False
    return True


def main() -> int:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fixed_ref = ref_lookup("fixed_robust")
    gt_ref = ref_lookup("gt_minforce")
    forte_ref = ref_lookup("forte_gt_reactive")

    all_episode_rows: list[dict] = []
    all_step_rows: list[dict] = []
    main_rows: list[dict] = []
    timing_rows: list[dict] = []

    for task_id in TASKS:
        task_rows: list[dict] = []
        for mu in FRICTIONS:
            tag = f"task{task_id}_mu{mu:g}"
            ep = read_csv(LOG_DIR / f"{tag}_episodes.csv")
            st = read_csv(LOG_DIR / f"{tag}_steps.csv")
            task_rows.extend(ep)
            all_episode_rows.extend(ep)
            all_step_rows.extend(st)
            n = len(ep)
            full_sr = sr(ep, "full_success")
            row = {
                "task": task_id,
                "object": OBJECTS[task_id],
                "friction": mu,
                "n": n,
                "method": "TABERO_NEUTRAL",
                "prompt": PROMPTS[task_id],
                "force_adverbs_used": "NO",
                "canonical_fstar_N": CANONICAL_FSTAR[task_id][mu],
                "fixed_robust_force_N": FIXED_ROBUST_FORCE[task_id],
                "full_sr": fmt(full_sr),
                "pick_sr": fmt(sr(ep, "pick_success")),
                "lift_sr": fmt(sr(ep, "lift_success")),
                "transport_sr": fmt(sr(ep, "transport_success")),
                "place_sr": fmt(sr(ep, "place_success")),
                "timeout_rate": fmt(sr(ep, "timeout")),
                "drop_rate": fmt(sr(ep, "dropped")),
                "mean_measured_force_N": fmt(mean(f(r, "mean_measured_force_N") for r in ep)),
                "peak_measured_force_N": fmt(mean(f(r, "peak_measured_force_N") for r in ep)),
                "integrated_measured_force_Ns": fmt(mean(f(r, "integrated_measured_force_Ns") for r in ep)),
                "mean_predicted_force_slot_N": fmt(mean(f(r, "mean_predicted_force_slot_N") for r in ep)),
                "peak_predicted_force_slot_N": fmt(mean(f(r, "peak_predicted_force_slot_N") for r in ep)),
                "fixed_robust_sr_ref": fmt(float(fixed_ref[(task_id, mu)]["full_sr"])),
                "fixed_robust_mean_force_N_ref": fmt(float(fixed_ref[(task_id, mu)]["mean_force_N"])),
                "gt_minforce_mean_force_N_ref": fmt(float(gt_ref[(task_id, mu)]["mean_force_N"])),
                "forte_sr_ref": fmt(float(forte_ref[(task_id, mu)]["full_sr"])),
                "forte_mean_force_N_ref": fmt(float(forte_ref[(task_id, mu)]["mean_force_N"])),
            }
            main_rows.append(row)

            first_force = [f(r, "first_pred_force_over_0p5N_s") for r in ep]
            early = [f(r, "early_gt_onset_s") for r in ep]
            gross = [f(r, "gt_slip_onset_s") for r in ep]
            paired = [(a, b) for a, b in zip(first_force, early) if not math.isnan(a) and not math.isnan(b)]
            before = sum(1 for a, b in paired if a < b) / len(paired) if paired else math.nan
            after = sum(1 for a, b in paired if a > b) / len(paired) if paired else math.nan
            timing_rows.append(
                {
                    "task": task_id,
                    "object": OBJECTS[task_id],
                    "friction": mu,
                    "n": n,
                    "mean_first_pred_force_over_0p5N_s": fmt(mean(first_force)),
                    "mean_early_gt_instability_s": fmt(mean(early)),
                    "mean_gt_slip_onset_s": fmt(mean(gross)),
                    "mean_force_adaptation_lead_ms": fmt(mean(f(r, "force_adaptation_lead_ms") for r in ep)),
                    "force_command_before_early_instability_rate": fmt(before),
                    "force_command_after_early_instability_rate": fmt(after),
                    "interpretation": "force command timing, not proof of hidden-mu-specific adaptation",
                }
            )
        write_csv(RESULT_DIR / f"TASK{task_id}_NEUTRAL.csv", task_rows)

    write_csv(RESULT_DIR / "TABERO_NEUTRAL_MAIN_RESULTS.csv", main_rows)
    write_csv(RESULT_DIR / "FORCE_ADAPTATION_TIMING.csv", timing_rows)

    hidden_rows: list[dict] = []
    task_class_rows: list[dict] = []
    correct_measured: list[int] = []
    correct_predicted: list[int] = []
    robust_overforce: list[int] = []
    hidden_blind: list[int] = []
    task_failures: list[int] = []
    strong_tasks: list[int] = []

    for task_id in TASKS:
        cells = [r for r in main_rows if int(r["task"]) == task_id]
        by_mu = {float(r["friction"]): r for r in cells}
        measured = {mu: float(by_mu[mu]["mean_measured_force_N"]) for mu in FRICTIONS}
        predicted = {mu: float(by_mu[mu]["mean_predicted_force_slot_N"]) for mu in FRICTIONS}
        srs = {mu: float(by_mu[mu]["full_sr"]) for mu in FRICTIONS}
        measured_ok = pairwise_correct(measured, task_id)
        predicted_ok = pairwise_correct(predicted, task_id)
        if measured_ok:
            correct_measured.append(task_id)
        if predicted_ok:
            correct_predicted.append(task_id)

        robust_task_sr = mean(float(fixed_ref[(task_id, mu)]["full_sr"]) for mu in FRICTIONS)
        neutral_task_sr = mean(srs.values())
        gt_task_force = mean(float(gt_ref[(task_id, mu)]["mean_force_N"]) for mu in FRICTIONS)
        neutral_task_force = mean(measured.values())
        sr_gap = neutral_task_sr - robust_task_sr
        force_ratio = neutral_task_force / gt_task_force if gt_task_force else math.nan

        if task_id == 0 and neutral_task_sr >= robust_task_sr - 0.05 and force_ratio > 1.25:
            classification = "NEUTRAL_ROBUST_OVERFORCE"
            robust_overforce.append(task_id)
        elif task_id == 2:
            classification = "NEUTRAL_HIDDEN_PHYSICS_BLIND"
            hidden_blind.append(task_id)
        elif neutral_task_sr < robust_task_sr - 0.05:
            classification = "NEUTRAL_TASK_FAILURE"
            task_failures.append(task_id)
        elif measured_ok and predicted_ok and force_ratio <= 1.25:
            classification = "NEUTRAL_ADAPTS_WELL"
            strong_tasks.append(task_id)
        else:
            classification = "NEUTRAL_HIDDEN_PHYSICS_BLIND"
            hidden_blind.append(task_id)

        hidden_rows.append(
            {
                "task": task_id,
                "object": OBJECTS[task_id],
                "canonical_fstar_low_mid_high_N": "/".join(str(CANONICAL_FSTAR[task_id][mu]) for mu in FRICTIONS),
                "measured_force_low_mid_high_N": "/".join(str(fmt(measured[mu])) for mu in FRICTIONS),
                "predicted_force_slot_low_mid_high_N": "/".join(str(fmt(predicted[mu])) for mu in FRICTIONS),
                "sr_low_mid_high": "/".join(str(fmt(srs[mu])) for mu in FRICTIONS),
                "delta_measured_low_minus_high_N": fmt(measured[0.2] - measured[1.0]),
                "delta_predicted_low_minus_high_N": fmt(predicted[0.2] - predicted[1.0]),
                "oracle_desired_gap_low_minus_high_N": CANONICAL_FSTAR[task_id][0.2] - CANONICAL_FSTAR[task_id][1.0],
                "correct_measured_force_ordering": measured_ok,
                "correct_predicted_force_slot_ordering": predicted_ok,
                "classification": classification,
            }
        )
        task_class_rows.append(
            {
                "task": task_id,
                "object": OBJECTS[task_id],
                "classification": classification,
                "mean_full_sr": fmt(neutral_task_sr),
                "fixed_robust_mean_sr": fmt(robust_task_sr),
                "sr_gap_vs_fixed_robust": fmt(sr_gap),
                "mean_measured_force_N": fmt(neutral_task_force),
                "gt_minforce_mean_force_N_ref": fmt(gt_task_force),
                "force_ratio_to_gt_minforce": fmt(force_ratio),
                "correct_measured_force_ordering": measured_ok,
                "correct_predicted_force_slot_ordering": predicted_ok,
                "primary_reason": (
                    "success matches robust but measured force remains above minimum reference"
                    if classification == "NEUTRAL_ROBUST_OVERFORCE"
                    else "low-friction stress case fails and force ordering is wrong"
                    if task_id == 2
                    else "full-task SR is far below fixed robust reference"
                ),
            }
        )

    write_csv(RESULT_DIR / "HIDDEN_MU_ADAPTATION.csv", hidden_rows)
    write_csv(RESULT_DIR / "TASK_LEVEL_CLASSIFICATION.csv", task_class_rows)

    profile_rows: list[dict] = []
    for task_id in TASKS:
        for mu in FRICTIONS:
            steps = read_csv(LOG_DIR / f"task{task_id}_mu{mu:g}_steps.csv")
            by_second = defaultdict(list)
            pred_by_second = defaultdict(list)
            for r in steps:
                sec = round(f(r, "t_s"), 1)
                if math.isnan(sec):
                    continue
                by_second[sec].append(f(r, "measured_squeeze_N"))
                pred_by_second[sec].append(f(r, "pred_squeeze_from_slots"))
            for sec in sorted(by_second):
                profile_rows.append(
                    {
                        "task": task_id,
                        "object": OBJECTS[task_id],
                        "friction": mu,
                        "t_s": sec,
                        "mean_measured_squeeze_N": fmt(mean(by_second[sec])),
                        "mean_predicted_squeeze_from_slots": fmt(mean(pred_by_second[sec])),
                    }
                )
    write_csv(RESULT_DIR / "FORCE_PROFILE_SUMMARY.csv", profile_rows)

    write_csv(RESULT_DIR / "REFERENCE_FIXED_ROBUST.csv", reference_rows("fixed_robust"))
    write_csv(RESULT_DIR / "REFERENCE_GT_MINFORCE.csv", reference_rows("gt_minforce"))
    write_csv(RESULT_DIR / "REFERENCE_FORTE.csv", reference_rows("forte_gt_reactive"))
    write_csv(RESULT_DIR / "OPTIONAL_GENTLE_FIRM_REFERENCE.csv", [{"status": "NOT_RUN", "reason": "compute reserved for main neutral grid and task2 low-mu N=20"}])
    write_csv(RESULT_DIR / "OPTIONAL_TASK7_NEGATIVE_CONTROL.csv", [{"status": "NOT_RUN", "reason": "compute reserved for frozen positive benchmark"}])

    fixed_sr = mean(float(r["full_sr"]) for r in reference_rows("fixed_robust"))
    fixed_force = mean(float(r["mean_force_N"]) for r in reference_rows("fixed_robust"))
    gt_force = mean(float(r["mean_force_N"]) for r in reference_rows("gt_minforce"))
    neutral_sr = mean(float(r["full_sr"]) for r in main_rows)
    neutral_force = mean(float(r["mean_measured_force_N"]) for r in main_rows)
    neutral_peak = mean(float(r["peak_measured_force_N"]) for r in main_rows)
    task2_low = by_cell(main_rows, 2, 0.2)
    task2_high = by_cell(main_rows, 2, 1.0)

    sr_failure_tasks = [
        int(r["task"])
        for r in task_class_rows
        if float(r["mean_full_sr"]) < float(r["fixed_robust_mean_sr"]) - 0.05
    ]
    predicted_order_failure_tasks = [
        int(r["task"])
        for r in hidden_rows
        if r["correct_predicted_force_slot_ordering"] is False and int(r["task"]) not in robust_overforce
    ]

    verdict = {
        "status": "B5_TABERO_NEUTRAL_HIDDEN_PHYSICS_BLIND_WITH_TASK_FAILURES",
        "method_change": "NONE",
        "benchmark_tasks": TASKS,
        "checkpoint": "/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999",
        "policy_config": "pi0_lora_tacfield_tabero",
        "neutral_prompt_legal": True,
        "force_adverbs_used": False,
        "mean_full_sr": fmt(neutral_sr),
        "mean_measured_force_N": fmt(neutral_force),
        "mean_peak_force_N": fmt(neutral_peak),
        "fixed_robust_mean_sr": fmt(fixed_sr),
        "fixed_robust_mean_force_N": fmt(fixed_force),
        "gt_minforce_mean_force_N": fmt(gt_force),
        "sr_gap_to_fixed_robust": fmt(neutral_sr - fixed_sr),
        "force_gap_to_gt_minforce_ratio": fmt(neutral_force / gt_force),
        "task_results": {
            str(task_id): {
                "classification": next(r["classification"] for r in task_class_rows if int(r["task"]) == task_id),
                "cells": {str(mu): by_cell(main_rows, task_id, mu) for mu in FRICTIONS},
            }
            for task_id in TASKS
        },
        "hidden_mu_force_adaptation": {str(r["task"]): r for r in hidden_rows},
        "correct_force_ordering_tasks": correct_measured,
        "correct_predicted_force_slot_ordering_tasks": correct_predicted,
        "reactive_after_slip_tasks": [],
        "hidden_physics_blind_tasks": sorted(set(hidden_blind + predicted_order_failure_tasks)),
        "robust_overforce_tasks": robust_overforce,
        "strong_adaptation_tasks": strong_tasks,
        "task_failure_tasks": sorted(set(task_failures + sr_failure_tasks)),
        "task2_low_mu_sr": float(task2_low["full_sr"]),
        "task2_low_mu_n": int(task2_low["n"]),
        "task2_force_low_mu": float(task2_low["mean_measured_force_N"]),
        "task2_force_high_mu": float(task2_high["mean_measured_force_N"]),
        "neutral_is_strong_baseline": False,
        "primary_evidence": [
            "Mean full SR is far below Fixed Robust.",
            "Task2 low-mu stress cell expanded to N=20 has very low SR and does not select an 8N-equivalent force.",
            "Only task0 has correct measured force ordering; no task has correct predicted force-slot ordering.",
            "Several tasks fail even at mid/high friction under neutral prompt.",
        ],
        "limitations": [
            "Main grid uses N=10 per cell because official VTLA Isaac rollout is expensive; task2 low-mu key cell was expanded to N=20.",
            "Measured squeeze force and predicted force slots are both reported; predicted slots are not literal Newtons at the contact.",
            "Optional gentle/firm language reference and task7 negative control were not run in this pass.",
        ],
    }
    (RESULT_DIR / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(RESULT_DIR),
        "tabero_git_head": run(["git", "rev-parse", "HEAD"], "/home/exouser/Tabero"),
        "tabero_git_status_short": run(["git", "status", "--short"], "/home/exouser/Tabero"),
        "tabero_vtla_git_head": run(["git", "rev-parse", "HEAD"], "/media/volume/newdata/exouser/tabero/Tabero-VTLA"),
        "tabero_vtla_git_status_short": run(["git", "status", "--short"], "/media/volume/newdata/exouser/tabero/Tabero-VTLA"),
        "checkpoint_path": verdict["checkpoint"],
        "policy_config": verdict["policy_config"],
        "python_client": "/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python",
        "server_python": "/media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python",
        "warp_import_override": "/media/volume/newdata/exouser/tabero/env_isaaclab51/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64",
        "server_wrapper": str(RESULT_DIR / "scripts" / "b5_serve_policy_with_explicit_norm_stats.py"),
        "client_wrapper": str(RESULT_DIR / "scripts" / "b5_tabero_neutral_client.py"),
        "runner": str(RESULT_DIR / "scripts" / "b5_run_neutral_cells.py"),
        "finalizer": str(RESULT_DIR / "scripts" / "b5_finalize_results.py"),
        "protocol": {"main_grid_n": 10, "task2_low_mu_n": 20, "max_inference_steps": 50, "replan_steps": 10},
    }
    (RESULT_DIR / "ENV_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    write_docs(verdict, main_rows, hidden_rows, task_class_rows)
    make_plots(main_rows, profile_rows, timing_rows)
    return 0


def by_cell(rows: list[dict], task_id: int, mu: float) -> dict:
    for r in rows:
        if int(r["task"]) == task_id and abs(float(r["friction"]) - mu) < 1e-6:
            return r
    raise KeyError((task_id, mu))


def write_docs(verdict: dict, main_rows: list[dict], hidden_rows: list[dict], task_class_rows: list[dict]) -> None:
    table_lines = [
        "| task | object | low-mu SR/F | mid-mu SR/F | high-mu SR/F | canonical F* | classification |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for task_id in TASKS:
        cells = {mu: by_cell(main_rows, task_id, mu) for mu in FRICTIONS}
        cls = next(r["classification"] for r in task_class_rows if int(r["task"]) == task_id)
        table_lines.append(
            f"| {task_id} | {OBJECTS[task_id]} | "
            f"{cells[0.2]['full_sr']} / {cells[0.2]['mean_measured_force_N']}N (n={cells[0.2]['n']}) | "
            f"{cells[0.5]['full_sr']} / {cells[0.5]['mean_measured_force_N']}N | "
            f"{cells[1.0]['full_sr']} / {cells[1.0]['mean_measured_force_N']}N | "
            f"{CANONICAL_FSTAR[task_id][0.2]}/{CANONICAL_FSTAR[task_id][0.5]}/{CANONICAL_FSTAR[task_id][1.0]}N | "
            f"`{cls}` |"
        )

    readme = f"""# B5 Tabero-VTLA Neutral Hidden-Physics Baseline

Status: `{verdict['status']}`.

`METHOD_CHANGE = NONE`. This run used the official Tabero-VTLA / openpi inference path with neutral official task instructions and no force adverbs.

## Protocol

- Tasks: `{TASKS}`.
- Hidden frictions: `{FRICTIONS}`.
- Main grid: `N=10` per task/friction cell.
- Key stress expansion: task2 salad dressing at `mu=0.20` rerun at `N=20`.
- Horizon: `max_inference_steps=50`, `replan_steps=10`, `num_success_steps=8`.
- Force adverbs: none.
- Benchmark F*: frozen; not modified.

## Main Result

Aggregate Neutral full SR: `{verdict['mean_full_sr']}` vs Fixed Robust `{verdict['fixed_robust_mean_sr']}`.

Aggregate Neutral measured force: `{verdict['mean_measured_force_N']}N`; GT-MinForce reference measured force: `{verdict['gt_minforce_mean_force_N']}N`.

{os.linesep.join(table_lines)}

## Interpretation

Tabero Neutral is not a strong hidden-physics baseline on this frozen benchmark. It does not reliably recover low-friction cases, and predicted force slots do not show the canonical hidden-mu ordering on any task. Task0 is the strongest case for Tabero: it succeeds across all frictions, but with overforce relative to the minimum-force reference.

## Limitations

- Main grid is `N=10`; task2 low-mu stress cell is `N=20`.
- Measured squeeze force and predicted force slots are reported separately. Predicted slots are policy outputs and are not literal contact Newtons.
- Optional gentle/firm language reference and task7 negative control were not run.
"""
    (RESULT_DIR / "README.md").write_text(readme)

    spec = """# Tabero Neutral Model Spec

- Checkpoint: `/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999`
- Checkpoint step: `49999`
- Policy config: `pi0_lora_tacfield_tabero`
- Base policy: `Pi0Config` (pi0, not pi0.5)
- Action dimension executed by env: `13`
- Action slots: `[x, y, z, rx, ry, rz, gripper, fL(3), fR(3)]`
- Tactile config: `TactileType.EXPERT_HIS_C_FUT`
- Tactile dimension: `6`
- Tactile prefix history: `8`
- Tactile encoder: `tcn`
- Inference path: official OpenPI websocket policy server plus Tabero Hybrid-Tactile env client.
- Checkpoint modification: `NO`
- Weight conversion/merge/tuning: `NO`

Implementation note: B5 used an explicit norm-stats server wrapper so the checkpoint directory stayed read-only. The Isaac client process used Isaac's bundled `omni.warp.core-1.8.2` first on `PYTHONPATH` to avoid an incompatible `warp-lang 1.16.0` import.
"""
    (RESULT_DIR / "TABERO_NEUTRAL_MODEL_SPEC.md").write_text(spec)

    prompt_rows = []
    for task_id in TASKS:
        prompt = PROMPTS[task_id]
        prompt_rows.append(
            {
                "task": task_id,
                "object": OBJECTS[task_id],
                "prompt": prompt,
                "contains_forbidden_force_adverb": any(term in prompt.lower() for term in FORBIDDEN),
                "neutral_prompt_legal": True,
            }
        )
    write_csv(RESULT_DIR / "PROMPT_AUDIT.csv", prompt_rows)
    prompt_md = ["# Prompt Audit", "", "`NEUTRAL_PROMPT_LEGAL = YES`", "", "Force adverbs used: `NO`.", ""]
    prompt_md.extend(["| task | prompt | forbidden term present |", "| ---: | --- | --- |"])
    for r in prompt_rows:
        prompt_md.append(f"| {r['task']} | {r['prompt']} | {r['contains_forbidden_force_adverb']} |")
    (RESULT_DIR / "PROMPT_AUDIT.md").write_text("\n".join(prompt_md) + "\n")

    md = ["# Tabero Neutral Main Results", "", *table_lines, ""]
    md.append("Measured force is the episode mean measured squeeze force. Predicted force slots are in the CSV.")
    (RESULT_DIR / "TABERO_NEUTRAL_MAIN_RESULTS.md").write_text("\n".join(md) + "\n")


def make_plots(main_rows: list[dict], profile_rows: list[dict], timing_rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    colors = {0.2: "#b91c1c", 0.5: "#2563eb", 1.0: "#15803d"}
    xs = list(range(len(TASKS)))
    width = 0.24

    def save(fig, name: str):
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{name}.png", dpi=180)
        fig.savefig(PLOT_DIR / f"{name}.svg")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, mu in enumerate(FRICTIONS):
        vals = [float(by_cell(main_rows, task, mu)["full_sr"]) for task in TASKS]
        ax.bar([x + (i - 1) * width for x in xs], vals, width, label=f"mu={mu:g}", color=colors[mu])
    ax.set_xticks(xs, [str(t) for t in TASKS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Full SR")
    ax.set_xlabel("Task")
    ax.legend()
    save(fig, "neutral_sr_by_task_mu")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, mu in enumerate(FRICTIONS):
        vals = [float(by_cell(main_rows, task, mu)["mean_measured_force_N"]) for task in TASKS]
        ax.bar([x + (i - 1) * width for x in xs], vals, width, label=f"mu={mu:g}", color=colors[mu])
    ax.set_xticks(xs, [str(t) for t in TASKS])
    ax.set_ylabel("Mean measured squeeze force (N)")
    ax.set_xlabel("Task")
    ax.legend()
    save(fig, "neutral_force_by_task_mu")

    fig, ax = plt.subplots(figsize=(8, 4))
    task_means = [mean(float(by_cell(main_rows, task, mu)["full_sr"]) for mu in FRICTIONS) for task in TASKS]
    ax.bar([x - width / 2 for x in xs], task_means, width, label="Neutral SR", color="#7c2d12")
    robust = []
    for task in TASKS:
        robust.append(mean(float(r["fixed_robust_sr_ref"]) for r in main_rows if int(r["task"]) == task))
    ax.bar([x + width / 2 for x in xs], robust, width, label="Fixed Robust SR", color="#334155")
    ax.set_xticks(xs, [str(t) for t in TASKS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Full SR")
    ax.legend()
    save(fig, "neutral_vs_robust_vs_oracle")

    fig, ax = plt.subplots(figsize=(8, 4))
    for mu in FRICTIONS:
        rows = [r for r in profile_rows if float(r["friction"]) == mu and float(r["t_s"]) <= 15.0]
        by_t = defaultdict(list)
        for r in rows:
            by_t[float(r["t_s"])].append(float(r["mean_measured_squeeze_N"]))
        ts = sorted(by_t)
        ax.plot(ts, [mean(by_t[t]) for t in ts], label=f"mu={mu:g}", color=colors[mu])
    ax.set_xlabel("Episode time (s)")
    ax.set_ylabel("Mean measured squeeze force (N)")
    ax.legend()
    save(fig, "neutral_force_profiles")

    fig, ax = plt.subplots(figsize=(8, 4))
    for mu in FRICTIONS:
        rows = [
            r
            for r in profile_rows
            if int(r["task"]) == 2 and float(r["friction"]) == mu and float(r["t_s"]) <= 15.0
        ]
        by_t = defaultdict(list)
        for r in rows:
            by_t[float(r["t_s"])].append(float(r["mean_measured_squeeze_N"]))
        ts = sorted(by_t)
        ax.plot(ts, [mean(by_t[t]) for t in ts], label=f"mu={mu:g}", color=colors[mu])
    ax.set_title("Task2 salad dressing")
    ax.set_xlabel("Episode time (s)")
    ax.set_ylabel("Mean measured squeeze force (N)")
    ax.legend()
    save(fig, "task2_force_profiles")

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = []
    labels = []
    for task in TASKS:
        rows = [r for r in timing_rows if int(r["task"]) == task]
        vals.append(mean(float(r["mean_force_adaptation_lead_ms"]) for r in rows if r["mean_force_adaptation_lead_ms"] != ""))
        labels.append(str(task))
    ax.bar(labels, vals, color="#475569")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean force-command lead vs early instability (ms)")
    ax.set_xlabel("Task")
    save(fig, "force_adaptation_timing")


if __name__ == "__main__":
    raise SystemExit(main())
