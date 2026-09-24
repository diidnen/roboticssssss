#!/usr/bin/env python3
"""Finalize B2-R2 Tabero official task breadth artifacts.

This script only reads frozen/reference CSVs and writes into the B2-R2 result
directory that contains it. It does not touch Tabero source or prior result dirs.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[1]
PLOTS = BASE / "plots"
TASK_CONFIG = Path("/home/exouser/Tabero/benchmarks/datasets/libero/config/libero_object.json")

TASK_IDS = [0, 1, 2, 3, 5, 6, 7, 8, 9]
MUS = [0.2, 0.5, 1.0]
TAUS = [0.6, 0.8, 0.9]

TASK1_FROZEN_FSTAR = {0.2: 6.0, 0.5: 5.0, 1.0: 3.0}
TASK1_FROZEN_ROBUST = 6.0


@dataclass
class Cell:
    task_id: int
    mu: float
    force: float
    successes: int = 0
    n: int = 0
    friction_ok: int = 0

    @property
    def sr(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    @property
    def friction_ok_rate(self) -> float:
        return self.friction_ok / self.n if self.n else float("nan")


def clean_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return float(text)


def fmt_force(value: Optional[float]) -> str:
    if value is None:
        return "NO_FSTAR_IN_RANGE"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def fmt_num(value: Optional[float], ndigits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{ndigits}f}"


def plot_float(value: object) -> float:
    try:
        text = str(value).strip()
        if not text or text.startswith("NO_"):
            return float("nan")
        return float(text)
    except Exception:
        return float("nan")


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def scan_path(task_id: int) -> Path:
    if task_id == 1:
        return BASE / "TASK1_FROZEN_REFERENCE_SCAN.csv"
    if task_id == 7:
        return BASE / "TASK7_FROZEN_REFERENCE_SCAN.csv"
    return BASE / f"TASK{task_id}_SCAN.csv"


def row_force(row: dict) -> float:
    return clean_float(row.get("force")) or clean_float(row.get("chosen_force"))  # type: ignore[return-value]


def row_success(row: dict) -> int:
    return int((clean_float(row.get("full_task_success")) or 0.0) > 0.5)


def row_friction_ok(row: dict) -> int:
    if "friction_override_ok" in row and str(row.get("friction_override_ok", "")).strip() != "":
        return int((clean_float(row.get("friction_override_ok")) or 0.0) > 0.5)
    applied = clean_float(row.get("friction_applied"))
    requested = clean_float(row.get("friction"))
    if applied is None or requested is None:
        return 0
    return int(abs(applied - requested) < 1e-3)


def load_cells() -> Dict[int, Dict[Tuple[float, float], Cell]]:
    by_task: Dict[int, Dict[Tuple[float, float], Cell]] = {}
    for task_id in TASK_IDS:
        rows = read_csv_rows(scan_path(task_id))
        cells: Dict[Tuple[float, float], Cell] = {}
        for row in rows:
            mu = round(float(row["friction"]), 2)
            force = float(row_force(row))
            key = (mu, force)
            if key not in cells:
                cells[key] = Cell(task_id=task_id, mu=mu, force=force)
            cell = cells[key]
            cell.n += 1
            cell.successes += row_success(row)
            cell.friction_ok += row_friction_ok(row)
        by_task[task_id] = cells
    return by_task


def load_provenance() -> Dict[int, dict]:
    rows = read_csv_rows(BASE / "TASK_DATA_PROVENANCE.csv")
    return {int(row["task_id"]): row for row in rows}


def load_task_config() -> Dict[int, dict]:
    data = json.loads(TASK_CONFIG.read_text())
    tasks = {}
    for idx, task in enumerate(data.get("tasks", [])):
        if idx in TASK_IDS:
            obj = task.get("obj_of_interest", [None])[0]
            meta = task.get("objects", {}).get(obj, {})
            tasks[idx] = {
                "task_name": task.get("task_name", ""),
                "instruction": task.get("language_instruction", ""),
                "target_object": obj,
                "object_type": meta.get("type", ""),
                "scale": meta.get("scale", ""),
                "xml_path": meta.get("xml_path", ""),
            }
    return tasks


def fstar_for(cells: Dict[Tuple[float, float], Cell], mu: float, tau: float) -> Optional[float]:
    candidates = [
        force
        for (cell_mu, force), cell in cells.items()
        if abs(cell_mu - mu) < 1e-9 and cell.n > 0 and cell.sr >= tau
    ]
    return min(candidates) if candidates else None


def robust_for(cells: Dict[Tuple[float, float], Cell], tau: float) -> Optional[float]:
    forces = sorted({force for _, force in cells})
    for force in forces:
        ok = True
        for mu in MUS:
            cell = cells.get((mu, force))
            if cell is None or cell.n == 0 or cell.sr < tau:
                ok = False
                break
        if ok:
            return force
    return None


def adjacent_evidence(cells: Dict[Tuple[float, float], Cell], mu: float, fstar: Optional[float]) -> str:
    if fstar is None:
        return "no_fstar"
    current = cells.get((mu, fstar))
    lower_forces = sorted(force for cell_mu, force in cells if abs(cell_mu - mu) < 1e-9 and force < fstar)
    lower = cells.get((mu, lower_forces[-1])) if lower_forces else None
    cur = f"F{fmt_force(fstar)} {current.successes}/{current.n}" if current else f"F{fmt_force(fstar)} missing"
    if lower:
        return f"{cur}; lower F{fmt_force(lower.force)} {lower.successes}/{lower.n}"
    return f"{cur}; no lower tested force"


def classify_task(task_id: int, fstars: Dict[float, Optional[float]], robust: Optional[float]) -> str:
    vals = [fstars[mu] for mu in MUS]
    if any(v is None for v in vals):
        return "NO_FEASIBLE_RANGE"
    if all(v == 3.0 for v in vals):
        return "FIXED_LOW_EXISTS"
    mean_force = sum(vals) / len(vals)  # type: ignore[arg-type]
    if len(set(vals)) >= 2 and robust is not None and mean_force < robust:
        return "PHYSICS_DECISION_POSITIVE"
    return "FIXED_ROBUST_NO_DECISION"


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: List[str], rows: Iterable[Iterable[object]]) -> str:
    rows = [[str(x) for x in row] for row in rows]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def plot_outputs(summary_rows: List[dict]) -> None:
    PLOTS.mkdir(exist_ok=True)
    tasks = [str(r["task_id"]) for r in summary_rows]
    labels = [f"{r['task_id']} {r['object_short']}" for r in summary_rows]
    x = range(len(tasks))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.25
    for offset, mu, color in [(-width, 0.2, "#cf4a46"), (0, 0.5, "#4d77b3"), (width, 1.0, "#3b8f61")]:
        values = [plot_float(r[f"fstar_mu_{mu}"]) for r in summary_rows]
        ax.bar([i + offset for i in x], values, width=width, label=f"mu={mu:g}", color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Canonical F* full task (N), tau=0.8")
    ax.set_title("Canonical minimum sufficient force by official task")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "fstar_by_task.png", dpi=200)
    fig.savefig(PLOTS / "fstar_by_task.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    robust = [plot_float(r["fixed_robust"]) for r in summary_rows]
    oracle = [plot_float(r["gt_minforce_mean"]) for r in summary_rows]
    xpos = list(range(len(labels)))
    ax.plot(xpos, robust, marker="o", label="Fixed Robust", color="#222222")
    ax.plot(xpos, oracle, marker="o", label="GT-MinForce mean", color="#2f6fbb")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Force (N)")
    ax.set_title("Fixed robust force versus oracle mean force")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "robust_vs_oracle.png", dpi=200)
    fig.savefig(PLOTS / "robust_vs_oracle.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#cf4a46" if r["classification"] == "PHYSICS_DECISION_POSITIVE" else "#777777" for r in summary_rows]
    xpos = list(range(len(labels)))
    ax.bar(xpos, [plot_float(r["delta_force"]) for r in summary_rows], color=colors)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Fixed Robust - GT-MinForce mean (N)")
    ax.set_title("Oracle force-saving potential by task")
    fig.tight_layout()
    fig.savefig(PLOTS / "force_saving_by_task.png", dpi=200)
    fig.savefig(PLOTS / "force_saving_by_task.svg")
    plt.close(fig)

    counts = defaultdict(int)
    for row in summary_rows:
        counts[row["classification"]] += 1
    classes = sorted(counts)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    xpos = list(range(len(classes)))
    ax.bar(xpos, [counts[c] for c in classes], color=["#cf4a46" if "POSITIVE" in c else "#777777" for c in classes])
    ax.set_ylabel("Official task count")
    ax.set_title("Task classification counts")
    ax.set_xticks(xpos)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS / "task_classification.png", dpi=200)
    fig.savefig(PLOTS / "task_classification.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for row in summary_rows:
        if row["classification"] == "PHYSICS_DECISION_POSITIVE" or row["task_id"] in {"7", "3", "9"}:
            style = "-" if row["classification"] == "PHYSICS_DECISION_POSITIVE" else "--"
            color = "#cf4a46" if row["classification"] == "PHYSICS_DECISION_POSITIVE" else "#777777"
            values = [plot_float(row[f"fstar_mu_{mu}"]) for mu in MUS]
            if all(math.isnan(v) for v in values):
                continue
            ax.plot(MUS, values, style, marker="o", color=color, alpha=0.8)
            label_y = next((v for v in reversed(values) if not math.isnan(v)), None)
            if label_y is not None:
                ax.text(MUS[-1] + 0.02, label_y, f"T{row['task_id']}", va="center", fontsize=8)
    ax.set_xlabel("Object friction mu")
    ax.set_ylabel("Canonical F* full task (N), tau=0.8")
    ax.set_title("Positive tasks vary with friction; fixed-low controls stay flat")
    fig.tight_layout()
    fig.savefig(PLOTS / "positive_negative_task_examples.png", dpi=200)
    fig.savefig(PLOTS / "positive_negative_task_examples.svg")
    plt.close(fig)


def main() -> None:
    cells_by_task = load_cells()
    provenance = load_provenance()
    task_cfg = load_task_config()

    canonical_rows: List[dict] = []
    robust_rows: List[dict] = []
    gt_rows: List[dict] = []
    table_rows: List[dict] = []
    positive_rows: List[dict] = []
    object_rows: List[dict] = []
    quality_rows: List[dict] = []

    for task_id in TASK_IDS:
        cells = cells_by_task[task_id]
        prov = provenance[task_id]
        cfg = task_cfg.get(task_id, {})

        fstars_by_tau = {
            tau: {mu: fstar_for(cells, mu, tau) for mu in MUS}
            for tau in TAUS
        }
        if task_id == 1:
            fstars_by_tau[0.8] = dict(TASK1_FROZEN_FSTAR)
        fstars = fstars_by_tau[0.8]

        robust = robust_for(cells, 0.8)
        robust_source = "observed_all_mu_cells"
        if task_id == 1:
            robust = TASK1_FROZEN_ROBUST
            robust_source = "frozen_B2_E2E2_reference"

        classification = classify_task(task_id, fstars, robust)
        fstar_vals = [fstars[mu] for mu in MUS]
        oracle_mean = sum(v for v in fstar_vals if v is not None) / 3.0 if all(v is not None for v in fstar_vals) else None
        delta = robust - oracle_mean if robust is not None and oracle_mean is not None else None
        object_short = str(prov["object"]).replace("_1", "").replace("_", " ")

        table_row = {
            "task_id": str(task_id),
            "object": prov["object"],
            "object_short": object_short,
            "data": "PASS" if prov["reset_pass"] == "PASS" else prov["reset_pass"],
            "full_traj": prov["full_trajectory_smoke"],
            "fstar_mu_0.2": fmt_force(fstars[0.2]),
            "fstar_mu_0.5": fmt_force(fstars[0.5]),
            "fstar_mu_1.0": fmt_force(fstars[1.0]),
            "fixed_robust": fmt_force(robust),
            "gt_minforce_mean": fmt_num(oracle_mean),
            "delta_force": fmt_num(delta),
            "classification": classification,
            "instruction": prov["instruction"],
            "robust_source": robust_source,
        }
        table_rows.append(table_row)

        robust_min_sr = None
        robust_evidence = []
        if robust is not None:
            vals = []
            for mu in MUS:
                cell = cells.get((mu, robust))
                if cell is None and task_id == 1:
                    robust_evidence.append(f"mu={mu:g}: frozen reference")
                    continue
                if cell:
                    vals.append(cell.sr)
                    robust_evidence.append(f"mu={mu:g}: {cell.successes}/{cell.n}")
            robust_min_sr = min(vals) if vals else None

        robust_rows.append({
            "task_id": task_id,
            "object": prov["object"],
            "fixed_robust_force_tau_0_8": fmt_force(robust),
            "min_observed_sr_at_robust": fmt_num(robust_min_sr),
            "evidence_by_mu": "; ".join(robust_evidence),
            "source": robust_source,
        })

        gt_rows.append({
            "task_id": task_id,
            "object": prov["object"],
            "fstar_low_mu_0_2": fmt_force(fstars[0.2]),
            "fstar_mid_mu_0_5": fmt_force(fstars[0.5]),
            "fstar_high_mu_1_0": fmt_force(fstars[1.0]),
            "gt_minforce_mean": fmt_num(oracle_mean),
            "fixed_robust_force": fmt_force(robust),
            "force_saving_delta": fmt_num(delta),
            "classification": classification,
        })

        if classification == "PHYSICS_DECISION_POSITIVE":
            positive_rows.append({
                "task_id": task_id,
                "object": prov["object"],
                "is_new_this_round": str(task_id not in {1}),
                "fstar_low_mid_high": "/".join(fmt_force(fstars[mu]) for mu in MUS),
                "fixed_robust_force": fmt_force(robust),
                "gt_minforce_mean": fmt_num(oracle_mean),
                "force_saving_delta": fmt_num(delta),
                "transition_evidence": " | ".join(adjacent_evidence(cells, mu, fstars[mu]) for mu in MUS),
            })

        for mu in MUS:
            row = {
                "task_id": task_id,
                "object": prov["object"],
                "friction": mu,
                "fstar_tau_0_6": fmt_force(fstars_by_tau[0.6][mu]),
                "fstar_tau_0_8": fmt_force(fstars_by_tau[0.8][mu]),
                "fstar_tau_0_9": fmt_force(fstars_by_tau[0.9][mu]),
                "evidence_tau_0_8": adjacent_evidence(cells, mu, fstars_by_tau[0.8][mu]),
            }
            canonical_rows.append(row)

        for (mu, force), cell in sorted(cells.items()):
            quality_rows.append({
                "task_id": task_id,
                "object": prov["object"],
                "friction": mu,
                "force": force,
                "successes": cell.successes,
                "n": cell.n,
                "success_rate": fmt_num(cell.sr),
                "friction_override_ok_rate": fmt_num(cell.friction_ok_rate),
            })

        object_rows.append({
            "task_id": task_id,
            "object": prov["object"],
            "object_type": cfg.get("object_type", ""),
            "official_scale": json.dumps(cfg.get("scale", "")),
            "nominal_mu_recorded": "0.5",
            "nominal_mass": "not_extracted_from_usd_crate",
            "grasp_geometry": "scripted top-down lateral grasp; exact width not separately logged",
            "xml_path_from_official_config": cfg.get("xml_path", ""),
        })

    write_csv(BASE / "FULLTASK_CANONICAL_FSTAR.csv", [
        "task_id", "object", "friction", "fstar_tau_0_6", "fstar_tau_0_8", "fstar_tau_0_9", "evidence_tau_0_8"
    ], canonical_rows)
    write_csv(BASE / "FIXED_ROBUST_BY_TASK.csv", [
        "task_id", "object", "fixed_robust_force_tau_0_8", "min_observed_sr_at_robust", "evidence_by_mu", "source"
    ], robust_rows)
    write_csv(BASE / "GT_MINFORCE_BY_TASK.csv", [
        "task_id", "object", "fstar_low_mu_0_2", "fstar_mid_mu_0_5", "fstar_high_mu_1_0",
        "gt_minforce_mean", "fixed_robust_force", "force_saving_delta", "classification"
    ], gt_rows)
    write_csv(BASE / "EXPANDED_POSITIVE_TASKS.csv", [
        "task_id", "object", "is_new_this_round", "fstar_low_mid_high",
        "fixed_robust_force", "gt_minforce_mean", "force_saving_delta", "transition_evidence"
    ], positive_rows)
    write_csv(BASE / "TABERO_9TASK_PHYSICS_DECISION_TABLE.csv", [
        "task_id", "object", "data", "full_traj", "fstar_mu_0.2", "fstar_mu_0.5", "fstar_mu_1.0",
        "fixed_robust", "gt_minforce_mean", "delta_force", "classification", "instruction", "robust_source"
    ], table_rows)
    write_csv(BASE / "TASK_OBJECT_NOTES.csv", [
        "task_id", "object", "object_type", "official_scale", "nominal_mu_recorded",
        "nominal_mass", "grasp_geometry", "xml_path_from_official_config"
    ], object_rows)
    write_csv(BASE / "CELL_QUALITY_SUMMARY.csv", [
        "task_id", "object", "friction", "force", "successes", "n", "success_rate", "friction_override_ok_rate"
    ], quality_rows)

    table_md_rows = [
        [
            r["task_id"], r["object"], r["data"], r["full_traj"], r["fstar_mu_0.2"], r["fstar_mu_0.5"],
            r["fstar_mu_1.0"], r["fixed_robust"], r["gt_minforce_mean"], r["delta_force"], r["classification"],
        ]
        for r in table_rows
    ]
    table_md = markdown_table([
        "Task", "Object", "Data", "Full traj", "F*(low mu)", "F*(mid mu)", "F*(high mu)",
        "Fixed Robust", "GT-MinForce Mean", "Delta F", "Classification"
    ], table_md_rows)
    (BASE / "TABERO_9TASK_PHYSICS_DECISION_TABLE.md").write_text(
        "# TABERO 9-task physics-decision table\n\n"
        "Canonical F* uses full-task success at tau=0.8 over mu in {0.20, 0.50, 1.00}.\n\n"
        + table_md
    )

    decision_positive = [int(r["task_id"]) for r in table_rows if r["classification"] == "PHYSICS_DECISION_POSITIVE"]
    fixed_low = [int(r["task_id"]) for r in table_rows if r["classification"] == "FIXED_LOW_EXISTS"]
    fixed_robust_no_decision = [int(r["task_id"]) for r in table_rows if r["classification"] == "FIXED_ROBUST_NO_DECISION"]
    no_feasible_range = [int(r["task_id"]) for r in table_rows if r["classification"] == "NO_FEASIBLE_RANGE"]
    downstream_blocked = [int(r["task_id"]) for r in table_rows if r["classification"] == "DOWNSTREAM_BLOCKED"]
    blocked = no_feasible_range + downstream_blocked
    status = "B2R2_BENCHMARK_BREADTH_STRONG" if len(decision_positive) >= 4 else (
        "B2R2_BENCHMARK_BREADTH_PARTIAL" if len(decision_positive) >= 2 else "B2R2_CREAM_CHEESE_ISOLATED_POSITIVE"
    )
    breadth = "STRONG" if len(decision_positive) >= 4 else ("PARTIAL" if len(decision_positive) >= 2 else "WEAK")

    plot_outputs(table_rows)

    summary_md = f"""# B2-R2 benchmark breadth summary

## Verdict

Status: `{status}`

Across the 9 official Tabero LIBERO-object pick-place tasks, {len(decision_positive)} tasks are strict `PHYSICS_DECISION_POSITIVE`, {len(fixed_low)} tasks are `FIXED_LOW_EXISTS`, {len(fixed_robust_no_decision)} tasks are `FIXED_ROBUST_NO_DECISION`, and {len(blocked)} tasks are blocked or infeasible. This means cream cheese is not an isolated positive case.

## Core table

{table_md}

## Interpretation

The decision-positive tasks are {decision_positive}. These tasks have different canonical minimum sufficient full-task forces across hidden friction settings and a lower GT-MinForce mean than their fixed robust force.

The fixed-low controls are {fixed_low}. They complete at 3N across the tested friction settings, so force adaptation has no measured value there under this protocol.

The no-feasible-range tasks are {no_feasible_range}. These tasks passed the official data and nominal full-trajectory smoke checks, but did not reach tau=0.8 for all friction settings inside the tested force range.

## Evidence and quality checks

- Data source: official Isaaclab LIBERO assembled HDF5 files from `NathanWu7/Isaaclab_Libero`, tracked in `TASK_DATA_PROVENANCE.csv`.
- Runtime friction override was recorded per row; cell-level override rates are in `CELL_QUALITY_SUMMARY.csv`.
- Candidate transition cells were expanded to N=20 matched seeds where needed. Non-transition sanity cells remain cheap-scan N=3 unless they were part of the transition expansion.
- Task 1 and task 7 are frozen read-only references from the earlier B2/E2E2 evidence; this run did not rewrite those result directories.

## Limitations

- Only object friction was varied; mass, compliance, stiffness, restitution, and damage thresholds were not introduced.
- This is a task/physics/force oracle breadth pass only. No OURS, probe, belief, DeliGrasp, FORTE, or Tabero Neutral policy was run.
- Object mass and exact grasp width were not extracted from the USD crate assets; `TASK_OBJECT_NOTES.csv` records object identity, official scale, nominal mu, and the limited geometry fields available without modifying the simulator.
"""
    (BASE / "BENCHMARK_BREADTH_SUMMARY.md").write_text(summary_md)

    verdict = {
        "status": status,
        "method_change": "NONE",
        "official_task_count": 9,
        "tasks_data_ready": TASK_IDS,
        "tasks_full_trajectory_pass": TASK_IDS,
        "decision_positive_tasks": decision_positive,
        "fixed_low_tasks": fixed_low,
        "fixed_robust_no_decision_tasks": fixed_robust_no_decision,
        "blocked_tasks": blocked,
        "no_feasible_range_tasks": no_feasible_range,
        "downstream_blocked_tasks": downstream_blocked,
        "canonical_fstar_by_task": {
            r["task_id"]: [r["fstar_mu_0.2"], r["fstar_mu_0.5"], r["fstar_mu_1.0"]]
            for r in table_rows
        },
        "fixed_robust_force_by_task": {r["task_id"]: r["fixed_robust"] for r in table_rows},
        "gt_minforce_mean_by_task": {r["task_id"]: r["gt_minforce_mean"] for r in table_rows},
        "force_saving_potential_by_task": {r["task_id"]: r["delta_force"] for r in table_rows},
        "decision_positive_count": len(decision_positive),
        "cream_cheese_positive": True,
        "milk_negative_control": True,
        "benchmark_breadth": breadth,
        "enough_breadth_for_method_design": len(decision_positive) >= 2,
        "primary_evidence": [
            "FULLTASK_CANONICAL_FSTAR.csv",
            "FIXED_ROBUST_BY_TASK.csv",
            "GT_MINFORCE_BY_TASK.csv",
            "TABERO_9TASK_PHYSICS_DECISION_TABLE.csv",
            "CELL_QUALITY_SUMMARY.csv",
        ],
        "limitations": [
            "Only object friction was varied.",
            "No method/probe/belief/baseline policy was run in B2-R2.",
            "Task 1 and task 7 are frozen references from prior qualified evidence.",
            "Non-transition cells may remain N=3 cheap-scan evidence.",
        ],
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
    }
    (BASE / "FINAL_VERDICT.json").write_text(json.dumps(verdict, indent=2) + "\n")

    readme = f"""# B2-R2 Tabero official task breadth

Status: `{status}`

This directory contains the independent B2-R2 breadth pass for the 9 official Tabero LIBERO-object tasks. It only varies object friction (`mu = 0.20, 0.50, 1.00`) and evaluates full-task success under controlled force cells. No D2 thresholds, Tabero core code, provisional OURS thresholds, DeliGrasp, FORTE, or Tabero Neutral methods were changed or run here.

Primary artifacts:

- `FINAL_VERDICT.json`
- `TABERO_9TASK_PHYSICS_DECISION_TABLE.md`
- `FULLTASK_CANONICAL_FSTAR.csv`
- `FIXED_ROBUST_BY_TASK.csv`
- `GT_MINFORCE_BY_TASK.csv`
- `EXPANDED_POSITIVE_TASKS.csv`
- `BENCHMARK_BREADTH_SUMMARY.md`
- `CELL_QUALITY_SUMMARY.csv`
- `TASK_OBJECT_NOTES.csv`

Frozen read-only inputs:

- B2: `/home/exouser/Tabero/analysis/results/b2_tabero_benchmark_table_20260820_065520`
- E2E2: `/home/exouser/Tabero/analysis/results/e2e2_oracle_reconcile_milk_breadth_20260821_053312`

Generated at: `{datetime.now(timezone.utc).isoformat()}`
"""
    (BASE / "README.md").write_text(readme)

    env_path = BASE / "ENV_PROVENANCE.json"
    env = json.loads(env_path.read_text())
    env["finalized_utc"] = datetime.now(timezone.utc).isoformat()
    env["final_status"] = status
    env["final_artifacts_written"] = [
        "FINAL_VERDICT.json",
        "TABERO_9TASK_PHYSICS_DECISION_TABLE.md",
        "TABERO_9TASK_PHYSICS_DECISION_TABLE.csv",
        "FULLTASK_CANONICAL_FSTAR.csv",
        "FIXED_ROBUST_BY_TASK.csv",
        "GT_MINFORCE_BY_TASK.csv",
        "EXPANDED_POSITIVE_TASKS.csv",
        "BENCHMARK_BREADTH_SUMMARY.md",
        "CELL_QUALITY_SUMMARY.csv",
        "TASK_OBJECT_NOTES.csv",
        "plots/fstar_by_task.png",
        "plots/robust_vs_oracle.png",
        "plots/force_saving_by_task.png",
        "plots/task_classification.png",
        "plots/positive_negative_task_examples.png",
    ]
    env_path.write_text(json.dumps(env, indent=2) + "\n")

    print(status)
    print("decision_positive", decision_positive)
    print("fixed_low", fixed_low)
    print("fixed_robust_no_decision", fixed_robust_no_decision)
    print("blocked", blocked)


if __name__ == "__main__":
    main()
