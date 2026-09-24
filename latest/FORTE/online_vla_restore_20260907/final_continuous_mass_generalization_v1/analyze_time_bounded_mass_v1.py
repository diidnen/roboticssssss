#!/usr/bin/env python3
"""Frozen primary analysis and report for the 216-branch MASS subset."""
from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from time_bounded_mass_common import HERE, load_frozen_subset, read, sha256


def write_new_json(path: Path, value) -> None:
    if path.exists(): raise FileExistsError(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")


def csv_rows(path: Path):
    with path.open() as stream: return list(csv.DictReader(stream))


def rank(values):
    values = np.asarray(values, float); order = np.argsort(values, kind="mergesort"); result = np.empty(len(values), float); i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]: j += 1
        result[order[i:j]] = (i + j - 1) / 2; i = j
    return result


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0: return None
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    required_outputs = [HERE / name for name in (
        "TIME_BOUNDED_MASS_CLAIM_AUDIT.json", "FINAL_TIME_BOUNDED_MASS_REPORT.md",
        "TABLE_TIME_BOUNDED_MASS_PRIMARY.csv", "FIGURE_TIME_BOUNDED_MASS_COARSE_GRID.pdf",
        "FINAL_TIME_BOUNDED_MASS_TERMINAL.json",
    )]
    if any(path.exists() for path in required_outputs): raise RuntimeError("time-bounded analysis already exists; refusing rerun")
    subset, subset_sha = load_frozen_subset()
    lock = read(HERE / "TIME_BOUNDED_MASS_FEASIBILITY_CHECKPOINT_SELECTION_LOCK.json")
    if lock.get("postheldout_analyzer_sha256") != sha256(Path(__file__)):
        raise RuntimeError("post-HELDOUT analyzer changed after checkpoint selection")
    data_audit = read(HERE / "TIME_BOUNDED_MASS_DATA_AUDIT.json")
    model = read(HERE / "TIME_BOUNDED_MASS_MODEL_QUALIFICATION.json")
    heldout = read(HERE / "TIME_BOUNDED_MASS_HELDOUT_RESULTS.json")
    belief = read(HERE / "MASS_BELIEF_QUALIFICATION.json")
    reuse = read(HERE / "TIME_BOUNDED_MASS_REUSE_SUMMARY.json")
    pause = read(HERE / "ORIGINAL_648_QUEUE_PAUSE_AUDIT.json")
    incidents = read(HERE / "TIME_BOUNDED_MASS_INFRASTRUCTURE_INCIDENTS.json")
    if data_audit["status"] != "PASS" or data_audit["valid_branches"] != 216:
        raise RuntimeError("final 216-row data audit did not pass")
    if heldout["subset_sha256"] != subset_sha: raise RuntimeError("heldout/subset mismatch")

    predictions = csv_rows(HERE / "TIME_BOUNDED_MASS_HELDOUT_PREDICTIONS.csv")
    prediction_map = {(row["context_id"], float(row["force_N"])): float(row["p_success"]) for row in predictions}
    primary_rows = []
    for spec in subset["jobs"]:
        result = read(HERE / spec["relative_branch_path"] / "BRANCH_RESULT.json")
        outcome = result["outcome"]
        primary_rows.append({
            "queue_index": spec["queue_index"], "split": spec["split"], "root": spec["root"],
            "task": spec["task"], "context_id": spec["context_id"], "mass_kg": spec["mass_kg"],
            "force_N": spec["force_N"], "full_task_success_y": int(result["full_task_success_y"]),
            "lift_success": int(outcome.get("lift_success", 0)), "dropped": int(outcome.get("dropped", 0)),
            "place_success": int(outcome.get("place_success", 0)),
            "early_terminal": int(result.get("steps", 350) < 350), "steps": result.get("steps"),
            "branch_path": str(HERE / spec["relative_branch_path"]),
            "branch_result_sha256": sha256(HERE / spec["relative_branch_path"] / "BRANCH_RESULT.json"),
            "heldout_p_success": prediction_map.get((spec["context_id"], float(spec["force_N"])), ""),
        })
    table_path = HERE / "TABLE_TIME_BOUNDED_MASS_PRIMARY.csv"
    with table_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(primary_rows[0])); writer.writeheader(); writer.writerows(primary_rows)

    held_rows = [row for row in primary_rows if row["split"] == "HELDOUT"]
    aggregate = []
    for task in subset["tasks"]:
        for mass in subset["masses_kg"]:
            for force in subset["forces_N"]:
                group = [row for row in held_rows if row["task"] == task and row["mass_kg"] == mass and row["force_N"] == force]
                pred = [float(row["heldout_p_success"]) for row in group]
                aggregate.append({"task": task, "mass_kg": mass, "force_N": force, "n": len(group),
                                  "full_success_rate": float(np.mean([row["full_task_success_y"] for row in group])),
                                  "drop_rate": float(np.mean([row["dropped"] for row in group])),
                                  "mean_p_success": float(np.mean(pred))})
    task_force_mass_rho = []
    for task in subset["tasks"]:
        for force in subset["forces_N"]:
            group = [row for row in aggregate if row["task"] == task and row["force_N"] == force]
            task_force_mass_rho.append({"task": task, "force_N": force,
                                        "spearman_mass_vs_predicted_success": spearman([r["mass_kg"] for r in group], [r["mean_p_success"] for r in group])})
    task_force_sensitivity = []
    for task in subset["tasks"]:
        values = [row["p5_minus_p3"] for row in heldout["force_sensitivity_by_context"] if row["task"] == task]
        task_force_sensitivity.append({"task": task, "mean_p5_minus_p3": float(np.mean(values))})

    palette = {3.0: "#0072B2", 4.0: "#E69F00", 5.0: "#009E73"}
    markers = {3.0: "o", 4.0: "s", 5.0: "^"}
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), sharex=True, sharey=True)
    for axis, task in zip(axes.ravel(), subset["tasks"]):
        for force in subset["forces_N"]:
            group = sorted([row for row in aggregate if row["task"] == task and row["force_N"] == force], key=lambda row: row["mass_kg"])
            axis.plot([r["mass_kg"] for r in group], [r["mean_p_success"] for r in group],
                      color=palette[force], marker=markers[force], label=f"{force:g} N predicted")
            axis.scatter([r["mass_kg"] for r in group], [r["full_success_rate"] for r in group],
                         facecolors="none", edgecolors=palette[force], marker=markers[force], s=60, linewidths=1.2)
        axis.set_title(f"Task {task}"); axis.grid(alpha=.25); axis.set_ylim(-.05, 1.05)
    axes[1, 0].set_xlabel("Object mass (kg)"); axes[1, 1].set_xlabel("Object mass (kg)")
    axes[0, 0].set_ylabel("Success probability / rate"); axes[1, 0].set_ylabel("Success probability / rate")
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Held-out root: coarse mass-conditioned force grid\nfilled lines=model; open markers=observed full-task rate")
    fig.tight_layout(rect=(0, .09, 1, .94)); figure_path = HERE / "FIGURE_TIME_BOUNDED_MASS_COARSE_GRID.pdf"; fig.savefig(figure_path); plt.close(fig)

    predicted_ranges = []
    for task in subset["tasks"]:
        for force in subset["forces_N"]:
            values = [row["mean_p_success"] for row in aggregate if row["task"] == task and row["force_N"] == force]
            predicted_ranges.append(max(values) - min(values))
    mass_informative = float(np.mean(predicted_ranges)) > 1e-6
    force_by_task = [row["mean_p5_minus_p3"] for row in task_force_sensitivity]
    if model["qualified"] and mass_informative:
        mass_feasibility_claim = "SUPPORTED"
    elif mass_informative:
        mass_feasibility_claim = "MIXED"
    else:
        mass_feasibility_claim = "NOT_SUPPORTED"
    if model["qualified"] and min(force_by_task) >= 0:
        force_claim = "SUPPORTED"
    elif heldout["mean_predicted_p5_minus_p3"] >= 0:
        force_claim = "MIXED"
    else:
        force_claim = "NOT_SUPPORTED"
    claims = {
        "schema": "TIME_BOUNDED_MASS_CLAIM_AUDIT_V1",
        "COARSE_MASS_COVERAGE_CLAIM": "SUPPORTED",
        "COARSE_MASS_CONDITIONED_FEASIBILITY": mass_feasibility_claim,
        "COARSE_ADAPTIVE_FORCE_ALLOCATION": force_claim,
        "ROOT_DISJOINT_HELDOUT_GENERALIZATION": "SUPPORTED" if model["qualified"] else "NOT_SUPPORTED",
        "MASS_PHYSICAL_BELIEF_INFORMATIVE": "SUPPORTED" if belief["qualified"] else "NOT_SUPPORTED",
        "FINE_FORCE_CALIBRATION_CLAIM": "NOT_TESTED",
        "CONTINUOUS_FORCE_RESPONSE_CLAIM": "LIMITED_BY_COARSE_GRID",
        "UNSEEN_MASS_GENERALIZATION": "NOT_TESTED",
        "ARBITRARY_CONTINUOUS_MASS_GENERALIZATION": "NOT_TESTED",
        "ONLINE_VLA_TRANSFER": "NOT_TESTED",
        "SAME_ARCHITECTURE_WORKS_FOR_FRICTION_AND_MASS": "MIXED" if belief["qualified"] and model["qualified"] else "NOT_SUPPORTED",
        "JOINT_FRICTION_MASS_REASONING": "NOT_TESTED",
        "REAL_ROBOT_MASS_GENERALIZATION": "NOT_TESTED",
        "evidence": {"model_qualification": model["status"], "mean_predicted_mass_range": float(np.mean(predicted_ranges)),
                     "mean_heldout_p5_minus_p3": heldout["mean_predicted_p5_minus_p3"],
                     "taskwise_p5_minus_p3": task_force_sensitivity, "task_force_mass_spearman": task_force_mass_rho},
    }
    claim_path = HERE / "TIME_BOUNDED_MASS_CLAIM_AUDIT.json"; write_new_json(claim_path, claims)

    scientific_failures = sum(row["full_task_success_y"] == 0 for row in primary_rows)
    current_superset_valid = reuse["total_previously_completed_valid_branches"] + (216 - reuse["reused_primary_branches"])
    terminal = {
        "TIME_BOUNDED_MASS_STATUS": "COMPLETE_PASS" if model["qualified"] else "COMPLETE_MODEL_NOT_QUALIFIED",
        "ORIGINAL_SUPERSET_TARGET": 648, "REDUCED_PRIMARY_TARGET": 216,
        "VALID_BRANCHES_BEFORE_REDUCTION": pause["valid_branch_count_at_pause"],
        "VALID_SUPERSET_AFTER_ACTIVE_DRAIN": reuse["total_previously_completed_valid_branches"],
        "REUSED_PRIMARY_BRANCHES": reuse["reused_primary_branches"],
        "AUXILIARY_PRESERVED_BRANCHES": reuse["auxiliary_preserved_branches"],
        "NEW_BRANCHES_EXECUTED": 216 - reuse["reused_primary_branches"],
        "FINAL_PRIMARY_VALID_BRANCHES": 216, "TRAIN_BRANCHES": 144, "VAL_BRANCHES": 36, "HELDOUT_BRANCHES": 36,
        "TASKS": subset["tasks"], "MASSES_KG": subset["masses_kg"], "FORCES_N": subset["forces_N"],
        "ROOTS": sum(subset["roots_by_split"].values(), []),
        "INFRASTRUCTURE_FAILURES": sum(len(incident["records"]) for incident in incidents["incidents"]
                                        if incident["classification"] == "PRE_PHYSICS_LAUNCH_CLAIM_SCHEMA_MISMATCH"),
        "INFRASTRUCTURE_INCIDENTS": len(incidents["incidents"]),
        "SCIENTIFIC_FAILURES": scientific_failures, "RESULT_DRIVEN_RERUNS": 0,
        "COARSE_MASS_COVERAGE_CLAIM": claims["COARSE_MASS_COVERAGE_CLAIM"],
        "FINE_FORCE_CALIBRATION_CLAIM": "NOT_TESTED",
        "CONTINUOUS_FORCE_RESPONSE_CLAIM": "LIMITED_BY_COARSE_GRID",
        "ORIGINAL_648_PROTOCOL_PRESERVED": "YES", "ORIGINAL_648_QUEUE_RESUMABLE": "YES",
        "AUTOMATIC_EXPANSION_TO_648": "NO", "REMAINING_BRANCHES_IF_648_LATER_REQUESTED": 648 - current_superset_valid,
    }
    terminal_path = HERE / "FINAL_TIME_BOUNDED_MASS_TERMINAL.json"; write_new_json(terminal_path, terminal)

    belief_m = belief["heldout_metrics"]; held_m = heldout["metrics"]
    report = f"""# Final time-bounded MASS report

## 1. Executive conclusion

The user-authorized 216-branch MASS subset is complete and the frozen reduced feasibility model qualification is **{model['status']}**. The evidence supports **{claims['COARSE_MASS_CONDITIONED_FEASIBILITY']}** coarse mass-conditioned feasibility and **{claims['COARSE_ADAPTIVE_FORCE_ALLOCATION']}** coarse adaptive allocation across 3/4/5 N. It does not test fine-force calibration, arbitrary continuous-mass interpolation, or reduced-checkpoint online-VLA transfer.

## 2. Protocol change and preservation

The primary denominator was reduced solely for the fixed end-of-day compute budget: every task, mass anchor, and root was retained, while forces were restricted to 3, 4, and 5 N. The frozen subset SHA-256 is `{subset_sha}`. The original 648 artifacts and four launcher processes remain preserved and resumable; automatic expansion is disabled.

## 3. Reduced design

The design contains tasks 0/1/5/6, masses 0.05/0.10/0.20 kg, TRAIN roots 181000–181003, VAL root 181010, HELDOUT root 181011, and forces 3/4/5 N: 144 TRAIN + 36 VAL + 36 HELDOUT = 216 branches.

## 4. Reuse and infrastructure audit

Before reduced collection, {reuse['total_previously_completed_valid_branches']} original-superset branches were valid. The reduced denominator reused {reuse['reused_primary_branches']}; {reuse['auxiliary_preserved_branches']} intermediate-force branches remain auxiliary-only. Four claim-schema attempts failed before physics, were losslessly archived, and were not scientific reruns. No valid branch was rerun.

## 5. Current architecture and physical query

The unchanged worker uses the restored P4-B contact-conditioned tangential shear query, fixed friction 0.5, requested total object mass with proportional inertia scaling, candidate-independent restored state, a phase-free 8×64 decision sequence, the current controlled full-task endpoint, and the unchanged GRU(10,64)+MLP(54,64)+head feasibility architecture. True mass is a training condition only and is not a deployable belief input.

## 6. Data quality

The final audit passed {data_audit['valid_branches']}/216 unique primary branches, exact 3/4/5-N sibling sets, exact split roots, binary valid full-task labels, state equality, material readback, and frozen source hashes. Intermediate-force outcomes are excluded from every primary aggregate.

## 7. Mass belief carried into planning

The previously locked continuous mass belief passed its predeclared gate on root 181011: MAE {belief_m['MAE']:.4f} kg, RMSE {belief_m['RMSE']:.4f} kg, bias {belief_m['bias']:+.4f} kg, Spearman {belief_m['Spearman']:.3f}, 90% coverage {belief_m['coverage']['0.9']:.3f}, and ranking accuracy {belief_m['pairwise_ranking']['accuracy']:.3f}. This evaluates the three training anchors on a held-out root, not unseen mass coefficients.

## 8. Feasibility model selection and HELDOUT

Checkpoint selection used TRAIN and VAL only. HELDOUT labels were opened once after the lock. HELDOUT n={held_m['n']}, NLL={held_m['NLL']:.4f}, prevalence NLL={held_m['prevalence_NLL']:.4f}, Brier={held_m['Brier']:.4f}, AUROC={held_m['AUROC']}, AUPRC={held_m['AUPRC']}, and ECE={held_m['ECE_10bin']:.4f}. Mean predicted p(5 N)-p(3 N) was {heldout['mean_predicted_p5_minus_p3']:+.4f}.

![Coarse held-out mass-force grid](FIGURE_TIME_BOUNDED_MASS_COARSE_GRID.pdf)

The figure shows frozen model predictions as filled lines and observed full-task rates as open markers. It is a three-anchor sufficiency study, not a dense calibration curve.

## 9. Scientific outcomes

Across all 216 primary branches, {scientific_failures} full-task failures were observed. These are scientific outcomes, including proof-bearing early drops where applicable, and none triggered recollection. Detailed row-level provenance is in `TABLE_TIME_BOUNDED_MASS_PRIMARY.csv`.

## 10. Claim audit

- Coarse mass coverage: **{claims['COARSE_MASS_COVERAGE_CLAIM']}**
- Coarse mass-conditioned feasibility: **{claims['COARSE_MASS_CONDITIONED_FEASIBILITY']}**
- Coarse adaptive force allocation: **{claims['COARSE_ADAPTIVE_FORCE_ALLOCATION']}**
- Fine-force calibration: **NOT_TESTED**
- Continuous force response: **LIMITED_BY_COARSE_GRID**
- Unseen-mass interpolation: **NOT_TESTED**
- Online-VLA transfer for the reduced checkpoint: **NOT_TESTED**
- Shared friction/mass interface claim: **{claims['SAME_ARCHITECTURE_WORKS_FOR_FRICTION_AND_MASS']}**

## 11. Paper recommendation

This result is appropriate as a coarse second-physics feasibility study. It should not replace the closed friction result or be presented as equivalent evidence strength to the 720-branch friction feasibility dataset. Without a reduced-checkpoint online-VLA qualification and exact-held-out mass interpolation, the strongest wording is: “The same continuous-belief and phase-free feasibility interface was instantiated for mass, with coarse root-disjoint evidence across three mass and force anchors.” Main-paper promotion of continuous unseen-mass force adaptation remains unwarranted from this time-bounded subset alone.

## 12. Reproducibility

Primary table SHA-256: `{sha256(table_path)}`. Figure SHA-256: `{sha256(figure_path)}`. Claim audit SHA-256: `{sha256(claim_path)}`. Selection and HELDOUT provenance are recorded in the checkpoint lock and one-shot HELDOUT result.
"""
    report_path = HERE / "FINAL_TIME_BOUNDED_MASS_REPORT.md"; report_path.write_text(report)
    print("\n".join(f"{key} = {json.dumps(value) if isinstance(value, (list, dict)) else value}" for key, value in terminal.items()))


if __name__ == "__main__":
    main()
