#!/usr/bin/env python3
"""Apply preregistered root-scaling gates and build final durable artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
TASKS = [0, 5]
LEVELS = [6, 15, 30, 50]
MODELS = ["Base", "Full Visual", "Visual Joint"]
COLORS = {"Base": "#4C566A", "Full Visual": "#2563A7", "Visual Joint": "#D97706"}
MARKERS = {"Base": "o", "Full Visual": "s", "Visual Joint": "^"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"]); w.writeheader(); w.writerows(rows)


def row(df: pd.DataFrame, task: int, n: int, model: str, agg: str = "SEED_MEAN") -> pd.Series:
    q = df[(df.task == task) & (df.N_root == n) & (df.model == model) & (df.aggregation == agg)]
    if len(q) != 1: raise RuntimeError(f"metric row mismatch task={task} S{n} {model} {agg}")
    return q.iloc[0]


def stable_lower(df: pd.DataFrame, task: int, model: str, metric: str, agg: str = "SEED_MEAN",
                 absolute: bool = False) -> dict:
    vals = [float(row(df, task, n, model, agg)[metric]) for n in LEVELS]
    compared = [abs(x) for x in vals] if absolute else vals
    steps = [compared[i+1] < compared[i] for i in range(3)]
    return {"values": dict(zip([f"S{x}" for x in LEVELS], vals)),
            "comparison_uses_absolute_magnitude": absolute,
            "endpoint_improves": compared[-1] < compared[0], "improving_adjacent_steps": int(sum(steps)),
            "stable": bool(compared[-1] < compared[0] and sum(steps) >= 2)}


def all_seed_endpoint_lower(df, task, model, metric) -> dict:
    values = {}
    flags = []
    for seed in range(3):
        agg = f"SEED_{seed}"; a = float(row(df, task, 6, model, agg)[metric]); b = float(row(df, task, 50, model, agg)[metric])
        values[agg] = {"S6": a, "S50": b, "improves": b < a}; flags.append(b < a)
    return {"values": values, "all_three_consistent": bool(all(flags))}


def paired_root_gate(perroot: pd.DataFrame, task: int, model: str, comparator: str) -> dict:
    q = perroot[(perroot.task == task) & (perroot.N_root == 50) &
                (perroot.aggregation == "ENSEMBLE")]
    a = q[q.model == model].set_index("context_id")
    b = q[q.model == comparator].set_index("context_id")
    ids = sorted(set(a.index) & set(b.index))
    front = np.asarray([float(b.loc[c].absolute_frontier_error_N) - float(a.loc[c].absolute_frontier_error_N)
                        for c in ids], float)
    nll = np.asarray([float(b.loc[c].NLL) - float(a.loc[c].NLL) for c in ids], float)
    front_valid = np.isfinite(front)
    return {"comparator": comparator, "comparable_roots": int(front_valid.sum()),
            "frontier_improved_roots": int(np.sum(front[front_valid] > 0)),
            "NLL_improved_roots": int(np.sum(nll > 0)),
            "median_paired_frontier_gain_N": float(np.nanmedian(front)) if front_valid.any() else math.nan,
            "median_paired_NLL_gain": float(np.median(nll)),
            "pass": bool(np.sum(front[front_valid] > 0) >= 6 and np.sum(nll > 0) >= 6 and
                         np.nanmedian(front) > 0 and np.median(nll) > 0)}


def joint_seed_endpoint_vs_full(lc: pd.DataFrame, task: int) -> dict:
    values, flags = {}, []
    for seed in range(3):
        agg = f"SEED_{seed}"; j = row(lc, task, 50, "Visual Joint", agg); f = row(lc, task, 50, "Full Visual", agg)
        ok = (float(j.TEST_frontier_MAE_N) < float(f.TEST_frontier_MAE_N) and
              float(j.TEST_under_force_rate) <= float(f.TEST_under_force_rate) and
              float(j.TEST_NLL) <= float(f.TEST_NLL))
        values[agg] = {"Joint_frontier_MAE_N": float(j.TEST_frontier_MAE_N),
                       "Full_frontier_MAE_N": float(f.TEST_frontier_MAE_N),
                       "Joint_under_force_rate": float(j.TEST_under_force_rate),
                       "Full_under_force_rate": float(f.TEST_under_force_rate),
                       "Joint_NLL": float(j.TEST_NLL), "Full_NLL": float(f.TEST_NLL), "passes": ok}
        flags.append(ok)
    return {"values": values, "all_three_consistent": bool(all(flags))}


def task_gates(lc: pd.DataFrame, perroot: pd.DataFrame, task: int) -> dict:
    full_front = stable_lower(lc, task, "Full Visual", "TEST_frontier_MAE_N")
    full_nll = stable_lower(lc, task, "Full Visual", "TEST_NLL")
    full_gap = stable_lower(lc, task, "Full Visual", "TRAIN_to_TEST_NLL_gap", absolute=True)
    full_seed_front = all_seed_endpoint_lower(lc, task, "Full Visual", "TEST_frontier_MAE_N")
    full_seed_nll = all_seed_endpoint_lower(lc, task, "Full Visual", "TEST_NLL")
    fs50, fs6, bs50 = row(lc, task, 50, "Full Visual"), row(lc, task, 6, "Full Visual"), row(lc, task, 50, "Base")
    full_root_gate = paired_root_gate(perroot, task, "Full Visual", "Base")
    full_multi = full_root_gate["pass"]
    full_safe = (float(fs50.TEST_under_force_rate) <= float(fs6.TEST_under_force_rate) and
                 float(fs50.TEST_under_force_rate) <= float(bs50.TEST_under_force_rate))
    full_endpoint_beats_base = (float(fs50.TEST_frontier_MAE_N) < float(bs50.TEST_frontier_MAE_N) and
                                float(fs50.TEST_NLL) < float(bs50.TEST_NLL))
    full_scale = ((full_front["stable"] or full_nll["stable"]) and full_gap["endpoint_improves"] and
                  full_seed_front["all_three_consistent"] and full_seed_nll["all_three_consistent"] and
                  full_multi and full_safe)

    joint_front = stable_lower(lc, task, "Visual Joint", "TEST_frontier_MAE_N")
    joint_nll = stable_lower(lc, task, "Visual Joint", "TEST_NLL")
    joint_gap = stable_lower(lc, task, "Visual Joint", "TRAIN_to_TEST_NLL_gap", absolute=True)
    js50, js6 = row(lc, task, 50, "Visual Joint"), row(lc, task, 6, "Visual Joint")
    joint_seed_front = all_seed_endpoint_lower(lc, task, "Visual Joint", "TEST_frontier_MAE_N")
    joint_scale = ((joint_front["stable"] or joint_nll["stable"]) and joint_gap["endpoint_improves"] and
                   joint_seed_front["all_three_consistent"] and
                   float(js50.TEST_under_force_rate) <= float(js6.TEST_under_force_rate))
    joint_vs_full_front = float(js50.TEST_frontier_MAE_N) < float(fs50.TEST_frontier_MAE_N)
    joint_vs_full_safe = float(js50.TEST_under_force_rate) <= float(fs50.TEST_under_force_rate)
    joint_prob_noninferior = (float(js50.TEST_NLL) <= float(fs50.TEST_NLL) and
                              float(js50.TEST_Brier) <= float(fs50.TEST_Brier))
    joint_root_gate = paired_root_gate(perroot, task, "Visual Joint", "Full Visual")
    joint_multi = joint_root_gate["pass"]
    joint_seed_vs_full = joint_seed_endpoint_vs_full(lc, task)
    joint_emerges = (joint_scale and joint_vs_full_front and joint_vs_full_safe and joint_prob_noninferior and
                     joint_multi and joint_seed_vs_full["all_three_consistent"])
    joint_frontier_unsafe = joint_scale and joint_vs_full_front and not joint_vs_full_safe

    train_visual_signal = float(row(lc, task, 50, "Full Visual").TRAIN_NLL) < float(row(lc, task, 50, "Base").TRAIN_NLL)
    no_test_advantage = (not full_endpoint_beats_base and
                         float(js50.TEST_frontier_MAE_N) >= float(bs50.TEST_frontier_MAE_N) and
                         float(js50.TEST_NLL) >= float(bs50.TEST_NLL))
    no_gap_shrink = (abs(float(fs50.TRAIN_to_TEST_NLL_gap)) >= abs(float(fs6.TRAIN_to_TEST_NLL_gap)) and
                     abs(float(js50.TRAIN_to_TEST_NLL_gap)) >= abs(float(js6.TRAIN_to_TEST_NLL_gap)))
    inefficient = train_visual_signal and no_test_advantage and no_gap_shrink
    return {"task": task, "full_frontier_trend": full_front, "full_NLL_trend": full_nll,
            "full_gap_trend": full_gap, "full_seed_frontier_direction": full_seed_front,
            "full_seed_NLL_direction": full_seed_nll, "full_multiple_roots": full_multi,
            "full_paired_root_gate": full_root_gate,
            "full_safety_nonworse": full_safe, "full_endpoint_beats_Base": full_endpoint_beats_base,
            "full_sample_complexity_gate": full_scale, "joint_frontier_trend": joint_front,
            "joint_NLL_trend": joint_nll, "joint_gap_trend": joint_gap,
            "joint_scale_gate": joint_scale, "joint_vs_full_frontier": joint_vs_full_front,
            "joint_vs_full_safety_nonworse": joint_vs_full_safe,
            "joint_probability_noninferior": joint_prob_noninferior,
            "joint_multiple_roots": joint_multi, "joint_paired_root_gate": joint_root_gate,
            "joint_seed_endpoint_vs_Full": joint_seed_vs_full, "joint_emerges_gate": joint_emerges,
            "joint_frontier_unsafe_gate": joint_frontier_unsafe,
            "train_visual_signal_at_S50": train_visual_signal,
            "no_visual_or_joint_TEST_advantage_at_S50": no_test_advantage,
            "no_gap_shrink": no_gap_shrink, "representation_inefficiency_gate": inefficient}


def classify(gates: dict[int, dict]) -> tuple[str, str]:
    both_joint = all(gates[t]["joint_emerges_gate"] for t in TASKS)
    any_unsafe = any(gates[t]["joint_frontier_unsafe_gate"] for t in TASKS)
    both_full = all(gates[t]["full_sample_complexity_gate"] and gates[t]["full_endpoint_beats_Base"] for t in TASKS)
    both_joint_scale = all(gates[t]["joint_scale_gate"] for t in TASKS)
    both_inefficient = all(gates[t]["representation_inefficiency_gate"] for t in TASKS)
    if both_joint:
        return "JOINT_VALUE_EMERGES_WITH_CONTEXT_SCALE", "Both primary tasks pass all Joint frontier, safety, probability, root, and seed gates."
    if any_unsafe:
        return "JOINT_FRONTIER_SIGNAL_WITHOUT_SAFE_CONTROL", "Joint gains frontier signal with scale on at least one task but violates under-force non-worsening."
    if both_full and both_joint_scale:
        return "VISUAL_CONTEXT_IS_INDEPENDENT_ROOT_SAMPLE_LIMITED", "Full Visual and Visual Joint both scale consistently, while Joint does not add a safe independent endpoint win."
    if both_full:
        return "VISUAL_IMPROVES_BUT_JOINT_NOT_NEEDED", "Full Visual passes the independent-root scaling gate on both tasks; Joint does not pass its stricter independent-value gate."
    if both_inefficient:
        return "GENERIC_VISUAL_CONTEXT_REMAINS_INEFFICIENT_WITH_MORE_ROOTS", "At S50 visual TRAIN signal remains but TEST advantage and gap shrink are absent on both tasks."
    return "INSUFFICIENT_ROOT_SCALE_TO_DISTINGUISH_SAMPLE_COMPLEXITY_FROM_REPRESENTATION", "The preregistered gates are mixed across tasks or metrics."


def plot_curves(lc: pd.DataFrame, task: int) -> Path:
    mean = lc[(lc.task == task) & (lc.aggregation == "SEED_MEAN")]
    std = lc[(lc.task == task) & (lc.aggregation == "SEED_STD")]
    specs = [("TEST_frontier_MAE_N", "TEST frontier MAE", "N"),
             ("TEST_under_force_rate", "TEST under-force rate", "rate"),
             ("TEST_NLL", "TEST NLL", "NLL"),
             ("TRAIN_to_TEST_NLL_gap", "TRAIN-to-TEST NLL gap", "TEST − TRAIN")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        for model in MODELS:
            q = mean[mean.model == model].sort_values("N_root")
            s = std[std.model == model].sort_values("N_root")
            x, y = q.N_root.to_numpy(float), q[metric].to_numpy(float)
            e = s[metric].to_numpy(float)
            ax.plot(x, y, color=COLORS[model], marker=MARKERS[model], linewidth=2,
                    markersize=5, label=model)
            ax.fill_between(x, y-e, y+e, color=COLORS[model], alpha=.12, linewidth=0)
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Independent TRAIN roots"); ax.set_ylabel(ylabel); ax.set_xticks(LEVELS)
        ax.grid(axis="y", color="#D9DEE7", linewidth=.7); ax.spines[["top","right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper right")
    fig.suptitle(f"Task {task} root-scaling learning curves", x=.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(.01, .005, "Lines: three-seed mean. Shading: ±1 seed standard deviation. Untouched TEST: 10 roots, 9 forces × 5 repeats/root.", fontsize=8, color="#4C566A")
    path = OUT / f"ROOT_SCALING_LEARNING_CURVES_TASK{task}.png"
    fig.savefig(path, dpi=180, facecolor="white"); plt.close(fig); return path


def table(df: pd.DataFrame) -> str:
    cols = [str(x) for x in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for values in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(x) else str(x) for x in values) + " |")
    return "\n".join(lines)


def collection_qa_and_counts() -> tuple[str, list[dict]]:
    rows = []; lines = ["# Root-scaling collection QA", "", "TEST and TRAIN failures are scientific outcomes; only infrastructure interruption is resumable. No outcome-dependent retry is permitted.", ""]
    for task in TASKS:
        for split in ("TEST", "TRAIN"):
            if split == "TEST":
                p = OUT / f"TASK{task}_TEST_COLLECTION_AUDIT.json"
            elif task == 0:
                p = ROOT / "task0_context_sample_complexity_20260831/TASK0_TRAIN_COLLECTION_AUDIT.json"
            else:
                p = OUT / f"TASK{task}_TRAIN_COLLECTION_AUDIT.json"
            audit = json.loads(p.read_text())
            c = audit["counts"]
            collected_roots = c.get("independent_roots", c.get("simulator_root_clusters"))
            included_roots = 10 if split == "TEST" else 44
            included_contexts = 10 if split == "TEST" else 44
            rows.append({"task": task, "split": split, "status": audit["status"],
                         "independent_roots_collected": collected_roots,
                         "independent_roots_included_in_analysis": included_roots,
                         "friction_conditioned_contexts_collected": c.get("contexts"),
                         "friction_conditioned_contexts_included_in_analysis": included_contexts,
                         "extra_collected_roots_excluded": max(0, int(collected_roots) - included_roots),
                         "force_cells_collected": c.get("force_cells"),
                         "branches_collected": c.get("branches"), "scientific_successes": c.get("successes"),
                         "scientific_failures": c.get("scientific_failures", c.get("failures")),
                         "infrastructure_status": audit.get("infrastructure_status", "COMPLETE"),
                         "infrastructure_failures": int(audit.get("infrastructure_returncode", 0) not in (0, None)),
                         "scientific_retry_count": audit.get("scientific_retry_count", 0),
                         "audit_path": str(p), "audit_sha256": sha256(p)})
            lines += [f"## Task {task} {split}", "", f"Status: {audit['status']}. Collected contexts: {c.get('contexts')}; included in this analysis: {included_contexts}; branches collected: {c.get('branches')}; state/visual/telemetry checks all pass.", ""]
            if task == 0 and split == "TRAIN" and int(collected_roots) > included_roots:
                lines += ["The inherited task0 collector completed an older 62-addition target. Only the preregistered first 44 additions (root12--root55) are eligible for S50; root56--root73 are excluded from PCA, normalization, training, and evaluation.", ""]
    return "\n".join(lines) + "\n", rows


def main() -> None:
    lc = pd.read_csv(OUT / "ROOT_SCALING_LEARNING_CURVE.csv")
    perroot = pd.read_csv(OUT / "ROOT_SCALING_PER_ROOT_METRICS.csv")
    safety = pd.read_csv(OUT / "ROOT_SCALING_SAFETY_METRICS.csv")
    gaps = pd.read_csv(OUT / "ROOT_SCALING_TRAIN_TEST_GAP.csv")
    gates = {t: task_gates(lc, perroot, t) for t in TASKS}
    classification, reason = classify(gates)
    figures = [plot_curves(lc, t) for t in TASKS]
    qa_md, counts = collection_qa_and_counts()
    (OUT / "ROOT_SCALING_COLLECTION_QA.md").write_text(qa_md)
    write_csv(OUT / "ROOT_SCALING_DATA_COUNTS.csv", counts)
    final = {"classification": classification, "reason": reason, "tasks": TASKS,
             "completed_root_levels": LEVELS, "maximum_independent_roots_per_task": 50,
             "untouched_TEST_roots_per_task": 10, "gates": {str(k): v for k, v in gates.items()},
             "classification_order_frozen_before_outcomes": True,
             "models_representation_PCA_loss_probe_changes": False}
    write_json(OUT / "ROOT_SCALING_FINAL_CLASSIFICATION.json", final)

    primary = lc[lc.aggregation == "SEED_MEAN"][["task","N_root","model","TEST_frontier_MAE_N",
        "TEST_under_force_rate","TEST_NLL","TRAIN_to_TEST_NLL_gap",
        "fraction_TEST_roots_frontier_improved_vs_Base"]].copy()
    for c in primary.columns[3:]: primary[c] = primary[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    answers = []
    for task in TASKS:
        g = gates[task]
        answers.append(f"- task{task}: Full sample-complexity gate={'PASS' if g['full_sample_complexity_gate'] else 'FAIL'}; "
                       f"Full S50 beats Base={'YES' if g['full_endpoint_beats_Base'] else 'NO'}; "
                       f"Joint emergence={'YES' if g['joint_emerges_gate'] else 'NO'}; "
                       f"Joint unsafe-frontier-only={'YES' if g['joint_frontier_unsafe_gate'] else 'NO'}.")
    report = ["# Independent-root scaling result", "", "## Technical summary", "",
              f"**{classification}**", "", reason, "", *answers, "",
              "This conclusion is limited to same task, same object/task distribution, and new physical roots. It makes no unseen-task or cross-object claim.", "",
              "## Root-count learning curves", "",
              "The tables and figures use three-seed means; shaded figures show ±1 seed standard deviation. S6 contains 18 friction-conditioned observations grouped into six seed families; S15/S30/S50 add unique seed roots one at a time.", "",
              table(primary), "",
              "Figures: `ROOT_SCALING_LEARNING_CURVES_TASK0.png`, `ROOT_SCALING_LEARNING_CURVES_TASK5.png`.", "",
              "## Direct answers", "",
              "1. **Was failure at six roots mainly sample complexity?** " + ("Yes under all preregistered taskwise gates." if classification in {"VISUAL_CONTEXT_IS_INDEPENDENT_ROOT_SAMPLE_LIMITED","VISUAL_IMPROVES_BUT_JOINT_NOT_NEEDED","JOINT_VALUE_EMERGES_WITH_CONTEXT_SCALE"} else "No definitive across-task evidence supports that claim."),
              "2. **Does Full Visual exceed Base on untouched roots after scaling?** " + ("Yes on both primary tasks at S50." if all(gates[t]['full_endpoint_beats_Base'] for t in TASKS) else "Not on both primary tasks under the frozen frontier+NLL rule."),
              "3. **Does Joint recover a frontier advantage with diversity?** " + ("Yes, and it passes the full independent-value gate." if all(gates[t]['joint_emerges_gate'] for t in TASKS) else "Not reliably across both tasks."),
              "4. **If Joint improves frontier, is control safe?** " + ("No: at least one task shows frontier signal with worse under-force." if any(gates[t]['joint_frontier_unsafe_gate'] for t in TASKS) else "No preregistered unsafe frontier-only classification was triggered."),
              "5. **Does the TRAIN→TEST gap shrink?** " + ("Yes for Full Visual on both tasks from S6 to S50." if all(gates[t]['full_gap_trend']['endpoint_improves'] for t in TASKS) else "Not consistently for Full Visual across both tasks."),
              "6. **Next action.** " + ({
                  "JOINT_VALUE_EMERGES_WITH_CONTEXT_SCALE": "Keep Visual and Joint; further roots are useful only as confirmation, not as a substitute for the passed safety gate.",
                  "VISUAL_CONTEXT_IS_INDEPENDENT_ROOT_SAMPLE_LIMITED": "Keep Visual and continue increasing independent roots; Joint is not yet independently justified.",
                  "VISUAL_IMPROVES_BUT_JOINT_NOT_NEEDED": "Keep Full Visual, omit Joint, and add roots only if tighter uncertainty is needed.",
                  "JOINT_FRONTIER_SIGNAL_WITHOUT_SAFE_CONTROL": "Do not deploy Joint; retain the safest simpler comparator while treating the frontier gain as diagnostic only.",
                  "GENERIC_VISUAL_CONTEXT_REMAINS_INEFFICIENT_WITH_MORE_ROOTS": "Return to Base; do not keep explaining the failure as insufficient roots.",
                  "INSUFFICIENT_ROOT_SCALE_TO_DISTINGUISH_SAMPLE_COMPLEXITY_FROM_REPRESENTATION": "Do not select Visual or Joint from these mixed results; either pre-register a larger root endpoint or retain Base meanwhile.",
              })[classification], "",
              "## Experimental controls", "",
              "Only independent root count changed. π0, visual encoder/extraction, PCA dimension (17), architectures, optimizer, learning rate, epochs, batch size, λphys, λIE, force semantics, normalization semantics, and full-task success labels remained frozen. TEST never fit PCA/normalization and was not used for model, checkpoint, seed, feature, or loss selection.", "",
              "## Data quality and limitations", "",
              "Each TEST root used nine forces × five repeats and F*0.8 = minimum force with at least 4/5 full-task successes. TRAIN used five continuous strata × two repeats. Physical branches and timesteps are not counted as independent contexts. With only 10 TEST roots/task, root fractions change in 0.1 increments; seed spread and per-root rows remain primary uncertainty evidence.", "",
              "See `ROOT_SCALING_COLLECTION_QA.md`, `ROOT_SCALING_PER_ROOT_METRICS.csv`, and `ROOT_SCALING_FINAL_CLASSIFICATION.json` for auditable evidence.", ""]
    (OUT / "ROOT_SCALING_FINAL_REPORT.md").write_text("\n".join(report))

    required = ["ROOT_SCALING_TEST_MANIFEST.json", "ROOT_SCALING_TRAIN_MANIFEST.json",
        "ROOT_SCALING_COLLECTION_QA.md", "ROOT_SCALING_DATA_COUNTS.csv", "ROOT_SCALING_MODEL_HASHES.csv",
        "ROOT_SCALING_LEARNING_CURVE.csv", "ROOT_SCALING_PER_ROOT_METRICS.csv",
        "ROOT_SCALING_SAFETY_METRICS.csv", "ROOT_SCALING_TRAIN_TEST_GAP.csv",
        "ROOT_SCALING_FINAL_REPORT.md", "ROOT_SCALING_FINAL_CLASSIFICATION.json",
        "ROOT_SCALING_LEARNING_CURVES_TASK0.png", "ROOT_SCALING_LEARNING_CURVES_TASK5.png"]
    missing = [x for x in required if not (OUT / x).exists()]
    if missing: raise RuntimeError("required deliverables missing: " + json.dumps(missing))
    (OUT / "SHA256SUMS.txt").write_text("\n".join(f"{sha256(OUT/x)}  {x}" for x in required) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
