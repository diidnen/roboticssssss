#!/usr/bin/env python3
"""Read-only forensic of Point-WM decisions on the current Task 1 / Task 6 population."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
RUN_DIR = ROOT / "activeforcing_sample_efficiency_60_20260901_073611"
PRIOR_DIR = ROOT / "activeforcing_probe_conditioned_wm_20260901_064627"
OUT_DIR = RUN_DIR / "task1_task6_wm_forensic"
PIPELINE_DIR = ROOT / "activeforcing_data_population_20260831_194336"
TASKS = (1, 6)
FORCE_PENALTY = 0.035


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    return value


def _records(df: pd.DataFrame):
    return _clean(df.to_dict(orient="records"))


def _pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"


def _pp(v: float) -> str:
    return f"{100.0 * v:+.2f} pp"


def load_population():
    sys.path.insert(0, str(ROOT))
    import run_probe_conditioned_wm as old  # noqa: PLC0415
    import run_activeforcing_sample_efficiency_60 as sparse  # noqa: PLC0415

    _, traces, meta, _, _, _, _ = sparse.load_population(OUT_DIR, "population")
    md, _ = sparse.population_frame(traces, meta)
    base = md.rename(
        columns={
            "task": "task_id",
            "root_id": "root_uid",
            "repeat": "repeat_id",
            "force_N": "force",
            "success": "success_full",
        }
    ).copy()
    return old, md, base


def shard_rows(old, base: pd.DataFrame, budget: int, fold: int, seed: int) -> pd.DataFrame:
    if budget == 60:
        path = RUN_DIR / "shards" / f"sparse60_fold{fold}_seed{seed}.npz"
        z = np.load(path, allow_pickle=True)
        held_idx = z["held_idx"].astype(int)
        ids = base.iloc[held_idx]["branch_id"].astype(str).to_numpy()
        du = z["du_direct"].astype(float)
        p_direct = z["p_direct"].astype(float)
        score_wm = z["score_point"].astype(float)
        correction = z["delta_point"].astype(float)
    else:
        path = PRIOR_DIR / "shards" / f"fold{fold}_seed{seed}.npz"
        z = np.load(path, allow_pickle=True)
        held_idx = z["held_idx"].astype(int)
        ids = base.iloc[held_idx]["branch_id"].astype(str).to_numpy()
        p_direct = z["p_scalar"].astype(float)
        du = z["du_scalar"].astype(float)
        score_wm = z["score_point"].astype(float)
        correction = z["delta_point"].astype(float)

    pred = pd.DataFrame(
        {
            "branch_id": ids,
            "du_saved": du,
            "p_direct": p_direct,
            "score_wm": score_wm,
            "correction": correction,
        }
    )
    held = base.merge(pred, on="branch_id", how="inner", validate="one_to_one")
    if budget == 120:
        fmax = held["task_id"].map(old.FMAX).to_numpy(dtype=float)
        force = held["force"].to_numpy(dtype=float)
        p = held["p_direct"].to_numpy(dtype=float)
        held["du"] = p * ((fmax - force) / fmax) + (1.0 - p) * -1.0
    else:
        held["du"] = held["du_saved"]
    held["residual_target"] = held["R_real"] - held["du"]
    held["budget"] = budget
    held["fold"] = fold
    held["seed"] = seed
    held["friction_band"] = held["context_id"].str.extract(r"_(low|mid|high)_mu", expand=False)
    return held


def analyse(old, base: pd.DataFrame):
    decision_rows = []
    branch_rows = []
    slope_rows = []

    for budget in (60, 120):
        for fold in range(3):
            for seed in range(3):
                held = shard_rows(old, base, budget, fold, seed)
                held = held[held["task_id"].isin(TASKS)].copy()
                branch_rows.append(held)

                keys = ["task_id", "root_uid", "context_id", "repeat_id"]
                for key, group in held.groupby(keys, sort=True):
                    g = group.sort_values("force", kind="mergesort").reset_index(drop=True)
                    direct_i = int(np.argmax(g["du"].to_numpy()))
                    wm_i = int(np.argmax(g["score_wm"].to_numpy()))
                    direct = g.iloc[direct_i]
                    wm = g.iloc[wm_i]
                    success_forces = g.loc[g["success_full"] == 1, "force"]
                    frontier = float(success_forces.min()) if len(success_forces) else np.nan
                    slope = float(np.polyfit(g["force"], g["correction"], 1)[0])
                    corr = float(np.corrcoef(g["force"], g["correction"])[0, 1])
                    slope_rows.append(
                        {
                            "budget": budget,
                            "fold": fold,
                            "seed": seed,
                            "task": f"task{key[0]}",
                            "root_uid": key[1],
                            "context_id": key[2],
                            "repeat_id": key[3],
                            "friction_band": direct["friction_band"],
                            "correction_force_slope": slope,
                            "correction_force_corr": corr,
                        }
                    )
                    d_success = int(direct["success_full"])
                    w_success = int(wm["success_full"])
                    if d_success == 1 and w_success == 0:
                        transition = "success_to_failure"
                    elif d_success == 0 and w_success == 1:
                        transition = "failure_to_success"
                    elif d_success == 1:
                        transition = "success_to_success"
                    else:
                        transition = "failure_to_failure"
                    decision_rows.append(
                        {
                            "budget": budget,
                            "fold": fold,
                            "seed": seed,
                            "task": f"task{key[0]}",
                            "root_uid": key[1],
                            "context_id": key[2],
                            "repeat_id": key[3],
                            "friction_band": direct["friction_band"],
                            "direct_force": float(direct["force"]),
                            "wm_force": float(wm["force"]),
                            "force_delta_N": float(wm["force"] - direct["force"]),
                            "direct_success": d_success,
                            "wm_success": w_success,
                            "delta_success": w_success - d_success,
                            "transition": transition,
                            "changed_force": int(direct["branch_id"] != wm["branch_id"]),
                            "frontier_force": frontier,
                            "direct_underforce": int(np.isfinite(frontier) and direct["force"] < frontier),
                            "wm_underforce": int(np.isfinite(frontier) and wm["force"] < frontier),
                            "direct_realized_utility": float(direct["R_real"]),
                            "wm_realized_utility": float(wm["R_real"]),
                            "delta_realized_utility": float(wm["R_real"] - direct["R_real"]),
                            "wm_correction_at_direct": float(direct["correction"]),
                            "wm_correction_at_wm": float(wm["correction"]),
                            "correction_advantage": float(wm["correction"] - direct["correction"]),
                            "direct_utility_penalty": float(wm["du"] - direct["du"]),
                            "correction_force_slope": slope,
                            "correction_force_corr": corr,
                        }
                    )

    decisions = pd.DataFrame(decision_rows)
    branches = pd.concat(branch_rows, ignore_index=True)
    slopes = pd.DataFrame(slope_rows)

    def decision_summary(g):
        changed = g[g["changed_force"] == 1]
        harms = g[g["transition"] == "success_to_failure"]
        rescues = g[g["transition"] == "failure_to_success"]
        return pd.Series(
            {
                "episodes": len(g),
                "changed_decisions": int(g["changed_force"].sum()),
                "changed_fraction": g["changed_force"].mean(),
                "direct_sr": g["direct_success"].mean(),
                "wm_sr": g["wm_success"].mean(),
                "delta_sr": g["delta_success"].mean(),
                "harms": len(harms),
                "rescues": len(rescues),
                "mean_force_delta_N": g["force_delta_N"].mean(),
                "changed_force_delta_N": changed["force_delta_N"].mean(),
                "harm_force_delta_N": harms["force_delta_N"].mean(),
                "rescue_force_delta_N": rescues["force_delta_N"].mean(),
                "harm_correction_advantage": harms["correction_advantage"].mean(),
                "harm_direct_utility_penalty": harms["direct_utility_penalty"].mean(),
                "rescue_correction_advantage": rescues["correction_advantage"].mean(),
                "rescue_direct_utility_penalty": rescues["direct_utility_penalty"].mean(),
                "direct_underforce": g["direct_underforce"].mean(),
                "wm_underforce": g["wm_underforce"].mean(),
                "delta_underforce": (g["wm_underforce"] - g["direct_underforce"]).mean(),
                "direct_utility": g["direct_realized_utility"].mean(),
                "wm_utility": g["wm_realized_utility"].mean(),
                "delta_utility": g["delta_realized_utility"].mean(),
            }
        )

    transition = decisions.groupby(["budget", "task"], sort=True).apply(decision_summary).reset_index()
    friction = decisions.groupby(["budget", "task", "friction_band"], sort=True).apply(decision_summary).reset_index()
    roots = decisions.groupby(["budget", "task", "root_uid"], sort=True).apply(decision_summary).reset_index()
    seeds = decisions.groupby(["budget", "task", "seed"], sort=True).apply(decision_summary).reset_index()

    residual_rows = []
    for (budget, task), g in branches.groupby(["budget", "task_id"], sort=True):
        pred = g["correction"].to_numpy(dtype=float)
        target = g["residual_target"].to_numpy(dtype=float)
        residual_rows.append(
            {
                "budget": budget,
                "task": f"task{task}",
                "branches": len(g),
                "correction_target_corr": float(np.corrcoef(pred, target)[0, 1]),
                "correction_mae": float(np.mean(np.abs(pred - target))),
                "correction_mse": float(np.mean((pred - target) ** 2)),
                "predicted_correction_mean": float(np.mean(pred)),
                "target_residual_mean": float(np.mean(target)),
                "predicted_correction_std": float(np.std(pred)),
                "target_residual_std": float(np.std(target)),
            }
        )
    residual = pd.DataFrame(residual_rows)

    slope_summary = (
        slopes.groupby(["budget", "task", "friction_band"], sort=True)
        .agg(
            contexts=("correction_force_slope", "size"),
            mean_correction_force_slope=("correction_force_slope", "mean"),
            median_correction_force_slope=("correction_force_slope", "median"),
            positive_slope_fraction=("correction_force_slope", lambda s: float((s > 0).mean())),
            mean_correction_force_corr=("correction_force_corr", "mean"),
        )
        .reset_index()
    )
    residual = residual.merge(
        slopes.groupby(["budget", "task"], sort=True)
        .agg(
            mean_correction_force_slope=("correction_force_slope", "mean"),
            median_correction_force_slope=("correction_force_slope", "median"),
            positive_slope_fraction=("correction_force_slope", lambda s: float((s > 0).mean())),
            mean_correction_force_corr=("correction_force_corr", "mean"),
        )
        .reset_index(),
        on=["budget", "task"],
        how="left",
    )
    return decisions, branches, transition, friction, roots, seeds, residual, slope_summary


def training_balance(md: pd.DataFrame):
    selected = pd.read_csv(RUN_DIR / "SPARSE60_SELECTED_BRANCHES.csv")
    labels = md[["branch_id", "success"]].rename(columns={"success": "success_full"})
    selected = selected.merge(
        labels, on="branch_id", how="left", validate="many_to_one"
    )
    return (
        selected[selected["task"].isin(TASKS)]
        .groupby(["outer_fold", "task"], sort=True)
        .agg(train_rollouts=("branch_id", "size"), train_successes=("success_full", "sum"))
        .reset_index()
        .assign(
            task_label=lambda d: "task" + d["task"].astype(str),
            train_success_rate=lambda d: d["train_successes"] / d["train_rollouts"],
        )[["outer_fold", "task_label", "train_rollouts", "train_successes", "train_success_rate"]]
        .rename(columns={"task_label": "task"})
    )


def report_markdown(transition, friction, roots, seeds, residual, slope_summary, train):
    t60 = transition[transition["budget"] == 60].set_index("task")
    t120 = transition[transition["budget"] == 120].set_index("task")
    f60 = friction[friction["budget"] == 60].set_index(["task", "friction_band"])
    r60 = residual[residual["budget"] == 60].set_index("task")
    s60 = slope_summary[slope_summary["budget"] == 60].set_index(["task", "friction_band"])
    task1_roots = roots[(roots.budget == 60) & (roots.task == "task1")].sort_values("delta_sr")
    worst_root_a = task1_roots.iloc[0]
    worst_root_b = task1_roots.iloc[1]

    return f"""# Why Point-WM helps Task 6 but hurts Task 1

> Scope note: the current authoritative ActiveForcing population contains task0, task1, task5, and task6; it contains no task2. This report therefore treats the user's “task2” as the current worst-performing task, task1. No simulator rollout or retraining was run.

## Direct answer

Point-WM is not uniformly “good” or “bad.” Its learned residual correction changes the **ranking of candidate forces**, and that ranking points in opposite directions on these two tasks.

- **Task1, Sparse-60:** Direct {_pct(t60.loc['task1','direct_sr'])} → Point-WM {_pct(t60.loc['task1','wm_sr'])}, {_pp(t60.loc['task1','delta_sr'])}. It creates {int(t60.loc['task1','harms'])} success→failure flips and only {int(t60.loc['task1','rescues'])} rescue. Every lost success is in the low-friction band, where SR falls {_pp(f60.loc[('task1','low'),'delta_sr'])} and selected force falls {f60.loc[('task1','low'),'mean_force_delta_N']:+.3f} N on average.
- **Task6, Sparse-60:** Direct {_pct(t60.loc['task6','direct_sr'])} → Point-WM {_pct(t60.loc['task6','wm_sr'])}, {_pp(t60.loc['task6','delta_sr'])}. It produces {int(t60.loc['task6','rescues'])} failure→success flips and zero harm. The selected force rises {t60.loc['task6','mean_force_delta_N']:+.3f} N on average.

So the immediate cause is directional: on task1 low friction, the correction often makes lower-force candidates look too attractive; on task6, it supplies a useful higher-force safety margin.

## Evidence from matched held-out decisions

| Budget | Task | Direct SR | Point-WM SR | ΔSR | Harms | Rescues | Mean Δforce |
|---:|---|---:|---:|---:|---:|---:|---:|
| 60 | task1 | {_pct(t60.loc['task1','direct_sr'])} | {_pct(t60.loc['task1','wm_sr'])} | {_pp(t60.loc['task1','delta_sr'])} | {int(t60.loc['task1','harms'])} | {int(t60.loc['task1','rescues'])} | {t60.loc['task1','mean_force_delta_N']:+.3f} N |
| 60 | task6 | {_pct(t60.loc['task6','direct_sr'])} | {_pct(t60.loc['task6','wm_sr'])} | {_pp(t60.loc['task6','delta_sr'])} | {int(t60.loc['task6','harms'])} | {int(t60.loc['task6','rescues'])} | {t60.loc['task6','mean_force_delta_N']:+.3f} N |
| 120 | task1 | {_pct(t120.loc['task1','direct_sr'])} | {_pct(t120.loc['task1','wm_sr'])} | {_pp(t120.loc['task1','delta_sr'])} | {int(t120.loc['task1','harms'])} | {int(t120.loc['task1','rescues'])} | {t120.loc['task1','mean_force_delta_N']:+.3f} N |
| 120 | task6 | {_pct(t120.loc['task6','direct_sr'])} | {_pct(t120.loc['task6','wm_sr'])} | {_pp(t120.loc['task6','delta_sr'])} | {int(t120.loc['task6','harms'])} | {int(t120.loc['task6','rescues'])} | {t120.loc['task6','mean_force_delta_N']:+.3f} N |

The sign persists at 120 rollouts/task: task1 remains negative ({_pp(t120.loc['task1','delta_sr'])}) and task6 remains positive ({_pp(t120.loc['task6','delta_sr'])}). Sparse data amplifies task1's failure, but does not create the task dependence from nothing.

## Why task1 fails

Task1's damage is sharply conditional on friction:

- Low friction: Direct {_pct(f60.loc[('task1','low'),'direct_sr'])} → Point-WM {_pct(f60.loc[('task1','low'),'wm_sr'])}; {_pp(f60.loc[('task1','low'),'delta_sr'])}.
- Mid and high friction: both remain at 100% SR.
- In low friction, the correction-versus-force slope averages {s60.loc[('task1','low'),'mean_correction_force_slope']:+.4f}; only {_pct(s60.loc[('task1','low'),'positive_slope_fraction'])} of contexts have a positive slope. The correction therefore usually favors lower forces exactly where the success boundary is fragile.
- In task1 harm cases, moving to the WM-selected lower force costs {t60.loc['task1','harm_direct_utility_penalty']:+.4f} in Direct utility, but gains {t60.loc['task1','harm_correction_advantage']:+.4f} from the learned correction. The correction overwhelms Direct's penalty and flips the argmax downward.

This is broad, not a single bad root: five of six task1 roots have negative ΔSR in Sparse-60; the two worst roots are `{worst_root_a.root_uid}` ({_pp(worst_root_a.delta_sr)}) and `{worst_root_b.root_uid}` ({_pp(worst_root_b.delta_sr)}).

## Why task6 benefits

Task6's terminal outcome labels are almost saturated: Sparse-60 training folds contain 57–59 successes out of 60. Direct therefore sees only 1–3 failures per fold and has weak evidence about the lower safety boundary.

Point-WM's correction is positively related to force on task6: the mean correction-force slope is {r60.loc['task6','mean_correction_force_slope']:+.4f}, and {_pct(r60.loc['task6','positive_slope_fraction'])} of contexts have positive slopes. This raises force by {t60.loc['task6','mean_force_delta_N']:+.3f} N on average and by {t60.loc['task6','rescue_force_delta_N']:+.3f} N on the rescued episodes. It rescues the Direct under-force cases without turning any Direct success into a failure.

The gain is also distributed: four of six held-out roots improve, with no root getting worse. This is consistent with a safety-margin effect rather than one lucky root.

## What the residual diagnostics do—and do not—show

The branchwise correlation between the learned correction and its realized residual target is weak on both tasks: {r60.loc['task1','correction_target_corr']:+.3f} for task1 and {r60.loc['task6','correction_target_corr']:+.3f} for task6. Task1 also has larger correction MSE ({r60.loc['task1','correction_mse']:.3f} vs {r60.loc['task6','correction_mse']:.3f}).

Therefore task6's gain should **not** be described as proof that the WM globally predicts task6 physics accurately. The stronger claim supported here is narrower: its correction ranks the locally relevant force alternatives in a helpful direction on task6, and in a harmful direction on task1 low friction.

The archived shards save the final correction and decision scores, but not the predicted H8 trajectory or the fitted WM checkpoints. This analysis cannot decompose the error further into “trajectory prediction error” versus “utility/residual mapping error” without rerunning training, which was out of scope.

## Additional caveat specific to task1

Task1 has 140/180 terminal outcomes reconstructed in the authoritative population. Those terminal labels affect Direct BCE and the residual target; the PhysicsOnly H8 trajectory supervision itself remains physical trajectory supervision. This makes label reconstruction a plausible contributor to task1 residual miscalibration, but the current evidence does not establish it as the sole cause.

## Most honest interpretation

Point-WM is acting as a **task- and condition-dependent force-ranking correction**, not as a universally reliable physics oracle. It helps task6 because its upward force bias supplies missing safety margin in a nearly all-success task. It hurts task1 because, especially under low friction, its correction is optimistic about lower forces and crosses the true success boundary.

The next highest-value check—still without new simulator data—is a sensitivity audit on task1's reconstructed terminal labels and a saved-H8 prediction audit if checkpoints or predictions exist elsewhere. Until then, task6 should be presented as a robust directional benefit, while task1 is a clear failure mode of residual ranking under a sharp low-friction frontier.
"""


def build_artifact(transition, friction, roots, residual, train):
    generated = datetime.now(timezone.utc).isoformat()
    transition_view = transition.copy()
    transition_view["task_budget"] = transition_view["task"] + " / " + transition_view["budget"].astype(str)
    friction60 = friction[friction["budget"] == 60].copy()
    root60 = roots[roots["budget"] == 60].copy()
    source_id = "task1_task6_forensic"

    blocks = [
        {
            "type": "markdown",
            "title": "Why Point-WM helps Task 6 but hurts Task 1",
            "body": "# Why Point-WM helps Task 6 but hurts Task 1\n\n**Technical forensic report · read-only archived predictions · no retraining or simulator rollout**\n\n> Scope note: the current population has task0, task1, task5, and task6, but no task2. This report uses task1—the current worst-performing task—as the requested comparison with task6.",
        },
        {
            "type": "markdown",
            "title": "Technical summary",
            "body": "## Technical summary\n\nPoint-WM changes candidate-force ranking in opposite directions. Under Sparse-60, task1 falls from **89.81% to 74.07% SR (−15.74 pp)** because the learned correction favors lower forces in low-friction contexts; it creates **18 harms and one rescue**. Task6 rises from **86.11% to 95.37% (+9.26 pp)** because the correction adds force and rescues **10** Direct failures with **zero** harms. The sign persists at Full-120, so sparse data amplifies rather than invents the task dependence.",
            "sourceId": source_id,
        },
        {
            "type": "markdown",
            "title": "Key finding",
            "body": "## The failure and benefit occur in different physical regimes\n\nThe chart decomposes Point-WM minus Direct SR on the exact same outer-held-out episodes. Task1's entire loss is concentrated in low friction; task6 gains in low and high friction.",
            "sourceId": source_id,
        },
        {
            "type": "chart",
            "title": "Sparse-60 SR change by task and friction band",
            "subtitle": "Point-WM minus Direct; three seeds × held-out episodes",
            "dataset": "friction60",
            "chartType": "bar",
            "encodings": {
                "x": {"field": "friction_band", "type": "nominal", "label": "Friction band"},
                "y": {"field": "delta_sr", "type": "quantitative", "label": "Δ full-task success rate", "valueFormat": "percent"},
                "color": {"field": "task", "type": "nominal", "label": "Task"},
                "tooltip": [
                    {"field": "task", "label": "Task"},
                    {"field": "friction_band", "label": "Friction"},
                    {"field": "direct_sr", "label": "Direct SR", "valueFormat": "percent"},
                    {"field": "wm_sr", "label": "Point-WM SR", "valueFormat": "percent"},
                    {"field": "delta_sr", "label": "ΔSR", "valueFormat": "percent"},
                    {"field": "mean_force_delta_N", "label": "Mean Δforce (N)", "valueFormat": ".3f"},
                ],
            },
            "options": {"legend": {"position": "top"}, "xAxis": {"labelAngle": 0}},
            "sourceId": source_id,
        },
        {
            "type": "markdown",
            "title": "Interpretation",
            "body": "Task1 low-friction SR drops **47.22 pp** while selected force drops **0.795 N** on average. Mid and high friction stay at 100%. Task6 gains **11.11 pp** in low friction and **16.67 pp** in high friction while selecting more force. This is a directional ranking effect at the success frontier.",
            "sourceId": source_id,
        },
        {
            "type": "table",
            "title": "Matched decision transitions",
            "dataset": "transitions",
            "columns": [
                {"field": "budget", "label": "Rollouts/task", "format": "integer"},
                {"field": "task", "label": "Task"},
                {"field": "direct_sr", "label": "Direct SR", "format": "percent"},
                {"field": "wm_sr", "label": "Point-WM SR", "format": "percent"},
                {"field": "delta_sr", "label": "ΔSR", "format": "percent"},
                {"field": "harms", "label": "Success→failure", "format": "integer"},
                {"field": "rescues", "label": "Failure→success", "format": "integer"},
                {"field": "mean_force_delta_N", "label": "Mean Δforce (N)", "format": ".3f"},
            ],
            "defaultSort": [{"field": "budget", "direction": "asc"}, {"field": "task", "direction": "asc"}],
            "sourceId": source_id,
        },
        {
            "type": "markdown",
            "title": "Mechanism",
            "body": "## The learned correction overwhelms Direct in opposite directions\n\nOn task1 harm cases, the WM-selected lower-force candidate loses about **0.069** Direct utility but gains about **0.213** learned correction, flipping the argmax downward. On task6 rescues, a higher-force candidate loses about **0.123** Direct utility but gains about **0.258** correction, crossing the physical success boundary upward. Task6's Sparse-60 training folds contain only **1–3 failures out of 60**, so Direct's terminal labels weakly identify that boundary.",
            "sourceId": source_id,
        },
        {
            "type": "table",
            "title": "Sparse-60 root-level stability",
            "dataset": "root60",
            "columns": [
                {"field": "task", "label": "Task"},
                {"field": "root_uid", "label": "Root"},
                {"field": "direct_sr", "label": "Direct SR", "format": "percent"},
                {"field": "wm_sr", "label": "Point-WM SR", "format": "percent"},
                {"field": "delta_sr", "label": "ΔSR", "format": "percent"},
                {"field": "mean_force_delta_N", "label": "Mean Δforce (N)", "format": ".3f"},
            ],
            "defaultSort": [{"field": "task", "direction": "asc"}, {"field": "root_uid", "direction": "asc"}],
            "sourceId": source_id,
        },
        {
            "type": "markdown",
            "title": "Scope and metric definitions",
            "body": "## Scope, data, and metric definitions\n\nThe forensic reuses the authoritative 720-branch population and the already-complete strict nested grouped-root OOF shards. Sparse-60 uses the canonical first repeat; Full-120 uses both repeats on the same four outer-training roots/task. Evaluation is on unchanged held-out roots. **ΔSR** is Point-WM SR minus Direct SR. **Harm** is a Direct success changed to a Point-WM failure; **rescue** is the reverse. **Δforce** is Point-WM selected force minus Direct selected force on the matched episode.",
            "sourceId": source_id,
        },
        {
            "type": "markdown",
            "title": "Methodology",
            "body": "## Matched forensic methodology\n\nFor every budget, fold, seed, root, friction context, and repeat, the analysis reconstructs the Direct and Point-WM argmax from archived branch scores, then joins the selected branches to realized outcomes and utilities. It separately measures decision transitions, root concentration, friction-band effects, correction-versus-force slopes, and branchwise correction calibration. No model was fit and no validation membership was changed.",
            "sourceId": source_id,
        },
        {
            "type": "table",
            "title": "Residual correction diagnostics",
            "dataset": "residual",
            "columns": [
                {"field": "budget", "label": "Rollouts/task", "format": "integer"},
                {"field": "task", "label": "Task"},
                {"field": "correction_target_corr", "label": "Corr(correction, target)", "format": ".3f"},
                {"field": "correction_mse", "label": "Correction MSE", "format": ".3f"},
                {"field": "mean_correction_force_slope", "label": "Mean correction/force slope", "format": ".4f"},
                {"field": "positive_slope_fraction", "label": "Positive-slope contexts", "format": "percent"},
            ],
            "defaultSort": [{"field": "budget", "direction": "asc"}, {"field": "task", "direction": "asc"}],
            "sourceId": source_id,
        },
        {
            "type": "table",
            "title": "Sparse-60 terminal-label balance",
            "dataset": "training_balance",
            "columns": [
                {"field": "outer_fold", "label": "Outer fold", "format": "integer"},
                {"field": "task", "label": "Task"},
                {"field": "train_rollouts", "label": "Rollouts", "format": "integer"},
                {"field": "train_successes", "label": "Successes", "format": "integer"},
                {"field": "train_success_rate", "label": "Success rate", "format": "percent"},
            ],
            "defaultSort": [{"field": "task", "direction": "asc"}, {"field": "outer_fold", "direction": "asc"}],
            "sourceId": source_id,
        },
        {
            "type": "markdown",
            "title": "Limitations and robustness",
            "body": "## What this establishes—and what it does not\n\nThe sign is robust across budgets: task1 is negative and task6 positive at both 60 and 120 rollouts/task. Root breakdown rejects a one-root explanation. However, archived shards contain final correction scores but not predicted H8 trajectories or fitted checkpoints, so this report cannot uniquely assign error to trajectory prediction versus the downstream residual mapping. Branchwise correction-target correlation is weak on both tasks; task6's benefit is therefore evidence of locally useful **ranking**, not globally accurate physics prediction. Task1 also has 140/180 reconstructed terminal outcomes, which can affect Direct BCE and residual targets but not the saved physical H8 supervision itself.",
            "sourceId": source_id,
        },
        {
            "type": "markdown",
            "title": "Recommended next checks",
            "body": "## Recommended next checks\n\n1. Audit task1 conclusions against the reconstructed terminal-label subset, without collecting new rollouts.\n2. If archived H8 predictions/checkpoints exist elsewhere, measure trajectory error by friction band and force; this is the missing link needed to distinguish WM prediction error from residual-mapping error.\n3. In reporting, call task6 a robust directional safety-margin benefit and task1 a low-friction residual-ranking failure. Do not describe task6 as proof of universally accurate world modeling.",
            "sourceId": source_id,
        },
        {
            "type": "markdown",
            "title": "Further questions",
            "body": "## Further questions\n\n- Does task1's low-friction correction remain negative when restricted to directly observed terminal labels?\n- Are task6 rescues associated with a specific H8 physical channel, or only with the final residual score?\n- Is the correction-force slope stable across inner folds, or does it emerge from a subset of inner training roots?",
            "sourceId": source_id,
        },
    ]
    charts = []
    tables = []
    for index, block in enumerate(blocks, start=1):
        block["id"] = f"block_{index:02d}"
        if block["type"] == "markdown":
            block.pop("title", None)
        elif block["type"] == "chart":
            chart_id = f"chart_{index:02d}"
            chart = {k: v for k, v in block.items() if k not in {"id", "type", "chartType"}}
            # The portable schema's widget-level source contract is SQL-only.
            # This report is produced from local Python/NPZ forensics, so keep
            # provenance in manifest/top-level sources rather than invent SQL.
            chart.pop("sourceId", None)
            chart["source"] = {
                "label": f"Embedded snapshot dataset: {chart['dataset']}",
                "query": {"sql": f"SELECT * FROM {chart['dataset']}"},
            }
            chart.update({"id": chart_id, "type": block["chartType"]})
            charts.append(chart)
            block.clear()
            block.update({"id": f"block_{index:02d}", "type": "chart", "chartId": chart_id})
        elif block["type"] == "table":
            table_id = f"table_{index:02d}"
            table = {k: v for k, v in block.items() if k not in {"id", "type"}}
            table.pop("sourceId", None)
            table["source"] = {
                "label": f"Embedded snapshot dataset: {table['dataset']}",
                "query": {"sql": f"SELECT * FROM {table['dataset']}"},
            }
            if isinstance(table.get("defaultSort"), list):
                table["defaultSort"] = table["defaultSort"][0]
            table["id"] = table_id
            tables.append(table)
            block.clear()
            block.update({"id": f"block_{index:02d}", "type": "table", "tableId": table_id})

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Why Point-WM helps Task 6 but hurts Task 1",
            "subtitle": "Archived strict-OOF decision forensic; no retraining or simulator rollout",
            "generatedAt": generated,
            "blocks": blocks,
            "charts": charts,
            "tables": tables,
            "sources": [
                {
                    "id": source_id,
                    "label": "Derived forensic tables from authoritative archived OOF shards",
                    "path": "TASK1_TASK6_DECISION_FORENSIC.csv",
                }
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "friction60": _records(friction60),
                "transitions": _records(transition_view),
                "root60": _records(root60),
                "residual": _records(residual),
                "training_balance": _records(train),
            },
            "accessIssues": [],
        },
        "sources": [
            {
                "id": source_id,
                "query": {
                    "engine": "python",
                    "language": "python",
                    "description": "Read archived NPZ branch scores, reconstruct matched Direct and Point-WM selections, and aggregate by task, friction, root, seed, and budget.",
                    "tables_used": [
                        "shards/sparse60_fold*_seed*.npz",
                        "prior shards/fold*_seed*.npz",
                        "authoritative branch population",
                    ],
                    "filters": ["task_id in (1, 6)", "strict outer-held-out roots"],
                    "executed_at": generated,
                },
                "source_snapshot": {
                    "fetched_at": generated,
                    "row_count": int(len(friction60)),
                    "columns": list(friction60.columns),
                    "preview_rows": _records(friction60.head(6)),
                },
                "notes": "Task2 is absent from this population; task1 is used as the current worst-task comparison requested by the user.",
            }
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old, md, base = load_population()
    results = analyse(old, base)
    decisions, branches, transition, friction, roots, seeds, residual, slope_summary = results
    train = training_balance(md)

    outputs = {
        "TASK1_TASK6_DECISION_FORENSIC.csv": decisions,
        "TASK1_TASK6_BRANCH_RESIDUALS.csv": branches,
        "TASK1_TASK6_TRANSITION_SUMMARY.csv": transition,
        "TASK1_TASK6_FRICTION_BAND_SUMMARY.csv": friction,
        "TASK1_TASK6_ROOT_SUMMARY.csv": roots,
        "TASK1_TASK6_SEED_SUMMARY.csv": seeds,
        "TASK1_TASK6_RESIDUAL_DIAGNOSTICS.csv": residual,
        "TASK1_TASK6_CORRECTION_SLOPES.csv": slope_summary,
        "TASK1_TASK6_TRAIN_LABEL_BALANCE.csv": train,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT_DIR / name, index=False)

    report = report_markdown(transition, friction, roots, seeds, residual, slope_summary, train)
    (OUT_DIR / "TASK1_TASK6_WM_FORENSIC_REPORT.md").write_text(report, encoding="utf-8")

    artifact = build_artifact(transition, friction, roots, residual, train)
    (OUT_DIR / "artifact.json").write_text(json.dumps(_clean(artifact), indent=2), encoding="utf-8")

    chart_contract = {
        "material_findings": [
            "Task1 Sparse-60 harm is concentrated in low friction.",
            "Task6 Sparse-60 gains occur in low and high friction.",
            "The task-specific sign persists under Full-120.",
            "Root-level effects are distributed rather than single-root outliers.",
        ],
        "chart_map": [
            {
                "finding": "Opposite friction-conditioned ΔSR",
                "figure": "Sparse-60 SR change by task and friction band",
                "evidence": "TASK1_TASK6_FRICTION_BAND_SUMMARY.csv",
            }
        ],
        "omissions": [
            "No separate root chart: the 12-row root table is more audit-friendly than a second crowded figure.",
            "No H8 prediction-error chart: predicted trajectories/checkpoints are not present in archived shards.",
        ],
    }
    (OUT_DIR / "REPORT_SOURCE_NOTES.json").write_text(json.dumps(chart_contract, indent=2), encoding="utf-8")

    print(OUT_DIR)
    print(transition.to_string(index=False))
    print(friction[friction["budget"] == 60].to_string(index=False))


if __name__ == "__main__":
    main()
