#!/usr/bin/env python3
"""Offline counterfactual audit of ActiveForcing force-cost definitions.

This script never launches simulation or refits a model.  It reuses the frozen
candidate rows, Direct probabilities, and archived paired outcomes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE = Path(
    "/home/exouser/FORTE_e1diag/ACTIVEFORCING_E1_CRITICAL_CLOSURE_20260902_084000/"
    "E1_CANDIDATE_DECISION_CHAIN.csv"
)
CONFIG = REPO / "UTILITY_FINAL_CONFIG.json"
TASKS = [0, 1, 5, 6]
KEYS = ["task", "context_id", "repeat"]
FMAX = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
GLOBAL_FMIN = 3.0
GLOBAL_FREF = 8.0
PRIMARY_LAMBDA = 1.0 / 16.0
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260902


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_utilities(d: pd.DataFrame, fref: float = GLOBAL_FREF) -> pd.DataFrame:
    d = d.copy()
    p = d["p_success_active"]
    f = d["force_N"]
    d["candidate_rank"] = d.groupby(KEYS)["force_N"].rank(method="first").astype(int) - 1
    d["cost_A_task_specific"] = f / d["task_Fmax_N"]
    d["cost_B_global"] = f / fref
    raw_range_cost = (f - GLOBAL_FMIN) / (GLOBAL_FREF - GLOBAL_FMIN)
    d["cost_C_global_range_unclipped"] = raw_range_cost
    d["cost_C_global_range"] = raw_range_cost.clip(0.0, 1.0)
    d["cost_D_absolute"] = PRIMARY_LAMBDA * f
    d["utility_A_current"] = p * (1 - d["cost_A_task_specific"]) + (1 - p) * -1
    d["utility_B_global_8N"] = p * (1 - d["cost_B_global"]) + (1 - p) * -1
    d["utility_C_range_3_8N"] = p * (1 - d["cost_C_global_range"]) + (1 - p) * -1
    d["utility_D_absolute_lambda_1_16"] = p - PRIMARY_LAMBDA * f
    d["utility_E_gnp_style"] = p * (GLOBAL_FREF - f) + (1 - p) * -GLOBAL_FREF
    d["utility_success_only"] = p
    d["utility_fixed_max"] = f
    return d


METHODS = {
    "A_CURRENT_TASK_SPECIFIC": {
        "column": "utility_A_current",
        "formula": "p*(1-F/Fmax_task)+(1-p)*(-1)",
        "fref_N": np.nan,
        "lambda_per_N": np.nan,
    },
    "B_GLOBAL_8N": {
        "column": "utility_B_global_8N",
        "formula": "p*(1-F/8)+(1-p)*(-1)",
        "fref_N": 8.0,
        "lambda_per_N": np.nan,
    },
    "C_GLOBAL_RANGE_3_8N": {
        "column": "utility_C_range_3_8N",
        "formula": "p*(1-clip((F-3)/(8-3),0,1))+(1-p)*(-1)",
        "fref_N": 8.0,
        "lambda_per_N": np.nan,
    },
    "D_ABSOLUTE_P_MINUS_LAMBDA_F": {
        "column": "utility_D_absolute_lambda_1_16",
        "formula": "p-F/16",
        "fref_N": np.nan,
        "lambda_per_N": PRIMARY_LAMBDA,
    },
    "E_GNP_STYLE_GLOBAL_ABLATION": {
        "column": "utility_E_gnp_style",
        "formula": "p*(8-F)+(1-p)*(-8)",
        "fref_N": 8.0,
        "lambda_per_N": np.nan,
    },
    "SUCCESS_ONLY_DIAGNOSTIC": {
        "column": "utility_success_only",
        "formula": "p",
        "fref_N": np.nan,
        "lambda_per_N": 0.0,
    },
    "FIXED_MAX_DIAGNOSTIC": {
        "column": "utility_fixed_max",
        "formula": "F",
        "fref_N": np.nan,
        "lambda_per_N": np.nan,
    },
}


def select(d: pd.DataFrame, score: str) -> pd.DataFrame:
    # Stable low-force tie break after maximizing the supplied score.
    order = KEYS + [score, "force_N"]
    ascending = [True, True, True, False, True]
    return d.sort_values(order, ascending=ascending).groupby(KEYS, sort=False).head(1).copy()


def realized(y: pd.Series, cost: pd.Series) -> pd.Series:
    return y * (1 - cost) + (1 - y) * -1


def bootstrap_sr_ci(selected: pd.DataFrame) -> tuple[float, float]:
    # Cluster resampling by physical root within each task.
    root_means = selected.groupby("root_id")["success"].mean().to_numpy()
    if len(root_means) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(selected["task"].iloc[0]))
    draws = rng.choice(root_means, size=(BOOTSTRAPS, len(root_means)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def episode_oracle(d: pd.DataFrame) -> pd.DataFrame:
    def summarize(g: pd.DataFrame) -> pd.Series:
        wins = g.loc[g["success"] == 1, "force_N"]
        return pd.Series(
            {
                "oracle_has_success": int(len(wins) > 0),
                "oracle_min_success_force_N": wins.min() if len(wins) else np.nan,
                "oracle_max_force_success": int(g.loc[g["force_N"].idxmax(), "success"]),
            }
        )

    return d.groupby(KEYS, as_index=False).apply(summarize, include_groups=False).reset_index()


def selection_metrics(selected: pd.DataFrame, current: pd.DataFrame) -> dict:
    s = selected.copy()
    s["under_force"] = ((s["success"] == 0) & (s["oracle_has_success"] == 1)).astype(int)
    s["excess_force_N"] = np.where(
        s["success"] == 1,
        np.maximum(s["force_N"] - s["oracle_min_success_force_N"], 0.0),
        np.nan,
    )
    idx = KEYS
    a = current.set_index(idx)[["force_N", "success"]]
    b = s.set_index(idx)[["force_N", "success"]]
    pair = a.join(b, lsuffix="_A", rsuffix="_M")
    return {
        "episodes": len(s),
        "successes": int(s["success"].sum()),
        "archived_selected_branch_SR": s["success"].mean(),
        "mean_selected_force_N": s["force_N"].mean(),
        "under_force_rate": s["under_force"].mean(),
        "mean_excess_force_N_successes": s["excess_force_N"].mean(),
        "realized_current_A_utility": realized(s["success"], s["cost_A_task_specific"]).mean(),
        "realized_global_B_utility": realized(s["success"], s["cost_B_global"]).mean(),
        "realized_absolute_D_value": (s["success"] - PRIMARY_LAMBDA * s["force_N"]).mean(),
        "decision_change_rate_vs_A": (~np.isclose(pair["force_N_A"], pair["force_N_M"], atol=1e-12)).mean(),
        "rescue_count_vs_A": int(((pair["success_A"] == 0) & (pair["success_M"] == 1)).sum()),
        "collateral_count_vs_A": int(((pair["success_A"] == 1) & (pair["success_M"] == 0)).sum()),
        "both_success_count_vs_A": int(((pair["success_A"] == 1) & (pair["success_M"] == 1)).sum()),
        "both_failure_count_vs_A": int(((pair["success_A"] == 0) & (pair["success_M"] == 0)).sum()),
    }


def build_metrics(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    oracle = episode_oracle(d)
    selections: dict[str, pd.DataFrame] = {}
    for name, spec in METHODS.items():
        selections[name] = select(d, spec["column"]).merge(oracle, on=KEYS, validate="one_to_one")
    current = selections["A_CURRENT_TASK_SPECIFIC"]

    per_task_rows = []
    overall_rows = []
    for name, spec in METHODS.items():
        s = selections[name]
        for task in TASKS:
            st = s[s["task"] == task].copy()
            at = current[current["task"] == task].copy()
            m = selection_metrics(st, at)
            lo, hi = bootstrap_sr_ci(st)
            per_task_rows.append(
                {
                    "method": name,
                    "task": task,
                    "formula": spec["formula"],
                    "Fref_N": spec["fref_N"],
                    "lambda_per_N": spec["lambda_per_N"],
                    **m,
                    "SR_cluster_bootstrap_CI95_low": lo,
                    "SR_cluster_bootstrap_CI95_high": hi,
                    "evidence_scope": "TRAIN_DEV_ALREADY_OBSERVED",
                }
            )
        m = selection_metrics(s, current)
        task_sr = s.groupby("task")["success"].mean()
        overall_rows.append(
            {
                "method": name,
                "formula": spec["formula"],
                "Fref_N": spec["fref_N"],
                "lambda_per_N": spec["lambda_per_N"],
                **m,
                "macro_task_SR": task_sr.mean(),
                "min_task_SR": task_sr.min(),
                "evidence_scope": "TRAIN_DEV_ALREADY_OBSERVED",
            }
        )
    return pd.DataFrame(overall_rows), pd.DataFrame(per_task_rows), selections


def build_sensitivity(d: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    oracle = episode_oracle(d)
    for fref in [5.0, 6.0, 8.0]:
        score = d["p_success_active"] * (1 - d["force_N"] / fref) + (1 - d["p_success_active"]) * -1
        temp = d.assign(_score=score)
        selected = select(temp, "_score").merge(oracle, on=KEYS, validate="one_to_one")
        for task_label, st in [(str(t), selected[selected.task == t]) for t in TASKS] + [("ALL", selected)]:
            at = current if task_label == "ALL" else current[current.task == int(task_label)]
            m = selection_metrics(st, at)
            rows.append(
                {
                    "family": "B_GLOBAL_REFERENCE",
                    "parameter_name": "Fref_global_N",
                    "parameter_value": fref,
                    "task": task_label,
                    **m,
                    "macro_task_SR": selected.groupby("task")["success"].mean().mean(),
                    "parameter_origin": "project historical force supports; not selected from task6 outcomes",
                }
            )
    for lam in [1 / 10, 1 / 12, 1 / 16]:
        temp = d.assign(_score=d["p_success_active"] - lam * d["force_N"])
        selected = select(temp, "_score").merge(oracle, on=KEYS, validate="one_to_one")
        for task_label, st in [(str(t), selected[selected.task == t]) for t in TASKS] + [("ALL", selected)]:
            at = current if task_label == "ALL" else current[current.task == int(task_label)]
            m = selection_metrics(st, at)
            rows.append(
                {
                    "family": "D_ABSOLUTE_COST",
                    "parameter_name": "lambda_per_N",
                    "parameter_value": lam,
                    "task": task_label,
                    **m,
                    "macro_task_SR": selected.groupby("task")["success"].mean().mean(),
                    "parameter_origin": "lambda=1/(2*Fref), Fref in {5,6,8}; predeclared scale matching",
                }
            )
    return pd.DataFrame(rows)


def build_task6_cases(d: pd.DataFrame, selections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    t6 = d[d.task == 6].copy()
    names = {
        "A_CURRENT_TASK_SPECIFIC": "current",
        "B_GLOBAL_8N": "global",
        "C_GLOBAL_RANGE_3_8N": "range",
        "D_ABSOLUTE_P_MINUS_LAMBDA_F": "absolute",
        "E_GNP_STYLE_GLOBAL_ABLATION": "gnp_style",
        "SUCCESS_ONLY_DIAGNOSTIC": "success_only",
        "FIXED_MAX_DIAGNOSTIC": "fixed_max",
    }
    for method, short in names.items():
        chosen = selections[method][KEYS + ["branch_id", "force_N", "success"]].copy()
        chosen[f"selected_{short}"] = 1
        chosen = chosen.rename(
            columns={
                "branch_id": f"selected_{short}_branch_id",
                "force_N": f"selected_{short}_force_N",
                "success": f"selected_{short}_outcome",
            }
        )
        t6 = t6.merge(chosen, on=KEYS, how="left", validate="many_to_one")
        t6[f"is_selected_{short}"] = (
            t6["branch_id"] == t6[f"selected_{short}_branch_id"]
        ).astype(int)
        t6 = t6.drop(columns=[f"selected_{short}"])

    gt = d[d.selected_gt_friction == 1][KEYS + ["force_N", "success"]].rename(
        columns={"force_N": "gt_friction_selected_force_N", "success": "gt_friction_selected_outcome"}
    )
    qi = d[d.selected_query_ignored == 1][KEYS + ["force_N", "success"]].rename(
        columns={"force_N": "query_ignored_selected_force_N", "success": "query_ignored_selected_outcome"}
    )
    t6 = t6.merge(gt, on=KEYS, validate="many_to_one").merge(qi, on=KEYS, validate="many_to_one")
    t6["current_A_episode_failure"] = 1 - t6["selected_current_outcome"]
    t6["global_B_rescues_A_failure"] = (
        (t6["selected_current_outcome"] == 0) & (t6["selected_global_outcome"] == 1)
    ).astype(int)
    ordered = [
        "branch_id", "context_id", "root_id", "task", "repeat", "candidate_rank", "force_N", "mu",
        "success", "p_success_active", "p_success_gt_friction", "p_success_query_ignored",
        "cost_A_task_specific", "cost_B_global", "cost_C_global_range", "cost_D_absolute",
        "utility_A_current", "utility_B_global_8N", "utility_C_range_3_8N",
        "utility_D_absolute_lambda_1_16", "utility_E_gnp_style",
    ]
    ordered += [c for c in t6.columns if c not in ordered]
    return t6[ordered]


def calibration_decomposition(d: pd.DataFrame, selections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    x = d.copy()
    for name, short in [
        ("A_CURRENT_TASK_SPECIFIC", "A"),
        ("B_GLOBAL_8N", "B"),
        ("D_ABSOLUTE_P_MINUS_LAMBDA_F", "D"),
    ]:
        chosen = selections[name][KEYS + ["branch_id"]].rename(columns={"branch_id": f"selected_branch_{short}"})
        x = x.merge(chosen, on=KEYS, validate="many_to_one")
        x[f"selected_{short}"] = (x.branch_id == x[f"selected_branch_{short}"]).astype(int)

    rows = []
    for task in TASKS:
        subsets = [("task_overall", "ALL", x[x.task == task])]
        subsets += [
            ("candidate_rank", str(rank), x[(x.task == task) & (x.candidate_rank == rank)])
            for rank in range(5)
        ]
        for level, bucket, g in subsets:
            p = g.p_success_active.to_numpy()
            y = g.success.to_numpy()
            rows.append(
                {
                    "task": task,
                    "aggregation_level": level,
                    "force_bucket": bucket,
                    "n_branches": len(g),
                    "force_min_N": g.force_N.min(),
                    "force_mean_N": g.force_N.mean(),
                    "force_max_N": g.force_N.max(),
                    "direct_predicted_success_mean": p.mean(),
                    "empirical_archived_success_rate": y.mean(),
                    "calibration_gap_pred_minus_empirical": p.mean() - y.mean(),
                    "mean_absolute_probability_error": np.abs(p - y).mean(),
                    "brier_score": np.mean((p - y) ** 2),
                    "selected_rate_A": g.selected_A.mean(),
                    "selected_rate_B": g.selected_B.mean(),
                    "selected_rate_D": g.selected_D.mean(),
                    "evidence_scope": "paired archived TRAIN_DEV branches",
                }
            )
    return pd.DataFrame(rows)


def representative_geometry(d: pd.DataFrame, selections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Typical case: nearest to task medians of mean p and current selected force; outcome is not used.
    summaries = d.groupby(KEYS).agg(mean_p=("p_success_active", "mean")).reset_index()
    a = selections["A_CURRENT_TASK_SPECIFIC"][KEYS + ["force_N", "success"]].rename(
        columns={"force_N": "A_force", "success": "A_success"}
    )
    summaries = summaries.merge(a, on=KEYS, validate="one_to_one")
    picked = []
    for task in TASKS:
        g = summaries[summaries.task == task].copy()
        pm, fm = g.mean_p.median(), g.A_force.median()
        pscale = max(g.mean_p.std(), 1e-12)
        fscale = max(g.A_force.std(), 1e-12)
        g["distance"] = ((g.mean_p - pm) / pscale) ** 2 + ((g.A_force - fm) / fscale) ** 2
        row = g.sort_values(["distance", "context_id", "repeat"]).iloc[0]
        picked.append((task, row.context_id, int(row["repeat"]), "typical_no_outcome_selection"))

    # Diagnostic task6 failure: deterministic median among all five A failures, never best-result selected.
    failures = summaries[(summaries.task == 6) & (summaries.A_success == 0)].sort_values(["context_id", "repeat"])
    if len(failures):
        row = failures.iloc[len(failures) // 2]
        picked.append((6, row.context_id, int(row["repeat"]), "median_of_all_current_task6_failures"))

    parts = []
    for task, context, repeat, rule in picked:
        g = d[(d.task == task) & (d.context_id == context) & (d["repeat"] == repeat)].copy()
        g["representative_rule"] = rule
        parts.append(g)
    cols = [
        "representative_rule", "task", "context_id", "root_id", "repeat", "candidate_rank", "force_N",
        "p_success_active", "success", "cost_A_task_specific", "utility_A_current", "cost_B_global",
        "utility_B_global_8N", "cost_C_global_range", "utility_C_range_3_8N", "cost_D_absolute",
        "utility_D_absolute_lambda_1_16",
    ]
    return pd.concat(parts, ignore_index=True)[cols]


def provenance_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        g = d[d.task == task]
        rows.append(
            {
                "task": task,
                "Fmax_N": FMAX[task],
                "provenance_class": "A+C+E",
                "candidate_grid_max": True,
                "controller_physical_safety_limit": False,
                "dataset_collection_upper_bound": True,
                "reward_normalization_constant": False,
                "manually_tuned_task_parameter": True,
                "observed_candidate_min_N": g.force_N.min(),
                "observed_candidate_max_N": g.force_N.max(),
                "authoritative_support": {0: "3-5", 1: "4-6", 5: "3-5", 6: "3-4"}[task],
                "first_recovered_git_commit": "80ab3be09ce884f86cfc2037d3af30bc28061426",
                "source_symbol": "ROBUST_FORCE_BY_TASK / TASK_SUPPORT / FORCE_SUPPORT",
                "interpretation": "historical robust-force endpoint and task-specific collection/candidate support maximum; later reused as Utility denominator",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    d0 = pd.read_csv(SOURCE)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert d0.shape == (720, 20)
    assert set(d0.task.unique()) == set(TASKS)
    assert all(d0.groupby(KEYS).size() == 5)
    assert all(d0.groupby(KEYS)["selected_active"].sum() == 1)
    assert {int(k): float(v) for k, v in cfg["task_Fmax_N"].items()} == FMAX
    d = add_utilities(d0)
    assert np.allclose(d.utility_A_current, d.expected_utility_active, atol=1e-12)
    assert int((d.cost_C_global_range != d.cost_C_global_range_unclipped).sum()) == 0

    comparison, per_task, selections = build_metrics(d)
    sensitivity = build_sensitivity(d, selections["A_CURRENT_TASK_SPECIFIC"])
    cases = build_task6_cases(d, selections)
    decomposition = calibration_decomposition(d, selections)
    geometry = representative_geometry(d, selections)
    provenance = provenance_table(d)

    comparison.to_csv(HERE / "UTILITY_SCALE_COMPARISON.csv", index=False)
    per_task.to_csv(HERE / "UTILITY_SCALE_PER_TASK.csv", index=False)
    sensitivity.to_csv(HERE / "UTILITY_SCALE_SENSITIVITY.csv", index=False)
    cases.to_csv(HERE / "TASK6_UTILITY_COUNTERFACTUAL_CASES.csv", index=False)
    decomposition.to_csv(HERE / "TASKWISE_DIRECT_UTILITY_DECOMPOSITION.csv", index=False)
    geometry.to_csv(HERE / "COST_GEOMETRY_VISUALIZATION_DATA.csv", index=False)
    provenance.to_csv(HERE / "FMAX_PROVENANCE.csv", index=False)

    a6 = selections["A_CURRENT_TASK_SPECIFIC"].query("task == 6")
    b6 = selections["B_GLOBAL_8N"].query("task == 6")
    pair6 = a6.set_index(KEYS).join(
        b6.set_index(KEYS)[["force_N", "success"]], lsuffix="_A", rsuffix="_B"
    )
    scale_rescues = int(((pair6.success_A == 0) & (pair6.success_B == 1)).sum())
    a_failures = int((pair6.success_A == 0).sum())
    validation = {
        "status": "PASS",
        "input_rows": len(d),
        "episodes": int(d.groupby(KEYS).ngroups),
        "roots": int(d.root_id.nunique()),
        "tasks": TASKS,
        "five_candidates_per_episode": bool((d.groupby(KEYS).size() == 5).all()),
        "source_sha256": sha256(SOURCE),
        "utility_config_sha256": sha256(CONFIG),
        "current_utility_recomputed_exact": bool(
            np.allclose(d.utility_A_current, d.expected_utility_active, atol=1e-12)
        ),
        "current_selection_reproduced": bool(
            set(selections["A_CURRENT_TASK_SPECIFIC"].branch_id) == set(d.loc[d.selected_active == 1, "branch_id"])
        ),
        "global_range_clip_count": int((d.cost_C_global_range != d.cost_C_global_range_unclipped).sum()),
        "empirical_monotonicity_violations": int(
            sum(
                np.any(np.diff(g.sort_values("force_N").success.to_numpy()) < 0)
                for _, g in d.groupby(KEYS)
            )
        ),
        "direct_probability_monotonicity_violations": int(
            sum(
                np.any(np.diff(g.sort_values("force_N").p_success_active.to_numpy()) < -1e-12)
                for _, g in d.groupby(KEYS)
            )
        ),
        "task6_current_failures": a_failures,
        "task6_global_8N_rescues": scale_rescues,
        "task6_fraction_of_current_failures_rescued": scale_rescues / a_failures,
        "classification": "MIXED_DIRECT_AND_UTILITY_PROBLEM",
        "method_promotion": "NO",
        "scientific_scope": "TRAIN_DEV_ALREADY_OBSERVED_METHOD_DIAGNOSIS_NOT_FRESH_TEST",
    }
    write_json(HERE / "ANALYSIS_VALIDATION.json", validation)
    write_json(
        HERE / "SOURCE_MANIFEST.json",
        {
            "candidate_chain": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
            "utility_config": {"path": str(CONFIG), "sha256": sha256(CONFIG)},
            "global_reference_sources": [
                "/home/exouser/Tabero/analysis/p5s0a_true_matched_dataset.py",
                "/home/exouser/Tabero/analysis/p6g1r1_controller_grasp_vla_handoff.py",
                "/home/exouser/Tabero/analysis/p7b_gnp_physical_belief_force_planning.py",
            ],
            "frozen_dimensions": [
                "pi0", "P4-B query", "friction estimator", "Direct checkpoints",
                "Direct probabilities", "candidate forces", "episode tuples", "root split",
                "labels", "controller semantics",
            ],
        },
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
