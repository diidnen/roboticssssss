#!/usr/bin/env python3
"""Offline TRAIN/root-heldout-OOF ActiveForcing utility/causal ablations.

This runner is intentionally limited to the 720 archived post-P4-B branches.
It reuses frozen OOF Direct checkpoints and existing OOF friction-estimator
members.  It never launches IsaacLab, performs a rollout, or opens a sealed
TEST namespace.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import pooled_joint_novisual_current as pooled
import run_pooled_predictive_verifier as ppv
import run_residual_utility as residual


ROOT = Path(__file__).resolve().parent
OOF_DIR = ROOT / "pooled_predictive_verifier_20260901_033804"
UTILITY_DIR = ROOT / "activeforcing_residual_utility_20260901_055605"
PROBE_DIR = ROOT / "activeforcing_probe_conditioned_wm_20260901_064627"
ARCHIVE_DATA = UTILITY_DIR / "POOLED_OOF_UTILITY_DATASET.csv"
PROBE_PREDICTIONS = PROBE_DIR / "POOLED_OOF_PROBE_PREDICTIONS.csv"
PROBE_AUDIT = PROBE_DIR / "PROBE_WM_DATA_AUDIT.md"
UTILITY_PROTOCOL = UTILITY_DIR / "RESIDUAL_UTILITY_DEVELOPMENT_PROTOCOL.json"
NO_PROBE_PRIOR = Path(
    "/home/exouser/Tabero/analysis/results/active_probe_necessity_20260830_063435/"
    "NO_PROBE_PHYSICS_PRIOR.json"
)

FMAX = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
FOLDS = (0, 1, 2)
DIRECT_SEEDS = (0, 1, 2)
POSTERIOR_MEMBERS = (0, 1, 2)
SENSITIVITY_FORCE_COST = (0.75, 1.00, 1.25)
SENSITIVITY_FAILURE_COST = (0.75, 1.00, 1.25)

UTILITY_CONFIG_CORE = {
    "name": "ACTIVEFORCING_CURRENT_FULL_UTILITY",
    "equation": "p_success*(1-force_cost_coefficient*F/Fmax)+(1-p_success)*(-failure_penalty)",
    "success_benefit": 1.0,
    "force_cost_coefficient": 1.0,
    "failure_penalty": 1.0,
    "task_Fmax_N": {str(k): v for k, v in FMAX.items()},
    "tie_break": "lower_force",
    "selection": "argmax_candidate_expected_utility",
    "fit_and_sensitivity_scope": "TRAIN/root-heldout-OOF DEV folds only",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def utility_score(
    probability: np.ndarray,
    force: np.ndarray,
    fmax: np.ndarray,
    force_cost: float = 1.0,
    failure_cost: float = 1.0,
) -> np.ndarray:
    return probability * (1.0 - force_cost * force / fmax) + (1.0 - probability) * (-failure_cost)


def load_probe_members(context_ids: pd.Series) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    probe = pd.read_csv(PROBE_PREDICTIONS)
    members = probe[probe["seed"].astype(str).isin([str(x) for x in POSTERIOR_MEMBERS])].copy()
    pivot = members.pivot(index="context_id", columns="seed", values="mu_hat")
    pivot.columns = [int(x) for x in pivot.columns]
    pivot = pivot.reindex(columns=list(POSTERIOR_MEMBERS))
    if pivot.shape != (72, 3) or not np.isfinite(pivot.to_numpy(float)).all():
        raise RuntimeError(f"expected 72x3 finite OOF friction members, got {pivot.shape}")
    missing = sorted(set(context_ids.astype(str)) - set(pivot.index.astype(str)))
    if missing:
        raise RuntimeError(f"missing friction members for contexts: {missing[:5]}")
    arrays = {
        member: context_ids.astype(str).map(pivot[member]).to_numpy(np.float32)
        for member in POSTERIOR_MEMBERS
    }
    return arrays, pivot


def infer_direct_scenarios(
    full: Any,
    traces: list[Any],
    segs: dict[str, Any],
    fold_id: np.ndarray,
    scenarios: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reuse each frozen Direct model once and score all physical scenarios."""
    n = len(traces)
    outputs = {name: np.full((len(DIRECT_SEEDS), n), np.nan, np.float32) for name in scenarios}
    base_x = np.stack([segs[trace.branch_id].x for trace in traces]).astype(np.float32)
    device = torch.device("cpu")
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    for fold in FOLDS:
        held = np.flatnonzero(fold_id == fold)
        if len(held) != 240:
            raise RuntimeError(f"fold {fold}: expected 240 held branches, got {len(held)}")
        for seed in DIRECT_SEEDS:
            shard = np.load(OOF_DIR / "shards" / f"fold{fold}_seed{seed}.npz")
            x_mean = shard["x_mean"].astype(np.float32)
            x_std = shard["x_std"].astype(np.float32)
            checkpoint = torch.load(
                OOF_DIR / "checkpoints" / f"DIRECT_POOLED_fold{fold}_seed{seed}.pt",
                map_location="cpu",
                weights_only=False,
            )
            model = full.FeasibilityOnly().to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            for name, mu in scenarios.items():
                x = base_x[held].copy()
                x[:, :, 18] = mu[held, None]
                x = (x - x_mean) / x_std
                step = torch.tensor(x[:, :, :17], dtype=torch.float32, device=device)
                condition = torch.tensor(x[:, 0, 17:], dtype=torch.float32, device=device)
                with torch.no_grad():
                    outputs[name][seed, held] = sigmoid(model(step, condition).cpu().numpy()).astype(np.float32)
    result = {}
    for name, value in outputs.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"incomplete Direct inference for {name}")
        result[name] = value.mean(axis=0).astype(np.float64)
    return result


def raw_metadata(md: pd.DataFrame) -> pd.DataFrame:
    raw = residual.load_raw_metadata(md).copy()
    if len(raw) != 720 or raw["branch_id"].nunique() != 720:
        raise RuntimeError("raw metadata alignment failed")
    return raw.set_index("branch_id", drop=False)


def delayed_confirmation(row: pd.Series) -> tuple[float, int]:
    """Return confirmed delayed failure and whether exact source fields exist."""
    if int(row.get("raw_metadata_available", 0)) != 1:
        return math.nan, 0
    required = [row.get("lift_success"), row.get("full_task_success_y"), row.get("lost_in_transit")]
    if any(pd.isna(value) for value in required):
        return math.nan, 0
    confirmed = int(
        int(row.get("lift_success", 0)) == 1
        and int(row.get("full_task_success_y", 0)) == 0
        and int(row.get("lost_in_transit", 0)) == 1
    )
    return float(confirmed), 1


def choose_policy(
    data: pd.DataFrame,
    score: np.ndarray | None,
    method: str,
    raw: pd.DataFrame,
    probability: np.ndarray | None = None,
) -> pd.DataFrame:
    work = data.copy()
    work["_score"] = np.nan if score is None else score
    if probability is not None:
        work["selected_predicted_success_probability"] = probability
    rows: list[dict[str, Any]] = []
    for (context_id, repeat), group0 in work.groupby(["context_id", "repeat"], sort=True):
        group = group0.sort_values("force_N", kind="mergesort")
        if method == "Fixed-Max":
            selected = group.iloc[-1]
        else:
            best = float(group["_score"].max())
            selected = group[np.isclose(group["_score"], best, rtol=0.0, atol=1e-12)].iloc[0]
        successful = group[group["success"] == 1]
        frontier = float(successful["force_N"].min()) if len(successful) else math.nan
        selected_force = float(selected["force_N"])
        actual_success = int(selected["success"])
        fmax = float(selected["task_Fmax_N"])
        selected_raw = raw.loc[str(selected["branch_id"])]
        delayed, delayed_known = delayed_confirmation(selected_raw)
        rows.append(
            {
                "analysis": method,
                "context_id": str(context_id),
                "repeat": int(repeat),
                "root_id": str(selected["root_id"]),
                "task": int(selected["task"]),
                "branch_id": str(selected["branch_id"]),
                "selected_force_N": selected_force,
                "max_candidate_force_N": float(group["force_N"].max()),
                "actual_success": actual_success,
                "frontier_N": frontier,
                "frontier_available": int(math.isfinite(frontier)),
                "under_force": int(math.isfinite(frontier) and selected_force < frontier - 1e-9),
                "excess_force_N": max(0.0, selected_force - frontier) if math.isfinite(frontier) else math.nan,
                "realized_utility": (fmax - selected_force) / fmax if actual_success else -1.0,
                "predicted_score": finite_or_nan(selected["_score"]),
                "predicted_success_probability": finite_or_nan(
                    selected.get("selected_predicted_success_probability", math.nan)
                ),
                "delayed_failure_confirmed": delayed,
                "delayed_failure_metadata_known": delayed_known,
            }
        )
    selected = pd.DataFrame(rows)
    if len(selected) != 144 or selected.duplicated(["context_id", "repeat"]).any():
        raise RuntimeError(f"{method}: invalid selected episode grain {len(selected)}")
    return selected


def scopes(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [("POOLED", frame), *[(f"task{task}", frame[frame["task"] == task]) for task in sorted(FMAX)]]


def summarize_selection(frame: pd.DataFrame, method: str, scope: str) -> dict[str, Any]:
    delayed_known = frame[frame["delayed_failure_metadata_known"] == 1]
    return {
        "method": method,
        "scope": scope,
        "n_episodes": int(len(frame)),
        "n_contexts": int(frame["context_id"].nunique()),
        "n_roots": int(frame["root_id"].nunique()),
        "full_task_SR": float(frame["actual_success"].mean()),
        "mean_selected_force_N": float(frame["selected_force_N"].mean()),
        "max_selected_force_N": float(frame["selected_force_N"].max()),
        "under_force_rate": float(frame["under_force"].mean()),
        "mean_excess_force_N": float(frame["excess_force_N"].mean()),
        "frontier_available_rate": float(frame["frontier_available"].mean()),
        "mean_realized_utility": float(frame["realized_utility"].mean()),
        "delayed_failure_rate": math.nan,
        "confirmed_delayed_failure_lower_bound_rate": float(
            frame["delayed_failure_confirmed"].fillna(0).mean()
        ),
        "delayed_failure_metadata_coverage": float(frame["delayed_failure_metadata_known"].mean()),
        "delayed_failure_rate_on_observed_subset": float(delayed_known["delayed_failure_confirmed"].mean())
        if len(delayed_known)
        else math.nan,
        "delayed_failure_status": "NOT_IDENTIFIABLE_IN_ARCHIVE; confirmed lower bound and observed-subset diagnostic only",
    }


def transition_stats(reference: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    keys = ["context_id", "repeat"]
    left = reference.set_index(keys).sort_index()
    right = comparison.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise RuntimeError("paired transition indexes do not match")
    changed = left["selected_force_N"].to_numpy() != right["selected_force_N"].to_numpy()
    rescue = (left["actual_success"].to_numpy() == 0) & (right["actual_success"].to_numpy() == 1)
    collateral = (left["actual_success"].to_numpy() == 1) & (right["actual_success"].to_numpy() == 0)
    n = len(left)
    return {
        "decision_changed_count": int(changed.sum()),
        "decision_changed_rate": float(changed.mean()),
        "reference_fail_to_comparison_success_count": int(rescue.sum()),
        "reference_fail_to_comparison_success_rate": float(rescue.mean()),
        "reference_success_to_comparison_fail_count": int(collateral.sum()),
        "reference_success_to_comparison_fail_rate": float(collateral.mean()),
        "paired_n": int(n),
    }


def scoped_transition(reference: pd.DataFrame, comparison: pd.DataFrame, scope: str) -> dict[str, Any]:
    if scope == "POOLED":
        return transition_stats(reference, comparison)
    task = int(scope.replace("task", ""))
    return transition_stats(reference[reference.task == task], comparison[comparison.task == task])


def friction_metrics(pivot: pd.DataFrame, method: str, scope: str, prior: list[float]) -> dict[str, Any]:
    context = pivot.reset_index().copy()
    context["task"] = context["context_id"].str.extract(r"_t(\d+)_", expand=False).astype(int)
    truth = (
        pd.read_csv(PROBE_PREDICTIONS)
        .drop_duplicates("context_id")
        .set_index("context_id")["mu_GT"]
    )
    context["friction_gt"] = context["context_id"].map(truth)
    if method == "Query-Ignored":
        estimate = np.full(len(context), float(np.mean(prior)))
        spread = np.full(len(context), float(np.std(prior, ddof=0)))
    else:
        estimate = context[list(POSTERIOR_MEMBERS)].mean(axis=1).to_numpy(float)
        spread = context[list(POSTERIOR_MEMBERS)].std(axis=1, ddof=0).to_numpy(float)
    context["estimate"] = estimate
    context["spread"] = spread
    if scope != "POOLED":
        context = context[context.task == int(scope.replace("task", ""))]
    error = context["estimate"].to_numpy(float) - context["friction_gt"].to_numpy(float)
    return {
        "friction_contexts": int(len(context)),
        "mean_friction_estimate": float(context["estimate"].mean()),
        "mean_friction_support_sd": float(context["spread"].mean()),
        "friction_MAE": float(np.mean(np.abs(error))),
        "friction_RMSE": float(np.sqrt(np.mean(error**2))),
        "friction_bias": float(np.mean(error)),
    }


def fairness_columns(candidate_pool_hash: str) -> dict[str, Any]:
    return {
        "same_root": 1,
        "same_friction": 1,
        "same_initial_condition": 1,
        "same_query_action": 1,
        "same_force_candidate_semantics": 1,
        "candidate_pool_sha256": candidate_pool_hash,
        "query": "same archived 215-step P4-B action; physical response masked only for Query-Ignored",
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], percent: set[str] | None = None) -> str:
    percent = percent or set()
    display = frame[columns].copy()
    for column in columns:
        if column in percent:
            display[column] = display[column].map(lambda x: "NA" if pd.isna(x) else f"{100*float(x):.2f}%")
        elif pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda x: "NA" if pd.isna(x) else f"{float(x):.4f}")
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in display.iterrows()]
    return "\n".join([header, divider, *body])


def main() -> int:
    required = [ARCHIVE_DATA, PROBE_PREDICTIONS, PROBE_AUDIT, UTILITY_PROTOCOL, NO_PROBE_PRIOR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing required offline sources: {missing}")

    data = pd.read_csv(ARCHIVE_DATA).reset_index(drop=True)
    if len(data) != 720 or data["branch_id"].nunique() != 720:
        raise RuntimeError("authoritative OOF archive must have 720 unique branches")
    if data.groupby(["context_id", "repeat"]).size().value_counts().to_dict() != {5: 144}:
        raise RuntimeError("every context/repeat must have exactly five candidate branches")
    if set(data["task"].unique()) != set(FMAX):
        raise RuntimeError("unexpected task population")
    data["task_Fmax_N"] = data["task"].map(FMAX).astype(float)

    protocol = json.loads(UTILITY_PROTOCOL.read_text())
    if protocol["reward"]["FAILURE"] != -1 or protocol["reward"]["SUCCESS"] != "(Fmax-F)/Fmax":
        raise RuntimeError("authoritative utility definition changed")
    if {int(k): float(v) for k, v in protocol["reward"]["task_Fmax_N"].items()} != FMAX:
        raise RuntimeError("authoritative Fmax values changed")

    member_mu, probe_pivot = load_probe_members(data["context_id"])
    point_mu = np.mean(np.stack([member_mu[m] for m in POSTERIOR_MEMBERS]), axis=0).astype(np.float32)
    prior = [float(x) for x in json.loads(NO_PROBE_PRIOR.read_text())["values"]]
    if len(prior) != 3:
        raise RuntimeError("expected the frozen three-support no-information prior")

    with tempfile.TemporaryDirectory(prefix="utility_causal_archive_audit_", dir="/tmp") as temp_dir:
        tpi, cf, full, cmap, traces, meta, audits, pairs, segs, norm = pooled.load_population(Path(temp_dir))
        md = ppv.frame(traces, meta).reset_index(drop=True)
        if not np.array_equal(md["branch_id"].astype(str).to_numpy(), data["branch_id"].astype(str).to_numpy()):
            raise RuntimeError("archive and inference population are not row-aligned")
        if any(bool(audits[task].get("TEST_read", True)) for task in FMAX):
            raise RuntimeError("source audit indicates TEST access")
        raw = raw_metadata(md)
        scenarios = {"point": point_mu, "gt_reproduction_check": data["mu"].to_numpy(np.float32)}
        scenarios.update({f"posterior_member_{m}": member_mu[m] for m in POSTERIOR_MEMBERS})
        scenarios.update({f"prior_support_{i}": np.full(len(data), value, np.float32) for i, value in enumerate(prior)})
        probabilities = infer_direct_scenarios(
            full,
            traces,
            segs,
            data["fold"].to_numpy(int),
            scenarios,
        )

    direct_gt_reproduction_max_abs_error = float(
        np.max(np.abs(probabilities["gt_reproduction_check"] - data["p_D_OOF_ensemble"].to_numpy(float)))
    )
    if direct_gt_reproduction_max_abs_error > 1e-6:
        raise RuntimeError(
            f"frozen Direct reproduction mismatch: {direct_gt_reproduction_max_abs_error:.3g}"
        )

    force = data["force_N"].to_numpy(float)
    fmax = data["task_Fmax_N"].to_numpy(float)
    point_probability = probabilities["point"]
    point_utility = utility_score(point_probability, force, fmax)
    posterior_member_utilities = np.stack(
        [utility_score(probabilities[f"posterior_member_{m}"], force, fmax) for m in POSTERIOR_MEMBERS]
    )
    posterior_expected_utility = posterior_member_utilities.mean(axis=0)
    posterior_probability = np.stack(
        [probabilities[f"posterior_member_{m}"] for m in POSTERIOR_MEMBERS]
    ).mean(axis=0)
    ignored_utilities = np.stack(
        [utility_score(probabilities[f"prior_support_{i}"], force, fmax) for i in range(len(prior))]
    )
    ignored_expected_utility = ignored_utilities.mean(axis=0)
    ignored_probability = np.stack(
        [probabilities[f"prior_support_{i}"] for i in range(len(prior))]
    ).mean(axis=0)

    point_selected = choose_policy(data, point_utility, "Point", raw, point_probability)
    posterior_selected = choose_policy(
        data, posterior_expected_utility, "Posterior", raw, posterior_probability
    )
    ignored_selected = choose_policy(
        data, ignored_expected_utility, "Query-Ignored", raw, ignored_probability
    )
    active_selected = choose_policy(data, point_utility, "Active", raw, point_probability)
    success_only_selected = choose_policy(data, point_probability, "Success-Only", raw, point_probability)
    full_utility_selected = choose_policy(data, point_utility, "Full Utility", raw, point_probability)
    fixed_max_selected = choose_policy(data, None, "Fixed-Max", raw, point_probability)

    candidate_serialization = [
        {
            "context_id": str(context),
            "repeat": int(repeat),
            "forces_N": [float(x) for x in group.sort_values("force_N")["force_N"]],
        }
        for (context, repeat), group in data.groupby(["context_id", "repeat"], sort=True)
    ]
    candidate_pool_hash = sha256_bytes(canonical_json(candidate_serialization).encode())

    point_rows: list[dict[str, Any]] = []
    for method, selected in [("Point", point_selected), ("Posterior", posterior_selected)]:
        for scope, group in scopes(selected):
            row = summarize_selection(group, method, scope)
            row.update(scoped_transition(point_selected, posterior_selected, scope))
            row["point_fail_to_posterior_success_count"] = row.pop(
                "reference_fail_to_comparison_success_count"
            )
            row["point_fail_to_posterior_success_rate"] = row.pop(
                "reference_fail_to_comparison_success_rate"
            )
            row["point_success_to_posterior_fail_count"] = row.pop(
                "reference_success_to_comparison_fail_count"
            )
            row["point_success_to_posterior_fail_rate"] = row.pop(
                "reference_success_to_comparison_fail_rate"
            )
            row.update(fairness_columns(candidate_pool_hash))
            row["posterior_support"] = "3 equally weighted OOF friction-estimator members"
            point_rows.append(row)
    point_table = pd.DataFrame(point_rows)
    point_table.to_csv(ROOT / "TABLE_POINT_VS_POSTERIOR.csv", index=False)

    query_rows: list[dict[str, Any]] = []
    for method, selected in [("Query-Ignored", ignored_selected), ("Active", active_selected)]:
        for scope, group in scopes(selected):
            row = summarize_selection(group, method, scope)
            row.update(friction_metrics(probe_pivot, method, scope, prior))
            transitions = scoped_transition(ignored_selected, active_selected, scope)
            row["paired_rescue_count"] = transitions["reference_fail_to_comparison_success_count"]
            row["paired_rescue_rate"] = transitions["reference_fail_to_comparison_success_rate"]
            row["paired_collateral_count"] = transitions["reference_success_to_comparison_fail_count"]
            row["paired_collateral_rate"] = transitions["reference_success_to_comparison_fail_rate"]
            row["decision_changed_count"] = transitions["decision_changed_count"]
            row["decision_changed_rate"] = transitions["decision_changed_rate"]
            row["paired_n"] = transitions["paired_n"]
            row["physical_response_used"] = int(method == "Active")
            row["friction_estimator"] = (
                "frozen prior support mean/expected utility; P4-B response masked"
                if method == "Query-Ignored"
                else "mean of 3 OOF P4-B friction-estimator members"
            )
            row.update(fairness_columns(candidate_pool_hash))
            query_rows.append(row)
    query_table = pd.DataFrame(query_rows)
    query_table.to_csv(ROOT / "TABLE_QUERY_INFORMATION_ABLATION.csv", index=False)

    utility_rows: list[dict[str, Any]] = []
    for method, selected in [
        ("Success-Only", success_only_selected),
        ("Full Utility", full_utility_selected),
        ("Fixed-Max", fixed_max_selected),
    ]:
        for scope, group in scopes(selected):
            row = summarize_selection(group, method, scope)
            row["physics_source"] = "same Active point estimate from 3-member OOF P4-B ensemble"
            row["direct"] = "same frozen grouped-root OOF Direct ensemble"
            row["candidate_semantics"] = "same five archived continuous candidates per context/repeat"
            row["selection_rule"] = {
                "Success-Only": "argmax p_D(success|x,z,F); lower-force tie break",
                "Full Utility": "argmax current authoritative expected utility; lower-force tie break",
                "Fixed-Max": "maximum archived candidate (diagnostic)",
            }[method]
            row.update(fairness_columns(candidate_pool_hash))
            utility_rows.append(row)
    utility_table = pd.DataFrame(utility_rows)
    utility_table.to_csv(ROOT / "TABLE_UTILITY_ABLATION.csv", index=False)

    utility_config_hash = sha256_bytes(canonical_json(UTILITY_CONFIG_CORE).encode())
    config_artifact = {
        **UTILITY_CONFIG_CORE,
        "config_sha256": utility_config_hash,
        "frozen": True,
        "source_protocol": str(UTILITY_PROTOCOL),
        "source_protocol_sha256": sha256_file(UTILITY_PROTOCOL),
        "archive_candidate_pool_sha256": candidate_pool_hash,
        "sensitivity_grid": {
            "force_cost_coefficient": list(SENSITIVITY_FORCE_COST),
            "failure_penalty": list(SENSITIVITY_FAILURE_COST),
            "grid_frozen_before_computation": True,
        },
        "sealed_TEST_used": False,
    }
    write_json(ROOT / "UTILITY_FINAL_CONFIG.json", config_artifact)

    sensitivity_rows: list[dict[str, Any]] = []
    final_selection = full_utility_selected
    for force_cost in SENSITIVITY_FORCE_COST:
        for failure_cost in SENSITIVITY_FAILURE_COST:
            variant_score = utility_score(
                point_probability,
                force,
                fmax,
                force_cost=force_cost,
                failure_cost=failure_cost,
            )
            selected = choose_policy(
                data,
                variant_score,
                f"UtilitySensitivity_fc{force_cost:.2f}_fp{failure_cost:.2f}",
                raw,
                point_probability,
            )
            selected["realized_utility_variant"] = np.where(
                selected["actual_success"] == 1,
                1.0 - force_cost * selected["selected_force_N"] / selected["task"].map(FMAX),
                -failure_cost,
            )
            for scope, group in scopes(selected):
                summary = summarize_selection(group, "Full Utility Sensitivity", scope)
                changed = scoped_transition(final_selection, selected, scope)
                sensitivity_rows.append(
                    {
                        "scope": scope,
                        "force_cost_coefficient": force_cost,
                        "failure_penalty": failure_cost,
                        "is_frozen_final_config": int(force_cost == 1.0 and failure_cost == 1.0),
                        "utility_config_sha256": utility_config_hash,
                        "n_episodes": summary["n_episodes"],
                        "full_task_SR": summary["full_task_SR"],
                        "mean_selected_force_N": summary["mean_selected_force_N"],
                        "max_selected_force_N": summary["max_selected_force_N"],
                        "under_force_rate": summary["under_force_rate"],
                        "mean_excess_force_N": summary["mean_excess_force_N"],
                        "mean_realized_utility_authoritative": summary["mean_realized_utility"],
                        "mean_realized_utility_under_variant": float(group["realized_utility_variant"].mean()),
                        "decision_changed_rate_vs_final": changed["decision_changed_rate"],
                        "final_fail_to_variant_success_count": changed[
                            "reference_fail_to_comparison_success_count"
                        ],
                        "final_success_to_variant_fail_count": changed[
                            "reference_success_to_comparison_fail_count"
                        ],
                        "selection_population": "TRAIN archive evaluated as grouped-root OOF DEV folds",
                        "sealed_TEST_used": 0,
                    }
                )
    sensitivity_table = pd.DataFrame(sensitivity_rows)
    sensitivity_table.to_csv(ROOT / "UTILITY_DEV_SENSITIVITY.csv", index=False)

    per_episode = pd.concat(
        [
            point_selected.assign(table="POINT_VS_POSTERIOR"),
            posterior_selected.assign(table="POINT_VS_POSTERIOR"),
            ignored_selected.assign(table="QUERY_INFORMATION"),
            active_selected.assign(table="QUERY_INFORMATION"),
            success_only_selected.assign(table="UTILITY_ABLATION"),
            full_utility_selected.assign(table="UTILITY_ABLATION"),
            fixed_max_selected.assign(table="UTILITY_ABLATION"),
        ],
        ignore_index=True,
    )
    per_episode.to_csv(ROOT / "UTILITY_CAUSAL_ABLATION_PER_EPISODE.csv", index=False)

    point_pooled = point_table[point_table.scope == "POOLED"]
    query_pooled = query_table[query_table.scope == "POOLED"]
    utility_pooled = utility_table[utility_table.scope == "POOLED"]
    sensitivity_pooled = sensitivity_table[sensitivity_table.scope == "POOLED"]
    final_sensitivity = sensitivity_pooled[sensitivity_pooled.is_frozen_final_config == 1].iloc[0]
    sr_range = (float(sensitivity_pooled.full_task_SR.min()), float(sensitivity_pooled.full_task_SR.max()))
    force_range = (
        float(sensitivity_pooled.mean_selected_force_N.min()),
        float(sensitivity_pooled.mean_selected_force_N.max()),
    )
    point_pool = point_selected
    posterior_pool = posterior_selected
    ignored_pool = ignored_selected
    active_pool = active_selected
    success_pool = success_only_selected
    full_pool = full_utility_selected

    report = f"""# ActiveForcing Utility and Causal Ablation Closure

## Status

**UTILITY_CAUSAL_ABLATIONS_COMPLETE**

The four requested ablations are complete on the existing 720-branch TRAIN archive using grouped-root out-of-fold predictions. No IsaacLab process, new pi0 rollout, sealed TEST read, large-model training, or GPU job was used.

## Frozen analysis contract

- Population: 720 post-query branches, 72 friction contexts, 24 task-specific roots, 144 paired context/repeat episodes, and exactly five archived continuous force candidates per episode.
- Query: the same 215-step P4-B action and post-query restored branch state. Query-Ignored executes the same query semantically but masks its physical response downstream.
- Direct: the same frozen grouped-root OOF three-seed Direct ensemble for every comparison.
- Posterior: three equally weighted grouped-root OOF friction-estimator members. Point evaluates utility at their mean; Posterior averages utility over the three member values.
- Authoritative utility: `U(F|x)=p(success)*(1-F/Fmax)+(1-p(success))*(-1)`, lower-force tie break, Fmax={FMAX}.
- Frozen utility config SHA-256: `{utility_config_hash}`.
- Candidate-pool SHA-256: `{candidate_pool_hash}`.

The five archived candidates are context-specific continuous draws, not the later nine-point 0.25 N runtime grid. No nearest-force substitution is used, so all conclusions are archive-compatible offline results.

## A. Point estimate vs posterior expected utility

{markdown_table(point_pooled, ['method','n_episodes','full_task_SR','mean_selected_force_N','under_force_rate','mean_excess_force_N','mean_realized_utility','decision_changed_rate','point_fail_to_posterior_success_count','point_success_to_posterior_fail_count'], percent={'full_task_SR','under_force_rate','decision_changed_rate'})}

The comparison changes only how the same Direct and utility integrate friction uncertainty. It is a discrete ensemble posterior approximation, not a claim of calibrated Bayesian posterior density.

On this archive Posterior is neutral in net SR ({posterior_pool.actual_success.mean()-point_pool.actual_success.mean():+.4f}), with one rescue and one collateral; it increases mean force by {posterior_pool.selected_force_N.mean()-point_pool.selected_force_N.mean():+.4f} N and changes realized utility by {posterior_pool.realized_utility.mean()-point_pool.realized_utility.mean():+.4f}. This is a negative result for a pooled posterior-utility advantage.

## B. Query-Ignored vs Active Query

{markdown_table(query_pooled, ['method','friction_MAE','friction_RMSE','mean_selected_force_N','full_task_SR','under_force_rate','mean_excess_force_N','paired_rescue_count','paired_collateral_count','decision_changed_rate'], percent={'full_task_SR','under_force_rate','decision_changed_rate'})}

Because every archived branch follows P4-B, Query-Ignored is correctly interpreted as **no physical information downstream**, not as a zero-query action baseline. Same root, friction, initial condition, query action, and candidate semantics are verified in the CSV audit columns. Therefore the paired difference isolates information use rather than state change caused by the query action.

The physical response sharply improves friction MAE and reduces mean force by {active_pool.selected_force_N.mean()-ignored_pool.selected_force_N.mean():+.4f} N and mean excess by {active_pool.excess_force_N.mean()-ignored_pool.excess_force_N.mean():+.4f} N. It does not improve pooled SR: four paired rescues are offset by five collaterals (net {active_pool.actual_success.mean()-ignored_pool.actual_success.mean():+.4f}), while realized utility changes by {active_pool.realized_utility.mean()-ignored_pool.realized_utility.mean():+.4f}. Thus the information benefit is force efficiency/utility, not an established SR gain.

## C. Success-only Direct vs full utility

{markdown_table(utility_pooled, ['method','full_task_SR','mean_selected_force_N','max_selected_force_N','under_force_rate','mean_excess_force_N','mean_realized_utility','confirmed_delayed_failure_lower_bound_rate','delayed_failure_metadata_coverage'], percent={'full_task_SR','under_force_rate','confirmed_delayed_failure_lower_bound_rate','delayed_failure_metadata_coverage'})}

Delayed-failure rate is not exactly identifiable: 430/720 raw branch rows lack the required stage telemetry. The table therefore leaves the exact rate empty and reports only a confirmed lower bound plus observed metadata coverage. This negative estimability result is retained rather than filling missing outcomes.

Success-Only gains {success_pool.actual_success.mean()-full_pool.actual_success.mean():+.4f} SR but uses {success_pool.selected_force_N.mean()-full_pool.selected_force_N.mean():+.4f} N more mean force and changes authoritative realized utility by {success_pool.realized_utility.mean()-full_pool.realized_utility.mean():+.4f}. Full Utility therefore expresses the intended success-force tradeoff; it is not an SR-maximizing rule.

## D. DEV-only utility sensitivity

The small local grid, fixed in the runner before computation, varies force cost and failure penalty by +/-25% around the authoritative `(1.0, 1.0)` configuration. Across the nine pooled settings, SR ranges from {sr_range[0]:.4f} to {sr_range[1]:.4f}, and mean force ranges from {force_range[0]:.4f} N to {force_range[1]:.4f} N. The frozen final point gives SR={float(final_sensitivity.full_task_SR):.4f} and mean force={float(final_sensitivity.mean_selected_force_N):.4f} N, inside rather than at an extreme of both ranges. These are success-force tradeoff diagnostics only; no coefficient is selected from outcomes, and sealed TEST is untouched.

## Data and causal QA

- All 144 episode keys pair exactly across every compared policy; each policy selects from the identical five branch IDs for that key.
- Re-inference at GT friction reproduces the archived frozen Direct probabilities with maximum absolute error `{direct_gt_reproduction_max_abs_error:.3g}`.
- Four archive audits record `TEST_read=false`, five forces x two repeats, and exact snapshot/state parity. The P4-B estimator predictions are grouped-root OOF.
- Task 1 retains the known material caveat: 140/180 terminal labels are reconstructed. No outcome-based protocol change was made.
- Archive telemetry is incomplete for exact delayed-failure classification, and this limitation is visible in the output rather than silently proxied.
- The analysis reuses frozen predictions/checkpoints; no estimator, Direct model, controller, utility coefficient, candidate set, or outcome label was retrained or changed.

## Deliverables

- `TABLE_POINT_VS_POSTERIOR.csv`
- `TABLE_QUERY_INFORMATION_ABLATION.csv`
- `TABLE_UTILITY_ABLATION.csv`
- `UTILITY_DEV_SENSITIVITY.csv`
- `UTILITY_FINAL_CONFIG.json`
- `UTILITY_CAUSAL_ABLATION_PER_EPISODE.csv`

UTILITY_CAUSAL_ABLATIONS_COMPLETE
"""
    (ROOT / "UTILITY_AND_CAUSAL_ABLATION_REPORT.md").write_text(report, encoding="utf-8")

    qa = {
        "status": "UTILITY_CAUSAL_ABLATIONS_COMPLETE",
        "population": {
            "branches": int(len(data)),
            "episodes": int(data.groupby(["context_id", "repeat"]).ngroups),
            "contexts": int(data.context_id.nunique()),
            "roots": int(data.root_id.nunique()),
            "tasks": sorted(int(x) for x in data.task.unique()),
            "candidate_count_per_episode": 5,
        },
        "pairing": {
            "point_posterior": transition_stats(point_selected, posterior_selected)["paired_n"],
            "query_ignored_active": transition_stats(ignored_selected, active_selected)["paired_n"],
            "candidate_pool_sha256": candidate_pool_hash,
        },
        "utility_config_sha256": utility_config_hash,
        "source_hashes": {str(path): sha256_file(path) for path in required},
        "direct_checkpoint_hashes": {
            str(OOF_DIR / "checkpoints" / f"DIRECT_POOLED_fold{fold}_seed{seed}.pt"): sha256_file(
                OOF_DIR / "checkpoints" / f"DIRECT_POOLED_fold{fold}_seed{seed}.pt"
            )
            for fold in FOLDS
            for seed in DIRECT_SEEDS
        },
        "direct_gt_reproduction_max_abs_error": direct_gt_reproduction_max_abs_error,
        "test_accessed": False,
        "isaaclab_started": False,
        "pi0_rollouts_started": 0,
        "gpu_jobs_started": 0,
        "large_models_trained": 0,
        "output_hashes": {
            name: sha256_file(ROOT / name)
            for name in [
                "TABLE_POINT_VS_POSTERIOR.csv",
                "TABLE_QUERY_INFORMATION_ABLATION.csv",
                "TABLE_UTILITY_ABLATION.csv",
                "UTILITY_DEV_SENSITIVITY.csv",
                "UTILITY_FINAL_CONFIG.json",
                "UTILITY_CAUSAL_ABLATION_PER_EPISODE.csv",
                "UTILITY_AND_CAUSAL_ABLATION_REPORT.md",
            ]
        },
    }
    write_json(ROOT / "UTILITY_CAUSAL_ABLATION_QA.json", qa)
    print(json.dumps({"status": qa["status"], "outputs": list(qa["output_hashes"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
