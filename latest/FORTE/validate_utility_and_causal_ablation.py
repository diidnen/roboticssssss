#!/usr/bin/env python3
"""Independent arithmetic and grain checks for the offline closure outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import utility_and_causal_ablation_closure as closure


ROOT = Path(__file__).resolve().parent


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=tolerance, equal_nan=True))


def check_summary(detail: pd.DataFrame, table: pd.DataFrame, method: str) -> None:
    selected = detail[(detail["analysis"] == method) & (detail["table"] == table_name(method))]
    row = table[(table["method"] == method) & (table["scope"] == "POOLED")].iloc[0]
    assert len(selected) == 144
    assert selected.duplicated(["context_id", "repeat"]).sum() == 0
    checks = {
        "full_task_SR": selected.actual_success.mean(),
        "mean_selected_force_N": selected.selected_force_N.mean(),
        "under_force_rate": selected.under_force.mean(),
        "mean_excess_force_N": selected.excess_force_N.mean(),
        "mean_realized_utility": selected.realized_utility.mean(),
    }
    for column, expected in checks.items():
        assert close(row[column], expected), (method, column, row[column], expected)


def table_name(method: str) -> str:
    if method in {"Point", "Posterior"}:
        return "POINT_VS_POSTERIOR"
    if method in {"Query-Ignored", "Active"}:
        return "QUERY_INFORMATION"
    return "UTILITY_ABLATION"


def transitions(detail: pd.DataFrame, left: str, right: str) -> tuple[int, int, int]:
    keys = ["context_id", "repeat"]
    a = detail[(detail.analysis == left) & (detail.table == table_name(left))].set_index(keys).sort_index()
    b = detail[(detail.analysis == right) & (detail.table == table_name(right))].set_index(keys).sort_index()
    assert a.index.equals(b.index)
    changed = int((a.selected_force_N.to_numpy() != b.selected_force_N.to_numpy()).sum())
    rescue = int(((a.actual_success.to_numpy() == 0) & (b.actual_success.to_numpy() == 1)).sum())
    collateral = int(((a.actual_success.to_numpy() == 1) & (b.actual_success.to_numpy() == 0)).sum())
    return changed, rescue, collateral


def main() -> int:
    detail = pd.read_csv(ROOT / "UTILITY_CAUSAL_ABLATION_PER_EPISODE.csv")
    point = pd.read_csv(ROOT / "TABLE_POINT_VS_POSTERIOR.csv")
    query = pd.read_csv(ROOT / "TABLE_QUERY_INFORMATION_ABLATION.csv")
    utility = pd.read_csv(ROOT / "TABLE_UTILITY_ABLATION.csv")
    sensitivity = pd.read_csv(ROOT / "UTILITY_DEV_SENSITIVITY.csv")
    config = json.loads((ROOT / "UTILITY_FINAL_CONFIG.json").read_text())
    qa = json.loads((ROOT / "UTILITY_CAUSAL_ABLATION_QA.json").read_text())

    assert len(detail) == 7 * 144
    for method in ["Point", "Posterior"]:
        check_summary(detail, point, method)
    for method in ["Query-Ignored", "Active"]:
        check_summary(detail, query, method)
    for method in ["Success-Only", "Full Utility", "Fixed-Max"]:
        check_summary(detail, utility, method)

    point_transition = transitions(detail, "Point", "Posterior")
    point_row = point[(point.method == "Point") & (point.scope == "POOLED")].iloc[0]
    assert point_transition == (
        int(point_row.decision_changed_count),
        int(point_row.point_fail_to_posterior_success_count),
        int(point_row.point_success_to_posterior_fail_count),
    )
    query_transition = transitions(detail, "Query-Ignored", "Active")
    query_row = query[(query.method == "Active") & (query.scope == "POOLED")].iloc[0]
    assert query_transition == (
        int(query_row.decision_changed_count),
        int(query_row.paired_rescue_count),
        int(query_row.paired_collateral_count),
    )

    final_sensitivity = sensitivity[
        (sensitivity.scope == "POOLED") & (sensitivity.is_frozen_final_config == 1)
    ].iloc[0]
    full = utility[(utility.method == "Full Utility") & (utility.scope == "POOLED")].iloc[0]
    for column in ["full_task_SR", "mean_selected_force_N", "under_force_rate", "mean_excess_force_N"]:
        assert close(final_sensitivity[column], full[column]), column
    assert close(final_sensitivity.mean_realized_utility_authoritative, full.mean_realized_utility)

    expected_hash = hashlib.sha256(closure.canonical_json(closure.UTILITY_CONFIG_CORE).encode()).hexdigest()
    assert config["config_sha256"] == expected_hash == qa["utility_config_sha256"]
    assert qa["status"] == "UTILITY_CAUSAL_ABLATIONS_COMPLETE"
    assert qa["test_accessed"] is False and qa["gpu_jobs_started"] == 0
    assert qa["direct_gt_reproduction_max_abs_error"] <= 1e-6

    print(
        json.dumps(
            {
                "status": "PASS",
                "episode_rows_checked": len(detail),
                "point_transition": point_transition,
                "query_transition": query_transition,
                "utility_config_sha256": expected_hash,
                "direct_gt_reproduction_max_abs_error": qa[
                    "direct_gt_reproduction_max_abs_error"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
