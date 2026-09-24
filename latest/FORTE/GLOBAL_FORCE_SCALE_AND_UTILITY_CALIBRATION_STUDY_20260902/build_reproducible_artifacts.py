#!/usr/bin/env python3
"""Build the reviewed report snapshot and an executable audit notebook."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def records(path: str, columns: list[str] | None = None, query: str | None = None) -> list[dict]:
    d = pd.read_csv(HERE / path)
    if query:
        d = d.query(query)
    if columns:
        d = d[columns]
    # Round only the presentation snapshot; authoritative CSVs retain full precision.
    for col in d.select_dtypes(include="number").columns:
        d[col] = d[col].round(6)
    return json.loads(d.to_json(orient="records"))


snapshot = {
    "title": "ActiveForcing utility force-scale causal diagnosis",
    "status": "reviewed",
    "filters": [],
    "report": {"asOf": "2026-09-02"},
    "scope": "TRAIN/DEV already-observed paired branches; not fresh TEST",
    "decision": "MIXED_DIRECT_AND_UTILITY_PROBLEM",
    "methodPromotion": "NO",
    "queries": {
        "comparison": {
            "source": "UTILITY_SCALE_COMPARISON.csv",
            "rows": records(
                "UTILITY_SCALE_COMPARISON.csv",
                [
                    "method", "archived_selected_branch_SR", "macro_task_SR", "mean_selected_force_N",
                    "under_force_rate", "decision_change_rate_vs_A", "rescue_count_vs_A",
                    "collateral_count_vs_A",
                ],
            ),
        },
        "perTask": {
            "source": "UTILITY_SCALE_PER_TASK.csv",
            "rows": records(
                "UTILITY_SCALE_PER_TASK.csv",
                [
                    "method", "task", "archived_selected_branch_SR", "mean_selected_force_N",
                    "under_force_rate", "decision_change_rate_vs_A", "rescue_count_vs_A",
                    "collateral_count_vs_A",
                ],
                "method in ['A_CURRENT_TASK_SPECIFIC','B_GLOBAL_8N','C_GLOBAL_RANGE_3_8N','D_ABSOLUTE_P_MINUS_LAMBDA_F']",
            ),
        },
        "sensitivity": {
            "source": "UTILITY_SCALE_SENSITIVITY.csv",
            "rows": records(
                "UTILITY_SCALE_SENSITIVITY.csv",
                [
                    "family", "parameter_value", "task", "archived_selected_branch_SR",
                    "macro_task_SR", "mean_selected_force_N", "rescue_count_vs_A", "collateral_count_vs_A",
                ],
                "family == 'B_GLOBAL_REFERENCE' and (task == '6' or task == 'ALL')",
            ),
        },
        "calibration": {
            "source": "TASKWISE_DIRECT_UTILITY_DECOMPOSITION.csv",
            "rows": records(
                "TASKWISE_DIRECT_UTILITY_DECOMPOSITION.csv",
                [
                    "task", "aggregation_level", "force_bucket", "n_branches", "force_mean_N",
                    "direct_predicted_success_mean", "empirical_archived_success_rate",
                    "calibration_gap_pred_minus_empirical", "brier_score",
                ],
                "aggregation_level == 'task_overall' or task == 6",
            ),
        },
        "task6Failures": {
            "source": "TASK6_UTILITY_COUNTERFACTUAL_CASES.csv",
            "rows": records(
                "TASK6_UTILITY_COUNTERFACTUAL_CASES.csv",
                [
                    "context_id", "repeat", "force_N", "p_success_active", "utility_A_current",
                    "selected_global_force_N", "selected_global_outcome", "selected_success_only_force_N",
                    "selected_success_only_outcome", "gt_friction_selected_force_N",
                    "gt_friction_selected_outcome",
                ],
                "is_selected_current == 1 and selected_current_outcome == 0",
            ),
        },
        "geometry": {
            "source": "COST_GEOMETRY_VISUALIZATION_DATA.csv",
            "rows": records("COST_GEOMETRY_VISUALIZATION_DATA.csv"),
        },
    },
    "sourceManifest": "SOURCE_MANIFEST.json",
}
(HERE / "REPORT_SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": source.splitlines(keepends=True),
    }


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
}
cells = []
cells.append(md(
    "# ActiveForcing Utility force-scale audit\n\n"
    "## TL;DR\n\n"
    "This notebook independently inspects the generated counterfactual tables. The frozen archive supports "
    "`MIXED_DIRECT_AND_UTILITY_PROBLEM` and `METHOD_PROMOTION=NO`. All evidence is already-observed TRAIN/DEV."
))
cells.append(md(
    "## Context & methods\n\n"
    "A holds the task-specific denominator. B uses an 8 N project action-support reference; C uses the global "
    "3–8 N range; D uses `p-F/16`. The candidate set, Direct probabilities, outcomes, episode tuples, and all "
    "upstream modules are unchanged."
))
cells.append(code(
    "from pathlib import Path\n"
    "import json\n"
    "import pandas as pd\n"
    "HERE = Path.cwd()\n"
    "comparison = pd.read_csv(HERE / 'UTILITY_SCALE_COMPARISON.csv')\n"
    "per_task = pd.read_csv(HERE / 'UTILITY_SCALE_PER_TASK.csv')\n"
    "sensitivity = pd.read_csv(HERE / 'UTILITY_SCALE_SENSITIVITY.csv')\n"
    "cases = pd.read_csv(HERE / 'TASK6_UTILITY_COUNTERFACTUAL_CASES.csv')\n"
    "decomp = pd.read_csv(HERE / 'TASKWISE_DIRECT_UTILITY_DECOMPOSITION.csv')\n"
    "validation = json.loads((HERE / 'ANALYSIS_VALIDATION.json').read_text())\n"
    "print(json.dumps(validation, indent=2))"
))
cells.append(md("## Data\n\nValidate table grains and required fixed-source invariants."))
cells.append(code(
    "assert validation['status'] == 'PASS'\n"
    "assert validation['input_rows'] == 720 and validation['episodes'] == 144\n"
    "assert validation['five_candidates_per_episode']\n"
    "assert validation['current_utility_recomputed_exact'] and validation['current_selection_reproduced']\n"
    "print({'comparison_rows': len(comparison), 'per_task_rows': len(per_task), "
    "'task6_candidate_rows': len(cases), 'decomposition_rows': len(decomp)})"
))
cells.append(md("## Results\n\nCompare the primary rules at task and archive level."))
cells.append(code(
    "cols = ['method','archived_selected_branch_SR','mean_selected_force_N','under_force_rate',"
    "'decision_change_rate_vs_A','rescue_count_vs_A','collateral_count_vs_A']\n"
    "print(comparison[cols].round(4).to_string(index=False))"
))
cells.append(code(
    "primary = ['A_CURRENT_TASK_SPECIFIC','B_GLOBAL_8N','C_GLOBAL_RANGE_3_8N','D_ABSOLUTE_P_MINUS_LAMBDA_F']\n"
    "print(per_task[per_task.method.isin(primary)][['method','task','archived_selected_branch_SR',"
    "'mean_selected_force_N','rescue_count_vs_A','collateral_count_vs_A']].round(4).to_string(index=False))"
))
cells.append(code(
    "task6_failed = cases[(cases.is_selected_current == 1) & (cases.selected_current_outcome == 0)]\n"
    "print(task6_failed[['context_id','repeat','force_N','p_success_active','selected_global_force_N',"
    "'selected_global_outcome','selected_success_only_force_N','selected_success_only_outcome']].round(4).to_string(index=False))"
))
cells.append(code(
    "print(decomp[(decomp.task == 6) & (decomp.aggregation_level.isin(['task_overall','candidate_rank']))]"
    "[['aggregation_level','force_bucket','force_mean_N','direct_predicted_success_mean',"
    "'empirical_archived_success_rate','calibration_gap_pred_minus_empirical','brier_score']].round(4).to_string(index=False))"
))
cells.append(md(
    "## Takeaways\n\n"
    "- Global 8 N scaling raises task6 SR from 86.11% to 88.89% and rescues 1/5 current failures.\n"
    "- Four failures remain despite global scaling; failed lowest-force branches carry Direct p=0.9394–1.0000.\n"
    "- Absolute cost is not cross-task stable because it produces two task1 collateral failures.\n"
    "- The result supports a mixed diagnosis, not method promotion. Fresh roots would be required for any later claim."
))
nb["cells"] = cells
(HERE / "UTILITY_FORCE_SCALE_ANALYSIS.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
print("wrote REPORT_SNAPSHOT.json and UTILITY_FORCE_SCALE_ANALYSIS.ipynb")
