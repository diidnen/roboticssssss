#!/usr/bin/env python3
"""Independent delivery checks for the force-scale study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
required = [
    "FMAX_PROVENANCE_AUDIT.md", "FMAX_PROVENANCE.csv", "GNP_FORCE_SCALE_MAPPING.md",
    "UTILITY_SCALE_COMPARISON.csv", "UTILITY_SCALE_PER_TASK.csv", "UTILITY_SCALE_SENSITIVITY.csv",
    "TASK6_UTILITY_COUNTERFACTUAL_CASES.csv", "TASKWISE_DIRECT_UTILITY_DECOMPOSITION.csv",
    "DIRECT_UTILITY_INTERACTION_REPORT.md", "UTILITY_FORCE_SCALE_REPORT.md",
    "COST_GEOMETRY_VISUALIZATION_DATA.csv", "UTILITY_FORCE_SCALE_ANALYSIS.ipynb",
    "ANALYSIS_VALIDATION.json", "SOURCE_MANIFEST.json", "report_app/dist/index.html",
]
missing = [name for name in required if not (HERE / name).is_file()]
assert not missing, missing

comparison = pd.read_csv(HERE / "UTILITY_SCALE_COMPARISON.csv").set_index("method")
per_task = pd.read_csv(HERE / "UTILITY_SCALE_PER_TASK.csv").set_index(["method", "task"])
cases = pd.read_csv(HERE / "TASK6_UTILITY_COUNTERFACTUAL_CASES.csv")
decomp = pd.read_csv(HERE / "TASKWISE_DIRECT_UTILITY_DECOMPOSITION.csv")
provenance = pd.read_csv(HERE / "FMAX_PROVENANCE.csv")
validation = json.loads((HERE / "ANALYSIS_VALIDATION.json").read_text())
notebook = json.loads((HERE / "UTILITY_FORCE_SCALE_ANALYSIS.ipynb").read_text())
snapshot = json.loads((HERE / "REPORT_SNAPSHOT.json").read_text())
app_data = json.loads((HERE / "report_app/src/data.json").read_text())
html = (HERE / "report_app/dist/index.html").read_text(encoding="utf-8")
report = (HERE / "UTILITY_FORCE_SCALE_REPORT.md").read_text(encoding="utf-8")

assert validation["status"] == "PASS"
assert validation["method_promotion"] == "NO"
assert not (HERE / "ACTIVEFORCING_GLOBAL_SCALE_UTILITY.json").exists()
assert not (HERE / "GLOBAL_SCALE_FRESH_VALIDATION_PLAN.md").exists()
assert provenance.set_index("task")["Fmax_N"].to_dict() == {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
assert len(cases) == 180 and cases.groupby(["context_id", "repeat"]).size().eq(5).all()
assert len(decomp) == 24
assert np.isclose(per_task.loc[("A_CURRENT_TASK_SPECIFIC", 6), "archived_selected_branch_SR"], 31 / 36)
assert np.isclose(per_task.loc[("B_GLOBAL_8N", 6), "archived_selected_branch_SR"], 32 / 36)
assert per_task.loc[("B_GLOBAL_8N", 6), "rescue_count_vs_A"] == 1
assert per_task.loc[("B_GLOBAL_8N", 6), "collateral_count_vs_A"] == 0
assert comparison.loc["D_ABSOLUTE_P_MINUS_LAMBDA_F", "collateral_count_vs_A"] == 2
assert comparison.loc["E_GNP_STYLE_GLOBAL_ABLATION", "archived_selected_branch_SR"] == comparison.loc["B_GLOBAL_8N", "archived_selected_branch_SR"]
assert ((cases.is_selected_global == cases.is_selected_gnp_style).all())
assert notebook["metadata"]["execution"]["status"] == "completed"
assert not any(out.get("output_type") == "error" for cell in notebook["cells"] for out in cell.get("outputs", []))
assert snapshot["queries"] == app_data["queries"]
assert "MIXED_DIRECT_AND_UTILITY_PROBLEM" in html and "METHOD_PROMOTION = NO" in html
assert "UTILITY_FORCE_SCALE_CAUSAL_DIAGNOSIS_COMPLETE" in report and "`METHOD_PROMOTION = NO`" in report

result = {
    "status": "PASS",
    "required_files": len(required),
    "candidate_rows_task6": len(cases),
    "notebook_code_cells_executed": notebook["metadata"]["execution"]["code_cells"],
    "html_bytes": (HERE / "report_app/dist/index.html").stat().st_size,
    "promotion_artifacts_correctly_absent": True,
    "final_classification": validation["classification"],
    "method_promotion": validation["method_promotion"],
}
(HERE / "DELIVERABLE_QA.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
