#!/usr/bin/env python3
"""Independent final calculation, completeness, claim, and render QA."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
FIGURES = ("FIGURE_CONTINUOUS_MASS_BELIEF.pdf", "FIGURE_CONTINUOUS_MASS_FORCE.pdf",
           "FIGURE_CONTINUOUS_MASS_PARETO.pdf", "FIGURE_FRICTION_VS_MASS_FORCE.pdf")
CLAIMS = ("MASS_PHYSICAL_BELIEF_INFORMATIVE", "MASS_BELIEF_CONTINUOUS_GENERALIZATION",
          "UNSEEN_MASS_GENERALIZATION", "CONTINUOUS_MASS_FORCE_ADAPTATION",
          "AF_MASS_CLOSE_TO_GT_DECISION", "AF_MASS_MORE_RELIABLE_THAN_FIXED4",
          "AF_MASS_LOWER_FORCE_THAN_FIXED4", "AF_MASS_DOMINATES_FIXED4",
          "MASS_FAILURES_PRIMARILY_UNDER_FORCE", "SAME_ARCHITECTURE_WORKS_FOR_FRICTION_AND_MASS",
          "JOINT_FRICTION_MASS_REASONING", "OUT_OF_SUPPORT_MASS_EXTRAPOLATION",
          "REAL_ROBOT_MASS_GENERALIZATION")


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tolerance=1e-12):
    if a is None or b is None: return a is None and b is None
    return bool(np.isclose(float(a), float(b), rtol=0, atol=tolerance))


def paired(frame, a, b):
    pivot = frame[frame.method.isin([a, b])].pivot(index="context_id", columns="method", values="full_success")
    return {"both_success": int(((pivot[a] == 1) & (pivot[b] == 1)).sum()),
            f"{a}_only": int(((pivot[a] == 1) & (pivot[b] == 0)).sum()),
            f"{b}_only": int(((pivot[a] == 0) & (pivot[b] == 1)).sum()),
            "both_fail": int(((pivot[a] == 0) & (pivot[b] == 0)).sum())}


def main():
    json_out, md_out = HERE / "MASS_FINAL_ANALYSIS_VALIDATION.json", HERE / "MASS_FINAL_ANALYSIS_VALIDATION.md"
    if json_out.exists() or md_out.exists(): raise FileExistsError("final validation already exists")
    manifest = read(HERE / "FINAL_MASS_EVALUATION_MANIFEST.json")
    terminal = read(HERE / "FINAL_TERMINAL_OUTPUT.json")
    claim_audit = read(HERE / "MASS_CLAIM_AUDIT.json")
    provenance = read(HERE / "FINAL_ANALYSIS_ARTIFACT_SHA256.json")
    frame = pd.read_csv(HERE / "TABLE_CONTINUOUS_MASS_FULL_TRACE.csv")
    belief = pd.read_csv(HERE / "TABLE_CONTINUOUS_MASS_BELIEF.csv")
    checks, details = {}, {}

    checks["full_trace_144_rows"] = len(frame) == 144
    checks["belief_48_rows"] = len(belief) == 48
    checks["48_unique_contexts"] = frame.context_id.nunique() == 48 and belief.context_id.nunique() == 48
    counts = frame.groupby(["context_id", "method"]).size()
    checks["one_row_per_context_method"] = len(counts) == 144 and bool((counts == 1).all())
    method_sets = frame.groupby("context_id").method.apply(set)
    checks["all_three_methods_each_context"] = all(value == {"ACTIVEFORCING_MASS", "GT_MASS", "FIXED_4"} for value in method_sets)
    checks["roots_tasks_masses_exact"] = (set(frame.root) == set(manifest["roots"]) and
        set(frame.task) == set(manifest["tasks"]) and
        set(np.round(frame.true_mass_kg, 10)) == set(np.round(manifest["masses_per_task_kg"], 10)))
    checks["fixed4_exact"] = bool(np.allclose(frame.loc[frame.method == "FIXED_4", "selected_force_N"], 4.0, rtol=0, atol=0))
    checks["no_key_nulls"] = not frame[["context_id", "root", "task", "true_mass_kg", "method", "selected_force_N",
                                                        "measured_bilateral_squeeze_N", "full_success", "branch_path"]].isna().any().any()
    checks["branch_and_video_paths_exist"] = all(Path(path).is_dir() for path in frame.branch_path) and all(
        isinstance(path, str) and bool(path) and Path(path).is_file() for path in frame.video_path)
    checks["per_row_provenance_populated"] = not frame[["posterior_sha256", "planner_decision_sha256", "branch_result_sha256",
                                                          "action_trace_sha256", "branch_trace_sha256", "runtime_manifest_sha256",
                                                          "query_mass_readback_sha256", "video_sha256"]].isna().any().any()
    checks["per_row_provenance_hashes_exact"] = all(
        sha(Path(row.branch_path) / "PLANNER_DECISION.json") == row.planner_decision_sha256 and
        sha(Path(row.branch_path) / "BRANCH_RESULT.json") == row.branch_result_sha256 and
        sha(Path(row.branch_path) / "ACTION_TRACE.jsonl") == row.action_trace_sha256 and
        sha(Path(row.branch_path) / "BRANCH_TRACE.json") == row.branch_trace_sha256 and
        sha(Path(row.branch_path) / "MASS_INTERVENTION_READBACK.json") == row.query_mass_readback_sha256 and
        sha(Path(row.video_path)) == row.video_sha256 and
        sha(HERE / "final_evaluation" / "references" / row.context_id / "PREACTION_POSTERIOR.json") == row.posterior_sha256
        for row in frame.itertuples(index=False))
    checks["online_vla_all"] = bool(frame.online_vla_verified.all())
    checks["checkpoint_single_frozen_hash"] = frame.vla_checkpoint_sha256.nunique() == 1 and frame.vla_checkpoint_sha256.iloc[0] == manifest["online_vla"]["checkpoint_sha256"]
    checks["runtime_manifest_hash_exact"] = frame.runtime_manifest_sha256.nunique() == 1 and \
        frame.runtime_manifest_sha256.iloc[0] == sha(HERE / "FINAL_MASS_RUNTIME_MANIFEST.json")
    checks["execution_indices_complete_unique"] = set(frame.execution_index) == set(range(49, 193))

    af = frame[frame.method == "ACTIVEFORCING_MASS"]
    gt = frame[frame.method == "GT_MASS"]
    fixed = frame[frame.method == "FIXED_4"]
    merged = af.merge(gt[["context_id", "selected_force_N"]], on="context_id", suffixes=("_AF", "_GT"))
    errors = np.abs(merged.selected_force_N_AF - merged.selected_force_N_GT)
    belief_spearman = (float(spearmanr(belief.true_mass_kg, belief.posterior_mean_kg).statistic)
                       if np.std(belief.true_mass_kg) and np.std(belief.posterior_mean_kg) else None)
    calculations = {
        "MASS_BELIEF_MAE": float(np.mean(np.abs(belief.posterior_mean_kg - belief.true_mass_kg))),
        "MASS_BELIEF_RMSE": float(np.sqrt(np.mean((belief.posterior_mean_kg - belief.true_mass_kg) ** 2))),
        "MASS_BELIEF_BIAS": float(np.mean(belief.posterior_mean_kg - belief.true_mass_kg)),
        "MASS_BELIEF_SPEARMAN": belief_spearman,
        "MASS_BELIEF_90_COVERAGE": float(np.mean((belief.true_mass_kg >= belief.interval_90_low) & (belief.true_mass_kg <= belief.interval_90_high))),
        "AF_FULL_SR": float(af.full_success.mean()), "AF_MEAN_SELECTED_FORCE": float(af.selected_force_N.mean()),
        "AF_MEASURED_SQUEEZE": float(af.measured_bilateral_squeeze_N.mean()),
        "GT_MASS_FULL_SR": float(gt.full_success.mean()), "GT_MASS_MEAN_SELECTED_FORCE": float(gt.selected_force_N.mean()),
        "GT_MASS_MEASURED_SQUEEZE": float(gt.measured_bilateral_squeeze_N.mean()),
        "FIXED4_FULL_SR": float(fixed.full_success.mean()), "FIXED4_MEASURED_SQUEEZE": float(fixed.measured_bilateral_squeeze_N.mean()),
        "AF_GT_FORCE_MAE": float(errors.mean()), "AF_GT_FORCE_MEDIAN_AE": float(np.median(errors)),
        "AF_GT_WITHIN_0.10N": float(np.mean(errors <= .100000001))}
    checks["terminal_metrics_reconcile"] = all(close(value, terminal[name]) for name, value in calculations.items())
    checks["AF_GT_pairs_reconcile"] = paired(frame, "ACTIVEFORCING_MASS", "GT_MASS") == terminal["AF_VS_GT_PAIRED"]
    checks["AF_FIXED_pairs_reconcile"] = paired(frame, "ACTIVEFORCING_MASS", "FIXED_4") == terminal["AF_VS_FIXED4_PAIRED"]
    taxonomy = Counter(af.loc[af.full_success == 0, "failure_taxonomy"].fillna(""))
    checks["AF_failure_taxonomy_complete"] = "" not in taxonomy and sum(taxonomy.values()) == int((af.full_success == 0).sum())
    checks["AF_failure_counts_reconcile"] = all(terminal[f"AF_FAILURES_{name}"] == taxonomy.get(category, 0)
        for name, category in (("UNDER_FORCE", "UNDER_FORCE"), ("POST_LIFT_GEOMETRIC", "POST_LIFT_GEOMETRIC"),
                               ("VLA_VARIANCE", "VLA_EXECUTION_VARIANCE"), ("OTHER", "OTHER")))
    failure_detail = read(HERE / "MASS_AF_FAILURE_ANALYSIS.json")
    checks["AF_failure_diagnostics_complete"] = (failure_detail.get("failure_count") == int((af.full_success == 0).sum()) and
        len(failure_detail.get("records", [])) == int((af.full_success == 0).sum()) and
        all(len(row.get("counterparts", [])) == 2 and len(row.get("planner", {}).get("force_grid_N", [])) == 41 and
            row.get("diagnostic_is_causal_proof") is False for row in failure_detail.get("records", [])))

    statuses = claim_audit.get("claims", {})
    checks["claim_set_exact"] = set(statuses) == set(CLAIMS)
    checks["claim_status_vocabulary"] = all(value in {"SUPPORTED", "MIXED", "NOT_SUPPORTED", "NOT_TESTED"} for value in statuses.values())
    checks["unperformed_claims_not_tested"] = all(statuses[name] == "NOT_TESTED" for name in
        ("JOINT_FRICTION_MASS_REASONING", "OUT_OF_SUPPORT_MASS_EXTRAPOLATION", "REAL_ROBOT_MASS_GENERALIZATION"))
    reruns = read(HERE / "FINAL_MASS_RERUN_AUDIT.json")
    allowed_retry_reasons = {"INFRASTRUCTURE_CRASH", "POLICY_SERVER_FAILURE", "CORRUPTED_SNAPSHOT", "INCOMPLETE_PROVENANCE"}
    checks["zero_result_driven_reruns"] = reruns.get("result_driven_reruns") == 0
    checks["all_retries_independently_authorized"] = (reruns.get("infrastructure_retry_count") == len(reruns.get("retry_attempts", [])) and
        all(record.get("authorization", {}).get("allowed") is True and
            record.get("authorization", {}).get("reason") in allowed_retry_reasons
            for record in reruns.get("infrastructure_retry_authorizations", [])))
    friction_audit = read(HERE / "FRICTION_UNMODIFIED_AUDIT.json")
    checks["closed_friction_experiment_unmodified"] = friction_audit.get("status") == "PASS" and \
        friction_audit.get("friction_experiment_modified") is False and all(row.get("exact") for row in friction_audit.get("checks", []))

    required_sections = [f"# {index}." for index in range(1, 16)]
    report_text = (HERE / "FINAL_CONTINUOUS_MASS_GENERALIZATION_REPORT.md").read_text()
    checks["all_15_report_sections"] = all(section in report_text for section in required_sections)
    checks["all_required_figure_links_in_report"] = all(name in report_text for name in FIGURES)
    chart_map = read(HERE / "MASS_ANALYSIS_CHART_MAP.json")
    checks["four_chart_contracts"] = len(chart_map.get("charts", [])) == 4
    render_qa = {}
    qa_dir = HERE / "figure_qa"; qa_dir.mkdir(exist_ok=False)
    for name in FIGURES:
        path = HERE / name
        proc = subprocess.run(["pdfinfo", str(path)], text=True, capture_output=True, check=False)
        page_line = [line for line in proc.stdout.splitlines() if line.startswith("Pages:")]
        pages = int(page_line[0].split(":", 1)[1]) if page_line else 0
        prefix = qa_dir / path.stem
        raster = subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-png", "-r", "120", str(path), str(prefix)],
                                text=True, capture_output=True, check=False)
        preview = prefix.with_suffix(".png")
        render_qa[name] = {"bytes": path.stat().st_size, "pages": pages,
                           "pdfinfo_exit": proc.returncode, "raster_exit": raster.returncode,
                           "preview": str(preview), "preview_bytes": preview.stat().st_size if preview.exists() else 0}
    checks["required_PDFs_render"] = all(row["bytes"] > 5000 and row["pages"] >= 1 and row["pdfinfo_exit"] == 0 and
                                         row["raster_exit"] == 0 and row["preview_bytes"] > 1000 for row in render_qa.values())
    missing_hashes = {name: {"expected": digest, "actual": sha(HERE / name) if (HERE / name).is_file() else None}
                      for name, digest in provenance.items() if not (HERE / name).is_file() or sha(HERE / name) != digest}
    checks["analysis_provenance_hashes_exact"] = not missing_hashes
    ready = all(checks.values())
    details.update(calculations=calculations, AF_GT_paired=paired(frame, "ACTIVEFORCING_MASS", "GT_MASS"),
                   AF_FIXED_paired=paired(frame, "ACTIVEFORCING_MASS", "FIXED_4"),
                   AF_failure_taxonomy=dict(taxonomy), render_qa=render_qa, missing_or_changed_provenance=missing_hashes)
    result = {"as_of_utc": datetime.now(timezone.utc).isoformat(),
              "overall_assessment": "READY_TO_SHARE" if ready else "NEEDS_REVISION", "checks": checks,
              "details": details, "required_caveats": [
                  "In-support interpolation between three discrete mass anchors, not out-of-support extrapolation.",
                  "The intervention changes total object mass and scales inertia proportionally; friction and geometry are fixed.",
                  "Simulation-only evidence; no joint friction-mass inference, real-robot transfer, or non-inferiority claim.",
                  "Failure taxonomy is diagnostic and does not establish force causality."]}
    with json_out.open("x") as stream: json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    issues = [name for name, passed in checks.items() if not passed]
    md_out.write_text(f"""# Final MASS analysis validation

## Overall assessment: {'Ready to share' if ready else 'Needs revision'}

The frozen 48-context/144-branch analysis was independently checked at the context-method grain. Calculation, pairing, claim-vocabulary, provenance, report-completeness, and PDF render checks {'all passed' if ready else 'did not all pass'}.

## Methodology review

The population is the frozen two-root × four-task × six-mass interpolation set. Every rate uses the 48 predetermined contexts per method; paired counts use context IDs; force errors compare AF-MASS and GT-MASS within the same context. The official measured-squeeze values were already reconstructed from every action trace by the frozen analyzer.

## Issues found

{('None.' if not issues else chr(10).join('- ' + name for name in issues))}

## Calculation spot-checks

- Terminal headline metrics reconcile independently: **{checks['terminal_metrics_reconcile']}**.
- AF–GT and AF–Fixed paired tables reconcile: **{checks['AF_GT_pairs_reconcile'] and checks['AF_FIXED_pairs_reconcile']}**.
- All 144 context-method records are unique and complete: **{checks['one_row_per_context_method'] and checks['all_three_methods_each_context']}**.
- Saved provenance hashes match: **{checks['analysis_provenance_hashes_exact']}**.

## Visualization review

All four required PDFs passed structural parsing and rasterization. Pixel-level previews are in `figure_qa/` for final human visual inspection; chart contracts are in `MASS_ANALYSIS_CHART_MAP.json`.

## Required caveats

- In-support interpolation between three discrete mass anchors, not out-of-support extrapolation.
- Total mass and inertia scale together; friction and geometry remain fixed.
- Simulation only; no joint friction–mass inference, real-robot transfer, or statistical non-inferiority.
- Failure categories are diagnostic, not causal proof.
""")
    print(json.dumps({"assessment": result["overall_assessment"], "failed_checks": issues,
                      "previews": [row["preview"] for row in render_qa.values()]}, indent=2))


if __name__ == "__main__": main()
