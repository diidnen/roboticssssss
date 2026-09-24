#!/usr/bin/env python3
"""Audit the authoritative 720 TRAIN/DEV force-branch population.

This is deliberately read-only with respect to source archives.  It aligns the
frozen Direct/Utility table, prospective collection manifest, canonical labels,
and per-episode physical telemetry; then exports task support and separate local
lift/full-task empirical boundary status.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "BOUNDARY_AWARE_FULL_TASK_FEASIBILITY_DATA_CLOSURE_20260902"
COLLECTION = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
RUN_MANIFEST = COLLECTION / "PROSPECTIVE_TRAIN_RUN_MANIFEST.csv"
UTILITY = ROOT / "activeforcing_residual_utility_20260901_055605/POOLED_OOF_UTILITY_DATASET.csv"
AUDIT_ROOT = ROOT / "activeforcing_final_experiment_20260901_045000/_alltrain_verifier_audit"
TASKS = (0, 1, 5, 6)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def direct_branches(task: int) -> pd.DataFrame:
    path = COLLECTION / f"collection_train/task{task}/task{task}/branches.csv"
    d = pd.read_csv(path)
    return d


def canonical(task: int) -> pd.DataFrame:
    path = AUDIT_ROOT / f"task{task}/TASK{task}_CANONICAL_TRAIN_BRANCHES.csv"
    if path.exists():
        return pd.read_csv(path)
    d = direct_branches(task).copy()
    return pd.DataFrame({
        "context_id": d.context_id,
        "task": task,
        "root_id": d.root_id,
        "root_index": d.root_index,
        "friction_band": d.friction_band,
        "mu_GT": d.hidden_friction_analysis_only,
        "repeat": d.repeat_index,
        "force_N": d.requested_force_N,
        "success": d.full_task_success_y,
        "label_source": d.label_source.map(lambda _: "DIRECT_CUMULATIVE_BRANCH_LABEL"),
        "telemetry_path": d.telemetry_path,
    })


def telemetry_summary(path_value: str, force: float) -> dict:
    path = Path(str(path_value))
    result = {
        "telemetry_exists": 0,
        "local_lift_success": np.nan,
        "lift_height_m": np.nan,
        "lift_first_timestamp_s": np.nan,
        "measured_force_mean_N": np.nan,
        "measured_force_peak_N": np.nan,
        "force_tracking_mae_N": np.nan,
        "controller_tracking_error_N": np.nan,
        "telemetry_rows": 0,
        "telemetry_phase_count": 0,
    }
    if not path.exists():
        return result
    tr = pd.read_csv(path)
    result["telemetry_exists"] = 1
    result["telemetry_rows"] = len(tr)
    result["telemetry_phase_count"] = int(tr.phase.nunique()) if "phase" in tr else 0
    zcol = "object_z_analysis_only"
    if zcol in tr and len(tr):
        z = pd.to_numeric(tr[zcol], errors="coerce")
        if z.notna().any():
            z0 = float(z.dropna().iloc[0])
            dz = z - z0
            result["lift_height_m"] = float(dz.max())
            result["local_lift_success"] = int((dz >= 0.03).any())
            hit = tr.loc[dz >= 0.03]
            if len(hit) and "t_s" in hit:
                result["lift_first_timestamp_s"] = float(pd.to_numeric(hit.t_s, errors="coerce").dropna().iloc[0])
    if "measured_force_N" in tr:
        samples = pd.to_numeric(tr.measured_force_N, errors="coerce")
        if "mode" in tr:
            samples = samples[tr["mode"].astype(str) != "open"]
        samples = samples.dropna().to_numpy(float)
        if len(samples):
            steady = samples[len(samples) // 2 :]
            result["measured_force_mean_N"] = float(np.mean(steady))
            result["measured_force_peak_N"] = float(np.max(samples))
            result["force_tracking_mae_N"] = float(np.mean(np.abs(steady - force)))
            result["controller_tracking_error_N"] = float(np.mean(steady) - force)
    return result


def boundary_status(values: pd.DataFrame, label: str, prefix: str) -> dict:
    valid = values.dropna(subset=[label]).copy()
    valid[label] = valid[label].astype(int)
    succ = valid.loc[valid[label] == 1, "force_N"]
    fail = valid.loc[valid[label] == 0, "force_N"]
    low_succ = float(succ.min()) if len(succ) else np.nan
    high_fail = float(fail.max()) if len(fail) else np.nan
    if not len(valid):
        status = f"MISSING_{prefix}_LABELS"
    elif not len(fail):
        status = f"LEFT_CENSORED_{prefix}_BOUNDARY"
    elif not len(succ):
        status = f"RIGHT_CENSORED_{prefix}_BOUNDARY"
    elif high_fail < low_succ:
        status = f"BRACKETED_{prefix}_BOUNDARY"
    else:
        status = f"STOCHASTIC_OR_NONMONOTONIC_{prefix}_BOUNDARY"
    by_force = valid.groupby("force_N")[label].nunique()
    disagreement = int((by_force > 1).sum())
    interval = ""
    if math.isfinite(high_fail) and math.isfinite(low_succ) and high_fail < low_succ:
        interval = f"({high_fail:.9g},{low_succ:.9g}]"
    return {
        f"{prefix.lower()}_status": status,
        f"highest_tested_failing_force_{prefix.lower()}_N": high_fail,
        f"lowest_tested_successful_force_{prefix.lower()}_N": low_succ,
        f"{prefix.lower()}_transition_interval_N": interval,
        f"{prefix.lower()}_positive_n": int((valid[label] == 1).sum()),
        f"{prefix.lower()}_negative_n": int((valid[label] == 0).sum()),
        f"{prefix.lower()}_repeat_disagreement_force_cells": disagreement,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run = pd.read_csv(RUN_MANIFEST)
    utility = pd.read_csv(UTILITY)
    assert len(run) == len(utility) == 720
    # Six rows differ only by one binary floating-point ULP after CSV
    # serialization (maximum 4.44e-16 N).  Candidate identity is therefore
    # joined on a 12-decimal audit key while retaining both original values.
    for frame in (run, utility):
        frame["force_key_12dp"] = frame.force_N.round(12)
    keys = ["task", "root_id", "context_id", "force_key_12dp", "repeat"]
    check = run[keys].merge(utility[keys + ["success"]], on=keys, how="outer", indicator=True)
    assert (check._merge == "both").all(), check._merge.value_counts().to_dict()

    rows = []
    direct_counts = {}
    canonical_counts = {}
    for task in TASKS:
        d = direct_branches(task)
        c = canonical(task)
        direct_counts[task] = len(d)
        canonical_counts[task] = len(c)
        base = run[run.task == task].copy()
        base = base.merge(
            utility[["task", "root_id", "context_id", "force_key_12dp", "repeat", "branch_id", "success", "p_D_OOF_ensemble"]],
            on=keys, how="left", validate="one_to_one",
        )
        base["canonical_join_id"] = base.context_id + "_" + base.branch_label
        if task == 0:
            csmall = c[["context_id", "force_N", "repeat", "success", "label_source", "telemetry_path"]].copy()
            csmall["branch_id"] = d.branch_id.to_numpy()
            csmall = csmall[["branch_id", "success", "label_source", "telemetry_path"]]
            base = base.merge(csmall.rename(columns={"success": "canonical_full_task_success"}), on="branch_id", how="left", validate="one_to_one")
        else:
            csmall = c[["branch_id", "success", "label_source", "telemetry_path"]].copy().rename(columns={"branch_id": "canonical_join_id", "success": "canonical_full_task_success"})
            base = base.merge(csmall, on="canonical_join_id", how="left", validate="one_to_one")
        assert base.telemetry_path.notna().all()
        assert base.telemetry_path.map(lambda x: Path(x).exists()).all()
        assert (base.success == base.canonical_full_task_success).all()
        summaries = [telemetry_summary(p, float(f)) for p, f in zip(base.telemetry_path, base.force_N)]
        base = pd.concat([base.reset_index(drop=True), pd.DataFrame(summaries)], axis=1)
        # Direct branch fields are exact only when the cumulative branch row exists.
        direct_key = {}
        for r in d.to_dict("records"):
            direct_key[str(r["branch_id"])] = r
        direct_lift, direct_stage, direct_reason = [], [], []
        for r in base.to_dict("records"):
            x = direct_key.get(str(r["branch_id"]))
            direct_lift.append(np.nan if x is None else int(x["lift_success"]))
            direct_stage.append("" if x is None or pd.isna(x.get("failure_stage")) else str(x.get("failure_stage", "")))
            direct_reason.append("" if x is None or pd.isna(x.get("failure_reason")) else str(x.get("failure_reason", "")))
        base["direct_lift_success"] = direct_lift
        base["direct_failure_stage"] = direct_stage
        base["direct_failure_reason"] = direct_reason
        compared = base.direct_lift_success.notna()
        assert (base.loc[compared, "direct_lift_success"].astype(int) == base.loc[compared, "local_lift_success"].astype(int)).all()
        base["local_lift_label_source"] = np.where(compared, "DIRECT_CUMULATIVE_BRANCH_LABEL", "RECONSTRUCTED_FROM_TRAJECTORY_DZ_GE_0P03M")
        base["full_task_success"] = base.success.astype(int)
        base["failure_stage_audited"] = np.where(
            base.full_task_success == 1, "SUCCESS",
            np.where(base.local_lift_success == 0, "LOCAL_LIFT",
                     np.where(base.direct_failure_stage.astype(str).str.len() > 0, base.direct_failure_stage, "DOWNSTREAM_UNRESOLVED")),
        )
        rows.append(base)
    allrows = pd.concat(rows, ignore_index=True)
    assert len(allrows) == 720
    assert allrows.telemetry_exists.eq(1).all()
    allrows.to_csv(OUT / "AUTHORITATIVE_720_BOUNDARY_LABELS.csv", index=False)

    support_rows = []
    for task, q in allrows.groupby("task", sort=True):
        support_rows.append({
            "row_scope": "TASK_SUMMARY", "task": int(task), "root_id": "ALL", "friction_band": "ALL",
            "branches": len(q), "root_families": q.root_id.nunique(), "friction_contexts": q.context_id.nunique(),
            "friction_levels_per_root": int(q.groupby("root_id").context_id.nunique().min()),
            "force_cells_per_context": int(q.groupby("context_id").force_N.nunique().min()),
            "repeats_per_force_cell": int(q.groupby(["context_id", "force_N"]).size().min()),
            "requested_force_min_N": q.force_N.min(), "requested_force_max_N": q.force_N.max(),
            "requested_force_unique": q.force_N.nunique(), "measured_force_steady_mean_N": q.measured_force_mean_N.mean(),
            "measured_force_steady_min_N": q.measured_force_mean_N.min(), "measured_force_steady_max_N": q.measured_force_mean_N.max(),
            "tracking_mae_mean_N": q.force_tracking_mae_N.mean(), "tracking_abs_bias_mean_N": q.controller_tracking_error_N.abs().mean(),
            "local_lift_positive_n": int(q.local_lift_success.sum()), "local_lift_negative_n": int((1-q.local_lift_success).sum()),
            "full_task_positive_n": int(q.full_task_success.sum()), "full_task_negative_n": int((1-q.full_task_success).sum()),
            "delayed_failure_n": int(((q.local_lift_success == 1) & (q.full_task_success == 0)).sum()),
            "full_label_direct_n": int((q.label_source == "DIRECT_CUMULATIVE_BRANCH_LABEL").sum()),
            "full_label_reconstructed_n": int((q.label_source != "DIRECT_CUMULATIVE_BRANCH_LABEL").sum()),
            "local_label_direct_n": int((q.local_lift_label_source == "DIRECT_CUMULATIVE_BRANCH_LABEL").sum()),
            "local_label_reconstructed_n": int((q.local_lift_label_source != "DIRECT_CUMULATIVE_BRANCH_LABEL").sum()),
            "telemetry_present_n": int(q.telemetry_exists.sum()),
        })
        for (root, band), g in q.groupby(["root_id", "friction_band"], sort=True):
            support_rows.append({
                "row_scope": "TASK_ROOT_FRICTION", "task": int(task), "root_id": root, "friction_band": band,
                "branches": len(g), "root_families": 1, "friction_contexts": 1,
                "friction_levels_per_root": "", "force_cells_per_context": g.force_N.nunique(),
                "repeats_per_force_cell": int(g.groupby("force_N").size().min()),
                "requested_force_min_N": g.force_N.min(), "requested_force_max_N": g.force_N.max(),
                "requested_force_unique": g.force_N.nunique(), "measured_force_steady_mean_N": g.measured_force_mean_N.mean(),
                "measured_force_steady_min_N": g.measured_force_mean_N.min(), "measured_force_steady_max_N": g.measured_force_mean_N.max(),
                "tracking_mae_mean_N": g.force_tracking_mae_N.mean(), "tracking_abs_bias_mean_N": g.controller_tracking_error_N.abs().mean(),
                "local_lift_positive_n": int(g.local_lift_success.sum()), "local_lift_negative_n": int((1-g.local_lift_success).sum()),
                "full_task_positive_n": int(g.full_task_success.sum()), "full_task_negative_n": int((1-g.full_task_success).sum()),
                "delayed_failure_n": int(((g.local_lift_success == 1) & (g.full_task_success == 0)).sum()),
                "full_label_direct_n": int((g.label_source == "DIRECT_CUMULATIVE_BRANCH_LABEL").sum()),
                "full_label_reconstructed_n": int((g.label_source != "DIRECT_CUMULATIVE_BRANCH_LABEL").sum()),
                "local_label_direct_n": int((g.local_lift_label_source == "DIRECT_CUMULATIVE_BRANCH_LABEL").sum()),
                "local_label_reconstructed_n": int((g.local_lift_label_source != "DIRECT_CUMULATIVE_BRANCH_LABEL").sum()),
                "telemetry_present_n": int(g.telemetry_exists.sum()),
            })
    support = pd.DataFrame(support_rows)
    support.to_csv(OUT / "CURRENT_FORCE_SUPPORT_AUDIT.csv", index=False)

    boundary_rows = []
    for (task, root, band, mu, cid), q in allrows.groupby(["task", "root_id", "friction_band", "mu_GT", "context_id"], sort=True):
        rec = {
            "task": int(task), "root_id": root, "friction_band": band, "friction": mu, "context_id": cid,
            "branches": len(q), "force_cells": q.force_N.nunique(), "force_min_N": q.force_N.min(), "force_max_N": q.force_N.max(),
        }
        rec.update(boundary_status(q, "local_lift_success", "LOCAL"))
        rec.update(boundary_status(q, "full_task_success", "FULL"))
        rec["delayed_failure_n"] = int(((q.local_lift_success == 1) & (q.full_task_success == 0)).sum())
        rec["direct_p_low_force_mean"] = float(q.loc[q.force_N == q.force_N.min(), "p_D_OOF_ensemble"].mean())
        rec["empirical_low_force_full_success_rate"] = float(q.loc[q.force_N == q.force_N.min(), "full_task_success"].mean())
        boundary_rows.append(rec)
    boundaries = pd.DataFrame(boundary_rows)
    assert len(boundaries) == 72
    boundaries.to_csv(OUT / "EXISTING_BOUNDARY_STATUS.csv", index=False)

    task_summary = support[support.row_scope == "TASK_SUMMARY"].copy()
    md = [
        "# Current force support audit", "",
        "Status: `AUTHORITATIVE_720_AUDITED`", "",
        f"Frozen population: `{UTILITY}` (SHA-256 `{sha256(UTILITY)}`).", "",
        f"Collection manifest: `{RUN_MANIFEST}` (SHA-256 `{sha256(RUN_MANIFEST)}`).", "",
        "The 720 rows align one-to-one on task, root, context, force, and repeat. All 720 trajectory files are present. Local lift is reconstructed from the original collector's exact `object_z - initial_z >= 0.03 m` rule; direct cumulative lift labels agree on every row where both exist.", "",
        "| Task | Branches | Roots | Contexts | Force support (N) | Lift + / - | Full + / - | Lift=1, full=0 | Full labels direct/reconstructed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in task_summary.to_dict("records"):
        md.append(
            f"| {int(r['task'])} | {int(r['branches'])} | {int(r['root_families'])} | {int(r['friction_contexts'])} | "
            f"{r['requested_force_min_N']:.4f}–{r['requested_force_max_N']:.4f} | "
            f"{int(r['local_lift_positive_n'])}/{int(r['local_lift_negative_n'])} | "
            f"{int(r['full_task_positive_n'])}/{int(r['full_task_negative_n'])} | {int(r['delayed_failure_n'])} | "
            f"{int(r['full_label_direct_n'])}/{int(r['full_label_reconstructed_n'])} |"
        )
    md += [
        "", "## Boundary interpretation", "",
        "`LEFT_CENSORED` means the lowest tested force already succeeded; the transition lies below observed support and is not estimated. `RIGHT_CENSORED` means even the highest force failed. `STOCHASTIC_OR_NONMONOTONIC` preserves repeat disagreement or force-order reversals rather than inventing a frontier. Only `BRACKETED` rows expose a valid empirical interval `(highest failure, lowest success]`.", "",
        "## Data-quality caveats", "",
        "- Task1 has 140/180 frozen terminal-height fallback full-task labels. Its full-task boundary rows are development evidence with a material reconstructed-label caveat.",
        "- Task1 local-lift labels for rows absent from the cumulative branch table are reconstructed from physical trajectories using the original 3 cm rule.",
        "- Legacy `failure_stage=full_task_rollout` is too coarse. This audit only asserts `LOCAL_LIFT` when the 3 cm criterion fails; downstream failures without direct stage telemetry remain `DOWNSTREAM_UNRESOLVED`.",
        "- Requested force is the acquisition coordinate. Measured steady force, peak, tracking MAE, and bias are retained separately and are not substituted into the candidate identity.",
        "- These are observed TRAIN/DEV data, not fresh TEST evidence.", "",
    ]
    (OUT / "CURRENT_FORCE_SUPPORT_AUDIT.md").write_text("\n".join(md), encoding="utf-8")

    audit = {
        "status": "PASS_WITH_TASK1_LABEL_CAVEAT",
        "rows": len(allrows), "tasks": list(TASKS), "contexts": int(allrows.context_id.nunique()),
        "key_alignment_720_of_720": True, "force_join_tolerance": "12 decimal places; observed maximum source difference 4.440892098500626e-16 N", "telemetry_present_720_of_720": True,
        "direct_lift_vs_reconstructed_mismatches": 0,
        "direct_branch_counts": direct_counts, "canonical_counts": canonical_counts,
        "source_hashes": {str(p): sha256(p) for p in (UTILITY, RUN_MANIFEST)},
        "outputs": ["AUTHORITATIVE_720_BOUNDARY_LABELS.csv", "CURRENT_FORCE_SUPPORT_AUDIT.csv", "EXISTING_BOUNDARY_STATUS.csv"],
    }
    (OUT / "EXISTING_BOUNDARY_AUDIT_QA.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
