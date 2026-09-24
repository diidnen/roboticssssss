#!/usr/bin/env python3
"""Read-only audit of the authoritative old 720 P5-S0-C branches.

This script never edits the source/archive data.  It writes only a new audit
directory under /home/exouser.  The actual squeeze scalar is recomputed from
the recorded local left/right finger normal sensors using the frozen
ForcePositionAction definition: 2 * min(abs(left_z), abs(right_z)).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


OLD_ROOT = Path("/home/exouser/FORTE/gnp_style_continuous_20260830_125107")
OLD_DATA = OLD_ROOT / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
OLD_TELEMETRY = OLD_ROOT / "collection_long2/P5S0C_BRANCH_TELEMETRY"
OLD_LABELS = Path("/home/exouser/FORTE/activeforcing_final_closure_20260902_034923/E3_BRANCH_LABEL_AUDIT.csv")
E3_ROOT = Path("/media/volume/newdata/exouser/activeforcing_e3/P1_SIMPLIFIED_COLLECTION_FORMAL_20260903")
OUT = Path("/home/exouser/OLD720_FORCE_REALIZATION_AUDIT_20260903")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    mask = np.isfinite(a) & np.isfinite(b)
    x, y = a[mask], b[mask]
    if len(x) == 0:
        return {"n": 0, "pearson": None, "spearman": None, "mae_N": None, "rmse_N": None, "bias_N": None}
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else None
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 1 and np.std(rx) > 0 and np.std(ry) > 0 else None
    err = y - x
    return {"n": int(len(x)), "pearson": pearson, "spearman": spearman,
            "mae_N": float(np.mean(np.abs(err))), "rmse_N": float(np.sqrt(np.mean(err ** 2))),
            "bias_N": float(np.mean(err))}


def phase_metrics(td: pd.DataFrame) -> dict:
    td = td.sort_values("step").reset_index(drop=True)
    left = td["left_normal_force_N"].to_numpy(float)
    right = td["right_normal_force_N"].to_numpy(float)
    # This is the frozen controller's physical squeeze convention.
    squeeze = 2.0 * np.minimum(np.abs(left), np.abs(right))
    td = td.copy()
    td["recomputed_squeeze_N"] = squeeze
    grasp = td[td["phase"].astype(str).eq("branch_hold")]
    lift = td[td["phase"].astype(str).eq("lift")]
    pre = td[td["phase"].astype(str).isin(["pre_contact", "approach", "descend"])]
    def q(d: pd.DataFrame, prefix: str) -> dict:
        v = d["recomputed_squeeze_N"].to_numpy(float)
        return {
            f"{prefix}_n_steps": int(len(v)),
            f"{prefix}_force_mean_N": float(np.mean(v)) if len(v) else math.nan,
            f"{prefix}_force_median_N": float(np.median(v)) if len(v) else math.nan,
            f"{prefix}_force_peak_N": float(np.max(v)) if len(v) else math.nan,
            f"{prefix}_left_normal_mean_N": float(np.mean(np.abs(d["left_normal_force_N"].to_numpy(float)))) if len(v) else math.nan,
            f"{prefix}_right_normal_mean_N": float(np.mean(np.abs(d["right_normal_force_N"].to_numpy(float)))) if len(v) else math.nan,
        }
    # The old branch begins after the strict pre-probe grasp state.  No raw
    # pre-contact rows were recorded in this branch telemetry.
    out = {"pre_contact_recorded_steps": int(len(pre)), "grasp_phase": "branch_hold",
           "lift_phase": "lift", **q(grasp, "grasp"), **q(lift, "lift"), **q(pre, "pre_contact")}
    out["raw_left_normal_peak_N"] = float(np.max(np.abs(left))) if len(left) else math.nan
    out["raw_right_normal_peak_N"] = float(np.max(np.abs(right))) if len(right) else math.nan
    out["contact_bilateral_fraction_all"] = float(np.mean((td.contact_left.to_numpy(float) == 1) & (td.contact_right.to_numpy(float) == 1))) if len(td) else math.nan
    out["contact_loss_steps_after_lift"] = int(np.sum((td.step.to_numpy(int) >= int(lift.step.min()) if len(lift) else False) & (td.contact_state.astype(str).to_numpy() == "none")))
    # Local-lift success is the first threshold crossing used by P5-S0-C.
    z0 = float(td.object_z_analysis_only.iloc[0]) if len(td) else math.nan
    dz = td.object_z_analysis_only.to_numpy(float) - z0 if len(td) else np.array([])
    hits = np.flatnonzero(dz >= 0.03)
    out["success_step_derived"] = int(td.step.iloc[int(hits[0])]) if len(hits) else ""
    out["episode_length"] = int(len(td))
    out["episode_duration_s"] = float(td.t_s.iloc[-1]) if len(td) else math.nan
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(OLD_DATA)
    labels = pd.read_csv(OLD_LABELS)
    if len(d) != 720 or d.branch_id.nunique() != 720:
        raise RuntimeError(f"authoritative old dataset check failed rows={len(d)} unique={d.branch_id.nunique()}")
    if len(labels) != 720:
        raise RuntimeError(f"old label audit rows={len(labels)}")
    lab = labels[["branch_id", "local_lift_success"]].copy()
    d = d.merge(lab, on="branch_id", how="left", validate="one_to_one")
    if d.local_lift_success.isna().any():
        raise RuntimeError("missing local lift labels")
    telemetry_paths = [Path(str(x)) for x in d.telemetry_path]
    if sum(p.exists() for p in telemetry_paths) != 720:
        # The copied FORTE archive is authoritative but some rows retain the
        # original Tabero path; recover by basename only, without altering data.
        by_name = {p.name: p for p in OLD_TELEMETRY.glob("*.csv")}
        telemetry_paths = [p if p.exists() else by_name.get(p.name, Path("/missing")) for p in telemetry_paths]
    if sum(p.exists() for p in telemetry_paths) != 720:
        raise RuntimeError(f"telemetry check failed {sum(p.exists() for p in telemetry_paths)}/720")

    rows = []
    raw_cache = {}
    for r, p in zip(d.itertuples(index=False), telemetry_paths):
        td = pd.read_csv(p)
        required = {"left_normal_force_N", "right_normal_force_N", "phase", "step", "t_s", "contact_left", "contact_right", "contact_state", "object_z_analysis_only"}
        if not required <= set(td.columns):
            raise RuntimeError(f"missing raw fields in {p}")
        m = phase_metrics(td)
        req = float(r.requested_force_N)
        grasp_mean = m["grasp_force_mean_N"]
        grasp_peak = m["grasp_force_peak_N"]
        row = {
            "branch_id": r.branch_id, "context_id": r.context_id, "root_id": r.root_id,
            "task": int(r.task), "split": "TRAIN" if "_train_" in str(r.root_id) else "UNKNOWN",
            "friction_band": r.friction_band, "requested_force_N": req,
            "requested_force_bucket_N": int(round(req)), "manifest_realized_force_N": float(r.realized_force_N),
            "lift_success": int(r.local_lift_success), "full_task_success": int(r.full_task_success_y),
            "success_step": m["success_step_derived"], "episode_length": m["episode_length"],
            "telemetry_path": str(p), "pre_contact_recorded_steps": m["pre_contact_recorded_steps"],
            "grasp_force_mean_N": grasp_mean, "grasp_force_median_N": m["grasp_force_median_N"], "grasp_force_peak_N": grasp_peak,
            "lift_force_mean_N": m["lift_force_mean_N"], "lift_force_median_N": m["lift_force_median_N"], "lift_force_peak_N": m["lift_force_peak_N"],
            "grasp_left_normal_mean_N": m["grasp_left_normal_mean_N"], "grasp_right_normal_mean_N": m["grasp_right_normal_mean_N"],
            "lift_left_normal_mean_N": m["lift_left_normal_mean_N"], "lift_right_normal_mean_N": m["lift_right_normal_mean_N"],
            "grasp_contact_bilateral_fraction": float(td[td.phase.astype(str).eq("branch_hold")].apply(lambda x: int(x.contact_left == 1 and x.contact_right == 1), axis=1).mean()) if (td.phase.astype(str).eq("branch_hold")).any() else math.nan,
            "lift_contact_bilateral_fraction": float(td[td.phase.astype(str).eq("lift")].apply(lambda x: int(x.contact_left == 1 and x.contact_right == 1), axis=1).mean()) if (td.phase.astype(str).eq("lift")).any() else math.nan,
            "contact_loss_steps_after_lift": m["contact_loss_steps_after_lift"],
            "grasp_mean_overshoot": int(grasp_mean > req + 2.0),
            "grasp_peak_overshoot": int(grasp_peak > req + 3.0),
            "object_z_initial_m": float(td.object_z_analysis_only.iloc[0]), "object_z_final_m": float(td.object_z_analysis_only.iloc[-1]),
            "object_z_max_dz_m": float(np.max(td.object_z_analysis_only.to_numpy(float) - float(td.object_z_analysis_only.iloc[0]))),
            "measured_force_telemetry_mean_N": float(td["measured_force_N"].mean()),
        }
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "OLD720_PHASE_FORCE_METRICS.csv", index=False)

    # Requested vs raw finger-sensor squeeze correlations.
    req = metrics.requested_force_N.to_numpy(float)
    grasp = metrics.grasp_force_mean_N.to_numpy(float)
    lift = metrics.lift_force_mean_N.to_numpy(float)
    corr = [{"comparison": "requested_vs_grasp_actual", **stats(req, grasp)}, {"comparison": "requested_vs_lift_actual", **stats(req, lift)}]
    write_csv(OUT / "OLD720_REQUESTED_ACTUAL_CORRELATIONS.csv", corr)

    bucket_rows = []
    for b in range(1, 9):
        q = metrics[metrics.requested_force_bucket_N == b]
        bucket_rows.append({"requested_bucket_N": b, "n": len(q),
                            "requested_exact_min_N": q.requested_force_N.min() if len(q) else "", "requested_exact_max_N": q.requested_force_N.max() if len(q) else "",
                            "grasp_actual_mean_N": q.grasp_force_mean_N.mean() if len(q) else "", "grasp_actual_std_N": q.grasp_force_mean_N.std(ddof=1) if len(q) > 1 else "",
                            "lift_actual_mean_N": q.lift_force_mean_N.mean() if len(q) else "", "lift_actual_std_N": q.lift_force_mean_N.std(ddof=1) if len(q) > 1 else "",
                            "lift_SR": q.lift_success.mean() if len(q) else ""})
    write_csv(OUT / "OLD720_FORCE_BUCKET_SUMMARY.csv", bucket_rows)

    low = metrics[metrics.requested_force_N <= 3.0]
    extreme = metrics[(metrics.requested_force_N <= 3.0) & (metrics.grasp_force_mean_N >= 5.0)]
    low_plus2 = metrics[(metrics.requested_force_N <= 3.0) & (metrics.grasp_force_mean_N >= metrics.requested_force_N + 2.0)]
    print("LOW_COMMAND_COUNTS", {"requested_le_3": len(low), "actual_ge_req_plus2": len(low_plus2), "actual_ge_5": len(extreme)})
    print("LOW_COMMAND_LIFT_SR", float(low.lift_success.mean()) if len(low) else "N/A")
    # If the exact requested<=3N population is empty, print the nearest ten
    # observed branches as an explicit boundary diagnostic, never relabeling
    # them as <=3N.
    examples = (low_plus2.sort_values("requested_force_N") if len(low_plus2) else metrics.sort_values("requested_force_N").head(10))
    write_csv(OUT / "OLD720_LOW_COMMAND_HIGH_ACTUAL_EXAMPLES.csv", examples.head(20).to_dict("records"))

    overshoot = [{"scope": "overall", "n": len(metrics), "grasp_mean_overshoot_rate": metrics.grasp_mean_overshoot.mean(), "grasp_peak_overshoot_rate": metrics.grasp_peak_overshoot.mean()},
                 {"scope": "success", "n": int(metrics.lift_success.sum()), "grasp_mean_overshoot_rate": metrics.loc[metrics.lift_success == 1, "grasp_mean_overshoot"].mean(), "grasp_peak_overshoot_rate": metrics.loc[metrics.lift_success == 1, "grasp_peak_overshoot"].mean()},
                 {"scope": "failure", "n": int((metrics.lift_success == 0).sum()), "grasp_mean_overshoot_rate": metrics.loc[metrics.lift_success == 0, "grasp_mean_overshoot"].mean(), "grasp_peak_overshoot_rate": metrics.loc[metrics.lift_success == 0, "grasp_peak_overshoot"].mean()}]
    for b in range(1, 9):
        q = metrics[metrics.requested_force_bucket_N == b]
        if len(q): overshoot.append({"scope": f"bucket_{b}N", "n": len(q), "grasp_mean_overshoot_rate": q.grasp_mean_overshoot.mean(), "grasp_peak_overshoot_rate": q.grasp_peak_overshoot.mean()})
    write_csv(OUT / "OLD720_OVERSHOOT_SUMMARY.csv", overshoot)

    # Same-command success/failure force comparison is undefined for lift:
    # the authoritative local-lift label is degenerate (720/720 positive).
    same_force = []
    for b in range(1, 9):
        q = metrics[metrics.requested_force_bucket_N == b]
        same_force.append({"requested_bucket_N": b, "n": len(q), "actual_grasp_min_N": q.grasp_force_mean_N.min() if len(q) else "", "actual_grasp_max_N": q.grasp_force_mean_N.max() if len(q) else "", "actual_grasp_mean_N": q.grasp_force_mean_N.mean() if len(q) else "", "actual_grasp_std_N": q.grasp_force_mean_N.std(ddof=1) if len(q)>1 else "", "success_n": int(q.lift_success.sum()) if len(q) else "", "failure_n": int((q.lift_success==0).sum()) if len(q) else ""})
    write_csv(OUT / "OLD720_WITHIN_REQUESTED_FORCE_VARIANCE.csv", same_force)

    # Current E3 common-phase comparison: measured_squeeze_N is the logged
    # scalar available in E3, and is not substituted into old raw metrics.
    e3_rows = []
    for p in sorted(E3_ROOT.glob("P1_SIMPLIFIED_ROOT*/logs/*_steps.csv")):
        td = pd.read_csv(p)
        if "measured_squeeze_N" not in td:
            continue
        match = re.search(r"_F([0-9]+(?:\.[0-9]+)?)N$", p.parent.parent.name)
        if not match:
            continue
        req0 = float(match.group(1))
        # E3 step logs do not have the old runner's named phases.  Use only
        # logged physical events: grasp is bilateral contact before the first
        # 3-cm lift threshold; lift is the suffix after that threshold.
        contact = td["contact"].to_numpy(float) > 0 if "contact" in td else np.zeros(len(td), bool)
        z = td["obj_dz"].to_numpy(float) if "obj_dz" in td else np.full(len(td), np.nan)
        lift_hits = np.flatnonzero(z >= 0.03)
        lift_start = int(lift_hits[0]) if len(lift_hits) else len(td)
        grasp_mask = contact & (np.arange(len(td)) < lift_start)
        lift_mask = np.arange(len(td)) >= lift_start if len(lift_hits) else np.zeros(len(td), bool)
        e3_rows.append({"path": str(p), "requested_force_N": req0,
                        "grasp_actual_mean_N": float(td.loc[grasp_mask, "measured_squeeze_N"].mean()) if grasp_mask.any() else math.nan,
                        "lift_actual_mean_N": float(td.loc[lift_mask, "measured_squeeze_N"].mean()) if lift_mask.any() else math.nan,
                        "grasp_peak_N": float(td.loc[grasp_mask, "measured_squeeze_N"].max()) if grasp_mask.any() else math.nan,
                        "lift_peak_N": float(td.loc[lift_mask, "measured_squeeze_N"].max()) if lift_mask.any() else math.nan,
                        "lift_success": int(len(lift_hits) > 0), "n": len(td)})
    e3 = pd.DataFrame(e3_rows)
    if len(e3):
        e3.to_csv(OUT / "E3_COMMON_PHASE_FORCE_METRICS.csv", index=False)
    e3_corr = []
    if len(e3):
        e3_corr = [{"dataset": "E3", "comparison": "requested_vs_grasp_actual", **stats(e3.requested_force_N.to_numpy(float), e3.grasp_actual_mean_N.to_numpy(float))}, {"dataset": "E3", "comparison": "requested_vs_lift_actual", **stats(e3.requested_force_N.to_numpy(float), e3.lift_actual_mean_N.to_numpy(float))}]
    old_corr = [{"dataset": "OLD720", **r} for r in corr]
    if len(e3):
        e3_overshoot = {
            "dataset": "E3", "grasp_mean_overshoot_rate": float((e3.grasp_actual_mean_N > e3.requested_force_N + 2.0).mean()),
            "grasp_peak_overshoot_rate": float((e3.grasp_peak_N > e3.requested_force_N + 3.0).mean()),
            "lift_mean_overshoot_rate": float((e3.lift_actual_mean_N > e3.requested_force_N + 2.0).mean()),
            "lift_peak_overshoot_rate": float((e3.lift_peak_N > e3.requested_force_N + 3.0).mean()),
            "n": len(e3),
        }
    else:
        e3_overshoot = {"dataset": "E3", "n": 0}
    write_csv(OUT / "OLD720_VS_E3_FORCE_COMPARISON.csv", old_corr + e3_corr)
    write_csv(OUT / "OLD720_VS_E3_OVERSHOOT_COMPARISON.csv", [{"dataset": "OLD720", "grasp_mean_overshoot_rate": float(metrics.grasp_mean_overshoot.mean()), "grasp_peak_overshoot_rate": float(metrics.grasp_peak_overshoot.mean()), "lift_mean_overshoot_rate": float((metrics.lift_force_mean_N > metrics.requested_force_N + 2.0).mean()), "lift_peak_overshoot_rate": float((metrics.lift_force_peak_N > metrics.requested_force_N + 3.0).mean()), "n": len(metrics)}, e3_overshoot])

    provenance = {
        "status": "PASS", "old720_total": len(d), "old720_unique_branch_ids": int(d.branch_id.nunique()),
        "dataset_path": str(OLD_DATA), "dataset_sha256": sha256(OLD_DATA), "telemetry_dir": str(OLD_TELEMETRY), "telemetry_files": len(telemetry_paths),
        "old_collection_script": "/home/exouser/FORTE/gnp_style_continuous_collect.py",
        "old_downstream_runner": "/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py",
        "old_force_controller": "/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py",
        "old_force_controller_sha256": sha256(Path("/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py")),
        "current_e3_force_controller": "/media/volume/newdata/exouser/Tabero_e3lh/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py",
        "current_e3_force_controller_sha256": sha256(Path("/media/volume/newdata/exouser/Tabero_e3lh/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py")),
        "force_scalar_definition": "2*min(abs(left_normal_force_N),abs(right_normal_force_N)); raw local finger sensor z channels",
        "old_phase_definition": "PRE_CONTACT not recorded in branch telemetry; GRASP=branch_hold; LIFT=lift; later=transit/over_basket/place/release/settle",
        "old_local_lift_label": "trace reaches post-lift phase; audited local_lift_success=1 for all 720",
        "old_full_success": "lift_success==1 and max_basket_contact>0.05 and dropped==0",
        "requested_force_range_N": [float(d.requested_force_N.min()), float(d.requested_force_N.max())],
        "requested_force_unique_count": int(d.requested_force_N.nunique()), "requested_le_3_count": len(low),
        "grasp_mean_overshoot_rate": float(metrics.grasp_mean_overshoot.mean()), "grasp_peak_overshoot_rate": float(metrics.grasp_peak_overshoot.mean()),
        "low_command_high_actual_count": len(low_plus2), "low_command_high_actual_ge5_count": len(extreme),
        "local_lift_class_counts": {str(k): int(v) for k, v in d.local_lift_success.value_counts().to_dict().items()},
        "e3_rows_used": len(e3_rows),
        "controller_parity": "DIFFERENT",
        "controller_parity_detail": "force_position_action.py and tactile env config are byte-identical, but old P5 adds external _force_servo and strict post-probe handoff while E3 overwrites force slots after VLA inference without that outer servo",
        "no_source_or_archive_modified": True,
    }
    (OUT / "OLD720_FORCE_REALIZATION_AUDIT.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
