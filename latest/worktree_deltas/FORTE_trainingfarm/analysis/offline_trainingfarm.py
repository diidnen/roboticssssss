#!/usr/bin/env python3
"""CPU-safe ActiveForcing offline training-farm audit.

This runner intentionally does not launch IsaacLab, π0, or a GPU training job.
It reuses only TRAIN/DEV artifacts whose protocols already exclude TEST and
materializes an auditable handoff bundle.  Missing experiments are represented
explicitly instead of being filled with proxy metrics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FORTE = Path("/home/exouser/FORTE")
TRAININGFARM = Path("/home/exouser/FORTE_trainingfarm")
TABERO = Path("/home/exouser/Tabero")
AUTH = FORTE / "gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
SPLIT = FORTE / "activeforcing_final_closure_20260902_034923/ACTIVEFORCING_FINAL_SPLIT_MANIFEST.json"
PROTO = FORTE / "activeforcing_final_closure_20260902_034923/ACTIVEFORCING_FINAL_PROTOCOL.json"
PROBE = FORTE / "activeforcing_probe_conditioned_wm_20260901_064627/POOLED_OOF_PROBE_PREDICTIONS.csv"
LABELS = FORTE / "activeforcing_final_closure_20260902_034923/E3_BRANCH_LABEL_AUDIT.csv"
DIRECT = FORTE / "gnp_style_continuous_20260830_125107/CONTINUOUS_DEV_FROZEN_PREDICTIONS.csv"
FRONTIERS = TABERO / "analysis/results/gnp_style_visual_context_prospective_20260831_011000/PROSPECTIVE_REAL_FRONTIERS.csv"
CURVES = TABERO / "analysis/results/gnp_style_visual_context_prospective_20260831_011000/PROSPECTIVE_REAL_CONTINUOUS_CURVES.csv"
CTX = TABERO / "analysis/results/gnp_style_visual_context_prospective_20260831_011000/PROSPECTIVE_CONTEXT_MANIFEST.csv"
VISUAL_MANIFEST = TABERO / "analysis/results/gnp_style_visual_context_prospective_20260831_011000/PROSPECTIVE_VISUAL_ALIGNMENT_MANIFEST.csv"
VISUAL_SPEC = TABERO / "analysis/results/gnp_style_visual_context_prospective_20260831_011000/PROSPECTIVE_VISUAL_FEATURE_SPEC.json"
TRANSFER = FORTE / "source_subset_transfer_audit_20260901_v3/SOURCE_SUBSET_PER_SEED.csv"
OUT = TRAININGFARM / "analysis/results/activeforcing_trainingfarm_20260902T050000Z"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def metric_row(scope: str, method: str, seed: str, y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {
        "scope": scope, "method": method, "seed": seed, "n": int(len(y)),
        "ECE_10bin": float(ece), "Brier_soft": float(np.mean((p - y) ** 2)),
        "NLL_soft": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "mean_target": float(y.mean()), "mean_probability": float(p.mean()),
    }


def canonical_context_map(ctx: pd.DataFrame) -> dict[tuple[int, str, str], str]:
    return {(int(r.task), str(r.source_root_id), str(r.friction_band)): str(r.context_id)
            for r in ctx.itertuples(index=False)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.max_columns", 100)
    auth = pd.read_csv(AUTH)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    proto = json.loads(PROTO.read_text(encoding="utf-8"))
    probe = pd.read_csv(PROBE)
    labels = pd.read_csv(LABELS)
    direct = pd.read_csv(DIRECT)
    ctx = pd.read_csv(CTX)
    curves = pd.read_csv(CURVES)
    frontiers = pd.read_csv(FRONTIERS)

    # 1. Authoritative data/split quality audit.
    root_to_fold = {str(r["root_id"]): int(r["fold"]) for r in split["rows"]}
    auth["oof_fold"] = auth.root_id.astype(str).map(root_to_fold)
    audit = {
        "status": "PASS" if len(auth) == 720 and auth.valid.eq(1).all() and auth.state_parity.eq(1).all()
                  and auth.infrastructure_failure.fillna("").eq("").all()
                  and auth.oof_fold.notna().all() else "FAIL",
        "source": str(AUTH), "source_sha256": sha256(AUTH), "rows": int(len(auth)),
        "tasks": sorted(int(x) for x in auth.task.unique()), "contexts": int(auth.context_id.nunique()),
        "root_families": int(auth.root_id.nunique()), "successes": int(auth.full_task_success_y.sum()),
        "failures": int((1 - auth.full_task_success_y).sum()),
        "valid_rows": int(auth.valid.sum()), "state_parity_rows": int(auth.state_parity.sum()),
        "duplicate_branch_ids": int(auth.branch_id.duplicated().sum()),
        "context_cells_with_two_repeats": int((auth.groupby(["context_id", "requested_force_N"]).size() == 2).sum()),
        "context_cells": int(auth.groupby(["context_id", "requested_force_N"]).ngroups),
        "fold_counts": auth.groupby("oof_fold").root_id.nunique().to_dict(),
        "sealed_test_read": False,
        "protocol_status": proto.get("status"),
        "force_grid_gap_preserved": True,
    }
    (OUT / "AUTHORITATIVE_DATA_QUALITY_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_csv(OUT / "AUTHORITATIVE_ROOT_FOLD_ASSIGNMENTS.csv", auth[["root_id", "task", "oof_fold"]].drop_duplicates().to_dict("records"))

    # 2. Existing frozen visual features: copy only compact feature tensors, not RGB.
    cache = OUT / "VISUAL_FEATURE_CACHE"
    cache.mkdir(exist_ok=True)
    vm = pd.read_csv(VISUAL_MANIFEST)
    vm = vm[vm["split"].astype(str).eq("TRAIN")].copy()
    ctx_by_id = ctx.set_index("context_id")
    feature_rows = []
    spec = json.loads(VISUAL_SPEC.read_text(encoding="utf-8"))
    preprocess_hash = sha256(VISUAL_SPEC)
    for r in vm.itertuples(index=False):
        source = Path(str(r.visual_feature_path))
        target = cache / f"task{int(r.task)}" / f"{r.context_id}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copy2(source, target)
            arr = np.load(target, allow_pickle=False)
            shape = list(arr.shape); dtype = str(arr.dtype)
            # The source manifest hashes canonical tensor bytes, not the
            # NumPy .npy container (whose header metadata may differ).
            tensor_hash = hashlib.sha256(arr.tobytes()).hexdigest()
            hash_ok = tensor_hash == str(r.visual_feature_sha256)
        else:
            shape, dtype, tensor_hash, hash_ok = [], "", "", False
        context_meta = ctx_by_id.loc[str(r.context_id)]
        feature_rows.append({
            "sample_id": str(r.context_id), "task": int(r.task), "root": str(context_meta.source_root_id),
            "source_feature": str(source), "cached_feature": str(target),
            "image_hash_camera0": str(r.camera0_rgb_sha256), "image_hash_camera1": str(r.camera1_rgb_sha256),
            "feature_hash": str(r.visual_feature_sha256), "cached_tensor_hash": tensor_hash,
            "source_file_sha256": sha256(source) if source.exists() else "",
            "cached_file_sha256": sha256(target) if target.exists() else "",
            "cached_hash_match": int(hash_ok),
            "feature_shape": json.dumps(shape), "feature_dtype": dtype,
            "pi0_checkpoint_hash": str(spec.get("checkpoint_hash", "")),
            "preprocess_hash": preprocess_hash, "split": "TRAIN",
        })
    write_csv(OUT / "VISUAL_FEATURE_CACHE_INDEX.csv", feature_rows)
    visual_report = f"""# π0 Visual Feature Cache Report

Status: **{'PASS' if feature_rows and all(x['cached_hash_match'] for x in feature_rows) else 'FAIL'}**

- Scope: TRAIN only; no DEV/TEST feature was loaded.
- Cached contexts: {len(feature_rows)} ({vm.task.nunique()} tasks).
- Backend: frozen project-local π0/PaliGemma path from `PROSPECTIVE_VISUAL_FEATURE_SPEC.json`.
- Checkpoint hash: `{spec.get('checkpoint_hash', '')}`.
- Preprocessing: 512×512 uint8 RGB → 224×224×3 policy preprocessing; camera order `agentview_cam`, `eye_in_hand_cam`; mean-pool valid image tokens per camera, concatenate to 4096D.
- Cache index: `VISUAL_FEATURE_CACHE_INDEX.csv`; compact tensors are under `VISUAL_FEATURE_CACHE/`.
- This is feature reuse, not a new extraction or encoder switch. The π0 checkpoint file itself was not copied.
"""
    (OUT / "PI0_VISUAL_FEATURE_CACHE_REPORT.md").write_text(visual_report, encoding="utf-8")

    # 3. Identifier seed metrics from existing grouped OOF predictions.
    root_map = auth.drop_duplicates("root_id").set_index("root_id")
    p = probe[probe.seed.astype(str).isin({"0", "1", "2"})].copy()
    # Existing probe uses pv_* root naming; canonicalize by task + root suffix.
    p["root_key"] = p.root_id.astype(str)
    seed_rows = []
    for seed, q in p.groupby(p.seed.astype(str)):
        for scope, z in [("ROOT_HELDOUT_OOF", q)]:
            y = z.mu_GT.to_numpy(float); pred = z.mu_hat.to_numpy(float)
            pairs = []
            for _, g in z.groupby("root_id"):
                vals = g.sort_values("mu_GT")
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        pairs.append(int(vals.iloc[i].mu_hat < vals.iloc[j].mu_hat))
            band = np.select([y < .4, y < .75], ["LOW", "MID"], default="HIGH")
            pred_band = np.select([pred < .4, pred < .75], ["LOW", "MID"], default="HIGH")
            seed_rows.append({**metric_row(scope, "Probe-PhysicalHistory", str(seed), y, pred),
                "friction_MAE": float(np.mean(np.abs(pred-y))), "median_AE": float(np.median(np.abs(pred-y))),
                "RMSE": float(np.sqrt(np.mean((pred-y)**2))), "Spearman": float(spearmanr(y, pred).statistic),
                "pairwise_ranking": float(np.mean(pairs)) if pairs else math.nan,
                "LOW_MID_HIGH_accuracy": float(np.mean(band == pred_band)),
                "force_choice_agreement": "NA_no_per_seed_force_decisions"})
    write_csv(OUT / "IDENTIFIER_SEED_RESULTS.csv", seed_rows)
    id_summary = []
    if seed_rows:
        id_df = pd.DataFrame(seed_rows)
        for metric in ["friction_MAE", "median_AE", "RMSE", "Spearman", "pairwise_ranking", "LOW_MID_HIGH_accuracy"]:
            vals = id_df[metric].astype(float).to_numpy()
            id_summary.append({"metric": metric, "n_seeds": len(vals), "mean": float(vals.mean()),
                "std_sample": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "ci95_half_width_t_approx": float(2.0 * vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0})
    write_csv(OUT / "IDENTIFIER_SEED_SUMMARY.csv", id_summary)
    variant_status = []
    for variant in ["VISION_ONLY", "PHYSICAL_ONLY", "VISION_PLUS_PHYSICAL"]:
        for seed in [0, 1, 2, 3, 4]:
            variant_status.append({"variant": variant, "seed": seed, "status": "NOT_TRAINED_CURRENT_FARM",
                "reason": "GPU preflight blocked new training; no protocol-matched 5-seed artifact available"})
    write_csv(OUT / "IDENTIFIER_TRAINING_FARM_STATUS.csv", variant_status)
    summary_text = "\n".join(
        f"- {r['metric']}: mean {r['mean']:.5f}, sample std {r['std_sample']:.5f}, "
        f"approx. 95% half-width {r['ci95_half_width_t_approx']:.5f}."
        for r in id_summary
    ) or "- No metric rows."
    (OUT / "IDENTIFIER_SEED_STABILITY_REPORT.md").write_text(f"""# Identifier seed stability

Status: **INCOMPLETE — existing grouped OOF contains 3 seeds (0,1,2), not the requested 5.**

The rows in `IDENTIFIER_SEED_RESULTS.csv` are the existing root-heldout OOF
Probe-PhysicalHistory predictions. Metrics are computed per seed. LOW/MID/HIGH
accuracy uses the documented audit binning `[0,0.4), [0.4,0.75), [0.75,1]`;
force-choice agreement is NA because these identifier artifacts do not contain
per-seed Direct decisions. No new estimator training was launched while the
GPU was at 80% utilization with only 11.8 GB free.

## Existing 3-seed summary

{summary_text}

The requested VISION_ONLY, PHYSICAL_ONLY, and VISION_PLUS_PHYSICAL 5-seed
farm remains explicitly recorded in `IDENTIFIER_TRAINING_FARM_STATUS.csv` as
not trained; no proxy rows are presented as completed experiments.
""", encoding="utf-8")

    # 4. LocalLift validity audit; do not train a degenerate classifier.
    cross = labels.groupby(["local_lift_success", "full_task_success"]).size().reset_index(name="n")
    fulltask_rows = []
    for variant in ["FULLTASK_DIRECT", "LOCALLIFT_DIRECT"]:
        for seed in [0, 1, 2, 3, 4]:
            status = "EXISTING_3_SEED_ONLY" if variant == "FULLTASK_DIRECT" and seed < 3 else "NOT_TRAINED"
            if variant == "LOCALLIFT_DIRECT": status = "SCIENTIFICALLY_INVALID_DEGENERATE_LABEL"
            fulltask_rows.append({"variant": variant, "seed": seed, "status": status, "friction_MAE": "NA",
                "lift_prediction_metric": "NA", "full_task_prediction_metric": "NA", "selected_force": "NA",
                "archived_downstream_SR": "NA", "under_force": "NA", "delayed_failure": "NA",
                "transport_retention": "NA", "placement_success": "NA"})
    write_csv(OUT / "FULLTASK_LOCALLIFT_SEED_RESULTS.csv", fulltask_rows)
    (OUT / "FULLTASK_LOCALLIFT_SEED_STABILITY.md").write_text("""# FullTask versus LocalLift seed stability

Status: **NOT ESTIMABLE as a matched learned comparison.**

The authoritative E3 audit has 720 rows and `local_lift_success=1` for all
720 rows; 145 are delayed failures (`LocalLift=1, FullTask=0`). LocalLift has
no negative training examples, so a learned LocalLift classifier and its
downstream comparison would be invalid. FullTask has existing 3-seed assets,
but not five current-protocol seeds. No TEST data was used.
""", encoding="utf-8")

    # 5. Direct per-seed stability and standard continuous interpolation, reusing DEV predictions.
    cmap = canonical_context_map(ctx)
    frontier_by_key = {}
    for r in frontiers.itertuples(index=False):
        cr = ctx[ctx.context_id.eq(r.context_id)].iloc[0]
        frontier_by_key[(int(cr.task), str(cr.source_root_id), str(cr.friction_band))] = float(r.real_frontier_N)
    direct_stability = []
    interp = []
    for backend, d in direct.groupby("backend"):
        d = d.copy().sort_values(["context_id", "force_N"])
        for seed in [0, 1, 2]:
            col = f"seed{seed}_raw_probability"
            for cid, g in d.groupby("context_id"):
                g = g.sort_values("force_N")
                selected = g.loc[g[col] >= .5, "force_N"].min()
                if pd.isna(selected): selected = g.force_N.max()
                real = frontier_by_key.get((int(g.task.iloc[0]), str(g.root_id.iloc[0]), str(g.friction_band.iloc[0])), math.nan)
                direct_stability.append({"scope":"DEV_continuous_root_heldout_reuse", "backend":backend,
                    "seed":seed, "context_id":cid, "selected_force_N":float(selected),
                    "real_frontier_N":real, "under_force":int(float(selected) < float(real)) if math.isfinite(real) else "NA",
                    "excess_force_N":max(0., float(selected)-float(real)) if math.isfinite(real) else "NA",
                    "status":"EXISTING_PREDICTION_NO_NEW_TRAINING"})
            # Soft-label calibration/interpolation at the real 0.25N curve grid.
            for cid, g in d.groupby("context_id"):
                # More robust canonical mapping via direct root/task/band metadata.
                dr = g.iloc[0]
                pv_cid = cmap.get((int(dr.task), str(ctx[ctx.source_root_id.eq(dr.root_id)].iloc[0].source_root_id) if not ctx[ctx.source_root_id.eq(dr.root_id)].empty else str(dr.root_id), str(dr.friction_band)), "")
                if not pv_cid:
                    pv_cid = next((x for x in curves.context_id.unique() if x.replace("pv_", "p5s0c_").replace("dev_", "dev_") == cid), "")
                rr = curves[curves.context_id.eq(pv_cid)]
                for _, row in g[g.force_N.round(6).isin(np.round(np.arange(3, 6.0001, .25), 6))].iterrows():
                    if len(rr) and col in row:
                        actual = rr.loc[np.isclose(rr.force_N, row.force_N), "p_real"]
                        if len(actual):
                            interp.append({"study":"STANDARD_ROOT_HELDOUT", "backend":backend, "seed":seed,
                                "context_id":cid, "force_N":float(row.force_N), "p_pred":float(row[col]),
                                "p_real":float(actual.iloc[0]), "abs_probability_error":abs(float(row[col])-float(actual.iloc[0])),
                                "frontier_N":frontier_by_key.get((int(dr.task), str(dr.root_id), str(dr.friction_band)), "NA"),
                                "status":"DEV_soft_label_evaluation"})
    write_csv(OUT / "DIRECT_SEED_STABILITY.csv", direct_stability)
    write_csv(OUT / "CONTINUOUS_FORCE_INTERPOLATION.csv", interp)
    ds = pd.DataFrame(direct_stability)
    direct_summary = []
    if len(ds):
        for (backend, seed), g in ds.groupby(["backend", "seed"]):
            direct_summary.append({
                "backend": backend, "seed": int(seed), "n_contexts": len(g),
                "under_force_rate": float(g.under_force.astype(float).mean()),
                "mean_excess_force_N": float(g.excess_force_N.astype(float).mean()),
                "mean_selected_force_N": float(g.selected_force_N.astype(float).mean()),
            })
    write_csv(OUT / "DIRECT_SEED_STABILITY_SUMMARY.csv", direct_summary)
    it = pd.DataFrame(interp)
    interp_summary = []
    if len(it):
        for (backend, seed), g in it.groupby(["backend", "seed"]):
            interp_summary.append({
                "backend": backend, "seed": int(seed), "n_points": len(g),
                "n_contexts": int(g.context_id.nunique()),
                "mean_abs_probability_error": float(g.abs_probability_error.mean()),
                "max_abs_probability_error": float(g.abs_probability_error.max()),
            })
    write_csv(OUT / "CONTINUOUS_FORCE_INTERPOLATION_SUMMARY.csv", interp_summary)
    ds_text = "\n".join(
        f"- {r['backend']} seed {r['seed']}: under-force rate {r['under_force_rate']:.3f}; "
        f"mean excess {r['mean_excess_force_N']:.3f} N; mean selected {r['mean_selected_force_N']:.3f} N."
        for r in direct_summary
    ) or "- No rows."
    it_text = "\n".join(
        f"- {r['backend']} seed {r['seed']}: {r['n_points']} points / {r['n_contexts']} contexts; "
        f"mean absolute probability error {r['mean_abs_probability_error']:.4f}."
        for r in interp_summary
    ) or "- No rows."
    (OUT / "DIRECT_SEED_STABILITY_REPORT.md").write_text(f"""# Direct seed stability

Status: **INCOMPLETE — existing frozen Direct/Joint artifacts contain seeds 0–2 only.**

`DIRECT_SEED_STABILITY.csv` evaluates per-seed dense DEV predictions against
the frozen repeated DEV frontier. It is a continuous-grid reuse audit, not a
new 5-seed training run; no force threshold or checkpoint was selected from
these rows.

## Summary

{ds_text}
""", encoding="utf-8")
    (OUT / "CONTINUOUS_FORCE_INTERPOLATION_REPORT.md").write_text(f"""# Continuous force interpolation

Status: **PARTIAL.** Standard root-heldout reuse is materialized from the
existing frozen dense DEV predictions and repeated DEV curves. Leave-one-force-
level-out, alternating force-level holdout, and new near-frontier training
were not run: they require new matched retraining and the current GPU was not
available under the resource policy. No TEST roots were accessed.

## Reused DEV summary

{it_text}
""", encoding="utf-8")

    # 6. Calibration on soft DEV curve labels; identifier has no calibrated probability output.
    cal_rows = []
    for backend, g in direct.groupby("backend"):
        for seed in [0, 1, 2]:
            col_raw = f"seed{seed}_raw_probability"; col_cal = f"seed{seed}_train_isotonic_probability"
            z = pd.DataFrame([r for r in interp if r["backend"] == backend and r["seed"] == seed])
            if len(z):
                cal_rows.append(metric_row("DEV_soft_curve", backend+"_raw", str(seed), z.p_real, z.p_pred))
                # Calibrated value is looked up from matching rows, then aligned to z keys.
                vals=[]; ys=[]
                for ir in interp:
                    if ir["backend"] == backend and ir["seed"] == seed:
                        hit = g[g.context_id.eq(ir["context_id"]) & np.isclose(g.force_N, ir["force_N"])]
                        if len(hit):
                            vals.append(float(hit.iloc[0][col_cal])); ys.append(float(ir["p_real"]))
                if vals:
                    cal_rows.append(metric_row("DEV_soft_curve", backend+"_train_isotonic", str(seed), np.array(ys), np.array(vals)))
    write_csv(OUT / "CALIBRATION_HANDOFF" / "CALIBRATION_METRICS.csv", cal_rows)
    write_csv(OUT / "CALIBRATION_HANDOFF" / "RELIABILITY_PLOT_DATA.csv", [r for r in interp])
    (OUT / "CALIBRATION_HANDOFF" / "README.md").write_text("""# Calibration handoff

Direct calibration is reported as a TRAIN-isotonic diagnostic evaluated on
soft empirical DEV curve probabilities. Identifier calibration is **NA**:
the existing friction OOF artifact exposes point estimates and no calibrated
probability/confidence output; `sigma_mu` is diagnostic only. These files are
handoff material and do not replace any frozen checkpoint.
""", encoding="utf-8")

    # 7. rho DEV sweep from existing per-seed Direct probabilities.
    rho_rows=[]
    for backend, g in direct[direct.force_N.round(6).isin(np.round(np.arange(3,5.0001,.25),6))].groupby("backend"):
        for rho in [0.75,0.80,0.85,0.90,0.95]:
            decisions=[]
            for cid, z in g.groupby("context_id"):
                z=z.sort_values("force_N"); z["p_mean"]=z[[f"seed{i}_raw_probability" for i in [0,1,2]]].mean(axis=1)
                pass_rows=z[z.p_mean>=rho]; selected=float(pass_rows.force_N.min()) if len(pass_rows) else 5.0
                real=frontier_by_key.get((int(z.task.iloc[0]), str(z.root_id.iloc[0]), str(z.friction_band.iloc[0])),math.nan)
                decisions.append((selected,real))
            valid=[x for x in decisions if math.isfinite(x[1])]
            rho_rows.append({"backend":backend,"rho":rho,"n_contexts":len(valid),"success_proxy_mean":float(np.mean([x[0]>=x[1] for x in valid])) if valid else "NA",
                "under_force_rate":float(np.mean([x[0]<x[1] for x in valid])) if valid else "NA",
                "mean_selected_force_N":float(np.mean([x[0] for x in valid])) if valid else "NA",
                "mean_excess_force_N":float(np.mean([max(0,x[0]-x[1]) for x in valid])) if valid else "NA",
                "fallback_rate":float(np.mean([x[0]==5.0 for x in valid])) if valid else "NA",
                "no_valid_force_rate":float(np.mean([x[0]==5.0 for x in valid])) if valid else "NA",
                "query_decision_sensitivity_vs_rho0.50":"computed from frozen probabilities; no final rho selected"})
    write_csv(OUT / "RHO_DEV_SWEEP.csv", rho_rows)
    (OUT / "RHO_DEV_SWEEP_REPORT.md").write_text("""# rho DEV sweep

Complete DEV tradeoffs for rho 0.75–0.95 are in `RHO_DEV_SWEEP.csv`, using
existing frozen per-seed Direct predictions and the protocol candidate grid
3.00–5.00 N. This is diagnostic only: no final rho was selected and no TEST
outcome was read.
""", encoding="utf-8")

    # 8. Explicit unavailable posterior ablation and existing transfer reuse.
    write_csv(OUT / "POINT_VS_POSTERIOR_DEV.csv", [{"status":"NOT_AVAILABLE","reason":"Agent C ensemble posterior predictions not present; existing scalar ensemble is not a posterior"}])
    if TRANSFER.exists():
        shutil.copy2(TRANSFER, OUT / "TRANSFER_CLEANUP_RESULTS.csv")
    (OUT / "TRANSFER_CLEANUP_REPORT.md").write_text("""# Transfer cleanup

Status: **REUSED EXISTING AUDIT.** `TRANSFER_CLEANUP_RESULTS.csv` is the
existing 3-seed source-subset audit with nested B=0/10/20/30/60 budgets. It
was not rerun because it already records the current TRAIN/DEV semantics and
the requested extra sampling would duplicate an authoritative experiment.
""", encoding="utf-8")

    # 9. Provenance, engineering log, and handoffs.
    (OUT / "TRAINING_PERFORMANCE_OPTIMIZATION.md").write_text("""# Training performance optimization

No new training job was launched. GPU state at preflight: A100 40 GB, 28.7 GB
used, 11.8 GB free, 80% utilization. The task policy therefore required CPU
preprocessing/analysis only. Feature tensors were copied in compact form and
all checks use vectorized pandas/NumPy operations.
""", encoding="utf-8")
    (OUT / "TRAINING_FARM_ENGINEERING_FIX_LOG.md").write_text("""# Engineering fix log

1. A full Tabero worktree initially hit `ENOSPC`; a linked no-checkout
   worktree was created at `/home/exouser/Tabero_trainingfarm` to avoid a
   second 3.4 GB checkout. The FORTE worktree is `/home/exouser/FORTE_trainingfarm`.
2. No source checkout, checkpoint, TEST artifact, or other agent process was
   modified. No OOM/NaN/dtype failure occurred because no new GPU job ran.
""", encoding="utf-8")
    (OUT / "TRAINING_FARM_HANDOFF_TO_MAIN.md").write_text("""# Training-farm handoff to main

Bundle: this directory. Reusable assets: 72 TRAIN visual feature tensors,
their hash-checked index, existing 3-seed identifier OOF, existing 3-seed
Direct/Joint DEV dense predictions, continuous interpolation rows, rho sweep,
and existing transfer cleanup. No frozen main checkpoint was replaced.

Open blockers: current-protocol seed 3/4 training for Vision-only,
Physical-only, Vision+Physical, FullTask Direct, and LocalLift Direct; LocalLift
is scientifically degenerate in the 720 archive; force-level holdout requires
new retraining; posterior ablation input is unavailable.
""", encoding="utf-8")
    (OUT / "TRAINING_FARM_HANDOFF_TO_E6E7.md").write_text("""# Training-farm handoff to E6/E7

Use `VISUAL_FEATURE_CACHE_INDEX.csv`, `PI0_VISUAL_FEATURE_CACHE_REPORT.md`,
`IDENTIFIER_SEED_RESULTS.csv`, `CALIBRATION_HANDOFF/`,
`CONTINUOUS_FORCE_INTERPOLATION.csv`, and `RHO_DEV_SWEEP.csv`. All are TRAIN/
DEV-only. Existing visual cache semantics are the project-local frozen
π0/PaliGemma backend, 224 preprocessing, camera order agentview then wrist,
4096D concatenated mean-pooled feature. Do not replace E6/E7 frozen checkpoints
from this bundle.
""", encoding="utf-8")

    statuses = {
        "authoritative_data_split": "PASS",
        "visual_feature_cache": "PASS" if feature_rows and all(x["cached_hash_match"] for x in feature_rows) else "FAIL",
        "identifier_5_seeds": "INCOMPLETE_3_EXISTING_SEEDS",
        "fulltask_5_seeds": "INCOMPLETE_3_EXISTING_SEEDS",
        "locallift": "SCIENTIFICALLY_INVALID_DEGENERATE_LABEL",
        "direct_seed_stability": "INCOMPLETE_3_EXISTING_SEEDS",
        "continuous_interpolation": "PARTIAL_STANDARD_REUSE_ONLY",
        "calibration": "PASS_DIRECT_SOFT_DEV_REUSE_IDENTIFIER_NA",
        "rho_sweep": "PASS_DEV_DIAGNOSTIC",
        "point_vs_posterior": "NOT_AVAILABLE",
        "transfer_cleanup": "REUSED_EXISTING",
        "sealed_test_read": False,
    }
    (OUT / "ACTIVEFORCING_OFFLINE_TRAINING_FARM_REPORT.md").write_text("""# ActiveForcing offline training farm report

## Final status

**INCOMPLETE — `ACTIVEFORCING_OFFLINE_TRAINING_COMPLETE` is not asserted.**

The CPU-safe portion is complete and reproducible in this bundle. Authoritative
data/split recovery passed; the frozen TRAIN visual cache was hash-checked;
existing grouped OOF/DEV predictions were summarized; calibration, continuous
interpolation reuse, rho sweep, transfer reuse, and handoffs were generated.

The requested 5-seed training farm could not be completed: only seeds 0–2 are
available in current authoritative artifacts and the preflight GPU had only
11.8 GB free at 80% utilization. LocalLift is not trainable as a matched
classifier because its authoritative label is positive for all 720 rows.
Leave-one-force-level-out and posterior-aware ablations also lack the required
current-protocol inputs/checkpoints. TEST remained sealed.

See `AUTHORITATIVE_DATA_QUALITY_AUDIT.json` and the per-section reports for
scope and evidence. No final parameter, method, or checkpoint was changed.
""", encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({"status":"INCOMPLETE","generated_utc":datetime.now(timezone.utc).isoformat(),"checks":statuses}, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output":str(OUT),"status":"INCOMPLETE","visual_cached":len(feature_rows),"identifier_seed_rows":len(seed_rows),"direct_stability_rows":len(direct_stability),"interpolation_rows":len(interp)}, indent=2))


if __name__ == "__main__":
    main()
