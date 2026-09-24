#!/usr/bin/env python3
"""CPU/offline E2 closure from frozen ActiveForcing artifacts.

This script deliberately reads only existing frozen CSV/JSON artifacts and
writes only into its own result directory.  It does not collect rollouts,
load/modify checkpoints, or touch any E5/Mass/Joint/Boundary/FORTE/E7 state.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


OUT = Path(__file__).resolve().parent
P4 = Path("/home/exouser/FORTE/activeforcing_probe_conditioned_wm_20260901_064627")
TRANSFER = Path("/home/exouser/FORTE/activeforcing_shared_physical_transfer_20260901_094722")
VIS = Path("/home/exouser/FORTE/structured_grasp_context_20260831")
PB = Path("/home/exouser/FORTE/activeforcing_full_claim_closure_20260902_134732/E6_E7")

SOURCES = {
    "p4_oof_predictions": P4 / "POOLED_OOF_PROBE_PREDICTIONS.csv",
    "p4_direct_archive": P4 / "PROBE_SCALAR_VS_RICH_DIRECT.csv",
    "p4_audit": P4 / "PROBE_WM_DATA_AUDIT.md",
    "physical_belief_predictions": PB / "PHYSICAL_BELIEF_PREDICTIONS.csv",
    "physical_belief_manifest": PB / "PHYSICAL_BELIEF_MANIFEST.json",
    "loto_predictions": TRANSFER / "UNIVERSAL_PROBE_LOTO_PREDICTIONS.csv",
    "loto_metrics": TRANSFER / "PHYSICS_ESTIMATOR_LOTO.csv",
    "downstream_transfer": TRANSFER / "NEW_TASK_FEWSHOT_TRANSFER_AGG.csv",
    "downstream_transfer_raw": TRANSFER / "NEW_TASK_FEWSHOT_TRANSFER.csv",
    "visual_root_metrics": VIS / "ROOT_HELDOUT_FOLD_METRICS.csv",
    "physical_context_audit": VIS / "ROOT_PHYSICAL_CONTEXT_AUDIT.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path}")
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        def clean(row: dict) -> dict:
            return {k: ("NA" if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()}
        w.writerows(clean(row) for row in rows)


def num(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v in (None, "", "NA", "nan", "NaN"):
        return math.nan
    return float(v)


def fmt(x: float | int | None, digits: int = 6) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def group(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict]]:
    out: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def band(v: float) -> str:
    return "LOW" if v < 0.4 else "MID" if v < 0.7 else "HIGH"


def pair_accuracy(rows: list[dict], y_key: str, p_key: str, keys: tuple[str, ...]) -> tuple[float, int]:
    good: list[bool] = []
    for g in group(rows, keys).values():
        for a, b in combinations(g, 2):
            ya, yb, pa, pb = num(a, y_key), num(b, y_key), num(a, p_key), num(b, p_key)
            if not all(math.isfinite(v) for v in (ya, yb, pa, pb)) or ya == yb:
                continue
            good.append(bool(np.sign(ya - yb) == np.sign(pa - pb)))
    return (float(np.mean(good)) if good else math.nan, len(good))


def point_metrics(rows: list[dict], y_key: str, p_key: str, pair_keys: tuple[str, ...]) -> dict:
    y = np.asarray([num(r, y_key) for r in rows], dtype=float)
    p = np.asarray([num(r, p_key) for r in rows], dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    e = p - y
    pair, pair_n = pair_accuracy([r for r in rows if math.isfinite(num(r, y_key)) and math.isfinite(num(r, p_key))], y_key, p_key, pair_keys)
    return {
        "n": int(len(y)),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e * e))),
        "median_AE": float(np.median(np.abs(e))),
        "bias": float(np.mean(e)),
        "Spearman": float(spearmanr(y, p).statistic),
        "pairwise_ranking_accuracy": pair,
        "pairwise_n": pair_n,
        "low_mid_high_accuracy": float(np.mean([band(a) == band(b) for a, b in zip(y, p)])),
    }


def task_macro(rows: list[dict], y_key: str, p_key: str) -> dict:
    by_task = group(rows, ("target_task",))
    ms = [point_metrics(g, y_key, p_key, ("root_id",)) for g in by_task.values()]
    return {
        "n": int(sum(m["n"] for m in ms)),
        "MAE": float(np.mean([m["MAE"] for m in ms])),
        "RMSE": float(np.mean([m["RMSE"] for m in ms])),
        "median_AE": float(np.mean([m["median_AE"] for m in ms])),
        "bias": float(np.mean([m["bias"] for m in ms])),
        "Spearman": float(np.mean([m["Spearman"] for m in ms])),
        "pairwise_ranking_accuracy": float(np.mean([m["pairwise_ranking_accuracy"] for m in ms])),
        "pairwise_n": int(sum(m["pairwise_n"] for m in ms)),
        "low_mid_high_accuracy": float(np.mean([m["low_mid_high_accuracy"] for m in ms])),
    }


def direct_metrics(rows: list[dict], method: str) -> dict:
    q = [r for r in rows if r["method"] == method]
    mean = lambda key: float(np.mean([num(r, key) for r in q]))
    return {
        "n": len(q),
        "unique_contexts": len({r["context_id"] for r in q}),
        "force_choice_agreement": mean("decision_agreement_with_GT"),
        "selected_force_N": mean("selected_force_N"),
        "under_force_rate": mean("under_force"),
        "mean_force_N": mean("selected_force_N"),
        "realized_utility": mean("realized_utility"),
        "downstream_SR": mean("success"),
    }


def downstream_metrics(rows: list[dict], estimator: str, budget: str = "60") -> dict:
    q = [r for r in rows if r["physics_estimator"] == estimator and r["budget"] == budget]
    weights = np.asarray([num(r, "episodes") for r in q], dtype=float)
    def mean(key: str) -> float:
        return float(np.average([num(r, key) for r in q], weights=weights))
    return {
        "n": int(weights.sum()),
        "force_choice_agreement": mean("decision_agreement_GT"),
        "selected_force_N": mean("mean_force"),
        "under_force_rate": mean("underforce"),
        "mean_force_N": mean("mean_force"),
        "realized_utility": mean("utility"),
        "downstream_SR": mean("sr"),
        "excess_force_N": mean("excess_force"),
    }


def visual_metrics(rows: list[dict], model: str) -> dict:
    q = [r for r in rows if r["model"] == model]
    mean = lambda key: float(np.mean([num(r, key) for r in q]))
    return {
        "n": int(sum(int(num(r, "heldout_contexts")) for r in q)),
        "n_root": int(len({root for r in q for root in json.loads(r["heldout_root_ids"])})),
        "NLL": mean("NLL"),
        "Brier": mean("Brier"),
        "frontier_MAE_N": mean("frontier_MAE_N"),
        "frontier_under_force_rate": mean("under_force_rate"),
    }


def uncertainty_metrics(rows: list[dict], manifest: dict) -> dict:
    q = [r for r in rows if r["split"] == "DEV"]
    mu = np.asarray([num(r, "ensemble_mean") for r in q])
    y = np.asarray([num(r, "friction_gt") for r in q])
    var = np.asarray([num(r, "total_variance") for r in q])
    err = mu - y
    sigma = np.sqrt(var)
    nll = float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * err * err / var))
    z68, z90, z95 = 1.0, 1.6448536269514722, 1.959963984540054
    cov = {"68": float(np.mean(np.abs(err) <= z68 * sigma)),
           "90": float(np.mean(np.abs(err) <= z90 * sigma)),
           "95": float(np.mean(np.abs(err) <= z95 * sigma))}
    order = np.argsort(sigma, kind="stable")
    risk = np.cumsum(np.abs(err)[order]) / np.arange(1, len(err) + 1)
    aurc = float(np.mean(risk))
    pacc, pn = pair_accuracy(q, "friction_gt", "ensemble_mean", ("root_id",))
    return {
        "n": len(q),
        "n_roots": len({r["root_id"] for r in q}),
        "member_count": int(manifest["member_count"]),
        "predictive_NLL": nll,
        "Brier": math.nan,
        "coverage_68": cov["68"],
        "coverage_90": cov["90"],
        "coverage_95": cov["95"],
        "uncertainty_error_Spearman": float(spearmanr(sigma, np.abs(err)).statistic),
        "AURC": aurc,
        "pairwise_ranking_accuracy": pacc,
        "pairwise_n": pn,
        "low_mid_high_accuracy": float(np.mean([band(a) == band(b) for a, b in zip(y, mu)])),
        "stored_coverage_90_matches": bool(abs(cov["90"] - np.mean([num(r, "covered_90") for r in q])) < 1e-12),
        "interval_scale_manifest": float(manifest["interval_scale"]),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_manifest() -> dict:
    out = {}
    for name, p in SOURCES.items():
        out[name] = {"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None,
                     "sha256": sha256(p) if p.exists() else None}
    return out


def base_row(variant: str, estimator: str, scope: str, priority: str, n: int, roots: int,
             seeds: str, target: str, status: str, source: str, caveat: str, aggregation: str) -> dict:
    return {
        "variant": variant, "canonical_estimator": estimator, "evaluation_scope": scope,
        "scope_priority": priority, "metric_aggregation": aggregation, "n_contexts": n,
        "n_root_families": roots, "n_members_or_seeds": seeds, "identification_target": target,
        "MAE": math.nan, "RMSE": math.nan, "median_AE": math.nan, "bias": math.nan,
        "Spearman": math.nan, "pairwise_ranking_accuracy": math.nan, "pairwise_n": math.nan,
        "low_mid_high_accuracy": math.nan, "predictive_NLL": math.nan, "Brier": math.nan,
        "coverage_68": math.nan, "coverage_90": math.nan, "coverage_95": math.nan,
        "uncertainty_error_Spearman": math.nan, "AURC": math.nan,
        "force_choice_agreement": math.nan, "selected_force_N": math.nan,
        "under_force_rate": math.nan, "mean_force_N": math.nan, "realized_utility": math.nan,
        "archived_downstream_SR_proxy": math.nan, "downstream_proxy_scope": "NA",
        "downstream_proxy_n": math.nan, "downstream_excess_force_N": math.nan,
        "visual_frontier_MAE_N": math.nan, "visual_frontier_under_force_rate": math.nan,
        "status": status, "source_artifact": source, "caveat": caveat,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in SOURCES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing frozen sources: " + "; ".join(missing))

    p4 = read_csv(SOURCES["p4_oof_predictions"])
    p4_point = [r for r in p4 if r["seed"] == "ENSEMBLE_MEAN"]
    p4_m = point_metrics(p4_point, "mu_GT", "mu_hat", ("root_id",))
    p4_raw = point_metrics([r for r in p4 if r["seed"] != "ENSEMBLE_MEAN"], "mu_GT", "mu_hat", ("root_id",))
    direct = read_csv(SOURCES["p4_direct_archive"])
    pb = read_csv(SOURCES["physical_belief_predictions"])
    manifest = json.loads(SOURCES["physical_belief_manifest"].read_text(encoding="utf-8"))
    pb_dev = [r for r in pb if r["split"] == "DEV"]
    pb_m = point_metrics(pb_dev, "friction_gt", "ensemble_mean", ("root_id",))
    unc = uncertainty_metrics(pb, manifest)
    loto = read_csv(SOURCES["loto_predictions"])
    explicit = [r for r in loto if r["method"] == "ExplicitSysID"]
    explicit_pooled = point_metrics(explicit, "mu_GT", "mu_hat", ("root_id",))
    explicit_macro = task_macro(explicit, "mu_GT", "mu_hat")
    loto_source = read_csv(SOURCES["loto_metrics"])
    explicit_registered = next(r for r in loto_source if r["target_task"] == "MACRO" and r["method"] == "ExplicitSysID")
    transfer = read_csv(SOURCES["downstream_transfer_raw"])
    visual = read_csv(SOURCES["visual_root_metrics"])
    d_no = direct_metrics(direct, "NoProbe-Direct")
    d_probe = direct_metrics(direct, "ProbeScalar-Direct")
    ds_exp = downstream_metrics(transfer, "ExplicitSysID")
    ds_gt = downstream_metrics(transfer, "GT")
    v_full = visual_metrics(visual, "Full Visual")
    v_joint = visual_metrics(visual, "Visual Joint")

    rows: list[dict] = []
    r = base_row("PRIOR", "NoProbe-Direct", "E1_720_ARCHIVE_PAIRED", "secondary", d_no["unique_contexts"], "NA", "1", "none", "COMPLETE_BASELINE",
                 str(SOURCES["p4_direct_archive"]), "Decision baseline only; no physical identifier.", "episode-pooled")
    r.update({"force_choice_agreement": d_no["force_choice_agreement"], "selected_force_N": d_no["selected_force_N"],
              "under_force_rate": d_no["under_force_rate"], "mean_force_N": d_no["mean_force_N"],
              "realized_utility": d_no["realized_utility"], "archived_downstream_SR_proxy": d_no["downstream_SR"],
              "downstream_proxy_scope": "E1_720_ARCHIVE_PAIRED", "downstream_proxy_n": d_no["n"]})
    rows.append(r)

    r = base_row("VISION_ONLY", "Full Visual", "same-task_ROOT_HELDOUT", "primary", v_full["n"], v_full["n_root"], "1", "feasibility/frontier",
                 "DIAGNOSTIC_ONLY_NOT_MU", str(SOURCES["visual_root_metrics"]),
                 "Binary feasibility/frontier model, not a friction μ estimator; no matched Direct+candidate+EU policy artifact.", "fold-mean")
    r.update({"predictive_NLL": v_full["NLL"], "Brier": v_full["Brier"], "visual_frontier_MAE_N": v_full["frontier_MAE_N"],
              "visual_frontier_under_force_rate": v_full["frontier_under_force_rate"]})
    rows.append(r)

    r = base_row("PHYSICAL_HISTORY_ONLY", "LearnedProbe_P4B_OOF", "same-task_ROOT_HELDOUT_OOF", "primary", p4_m["n"],
                 len({x["root_id"] for x in p4_point}), "3", "friction", "COMPLETE", str(SOURCES["p4_oof_predictions"]),
                 "Three trained fold/seed estimators; point metrics use the derived ENSEMBLE_MEAN row once per context.", "context-pooled")
    r.update({k: p4_m[k] for k in ("MAE", "RMSE", "median_AE", "bias", "Spearman", "pairwise_ranking_accuracy", "pairwise_n", "low_mid_high_accuracy")})
    r.update({"force_choice_agreement": d_probe["force_choice_agreement"], "selected_force_N": d_probe["selected_force_N"],
              "under_force_rate": d_probe["under_force_rate"], "mean_force_N": d_probe["mean_force_N"],
              "realized_utility": d_probe["realized_utility"], "archived_downstream_SR_proxy": d_probe["downstream_SR"],
              "downstream_proxy_scope": "E1_720_ARCHIVE_PAIRED_ProbeScalar-Direct", "downstream_proxy_n": d_probe["n"]})
    rows.append(r)

    r = base_row("PHYSICAL_HISTORY_ONLY_3MEM", "PhysicalBelief_3member", "DEV_ROOT_HELDOUT", "primary", pb_m["n"], unc["n_roots"],
                 str(unc["member_count"]), "friction+predictive_distribution", "DIAGNOSTIC_NOT_CALIBRATED", str(SOURCES["physical_belief_predictions"]),
                 "DEV-only diagnostic; no TEST rows loaded. Coverage/spread are reported descriptively, not as a calibrated-uncertainty claim.", "context-pooled")
    r.update({k: pb_m[k] for k in ("MAE", "RMSE", "median_AE", "bias", "Spearman", "pairwise_ranking_accuracy", "pairwise_n", "low_mid_high_accuracy")})
    r.update({k: unc[k] for k in ("predictive_NLL", "Brier", "coverage_68", "coverage_90", "coverage_95", "uncertainty_error_Spearman", "AURC")})
    rows.append(r)

    r = base_row("VISION_PLUS_PHYSICAL", "Visual_Joint", "same-task_ROOT_HELDOUT", "primary", v_joint["n"], v_joint["n_root"], "1", "feasibility/frontier",
                 "DIAGNOSTIC_ONLY_NOT_MU", str(SOURCES["visual_root_metrics"]),
                 "Closest visual-plus-physical feasibility/frontier diagnostic; not a μ estimator and no matched Direct+candidate+EU policy artifact.", "fold-mean")
    r.update({"predictive_NLL": v_joint["NLL"], "Brier": v_joint["Brier"], "visual_frontier_MAE_N": v_joint["frontier_MAE_N"],
              "visual_frontier_under_force_rate": v_joint["frontier_under_force_rate"]})
    rows.append(r)

    r = base_row("EXPLICIT_SYSID", "ExplicitSysID", "LOTO_TASK_OBJECT_HELDOUT", "secondary", explicit_macro["n"],
                 len({x["root_id"] for x in explicit}), "1", "friction", "COMPLETE_SECONDARY", str(SOURCES["loto_metrics"]),
                 "LOTO holds out task and object family together; cannot isolate semantic task transfer from object transfer. Main metrics use the frozen source task-balanced macro definition; raw pooled audit is in report.", "task-balanced-macro")
    r.update({k: explicit_macro[k] for k in ("MAE", "RMSE", "median_AE", "bias", "Spearman", "pairwise_ranking_accuracy", "pairwise_n", "low_mid_high_accuracy")})
    r.update({"force_choice_agreement": ds_exp["force_choice_agreement"], "selected_force_N": ds_exp["selected_force_N"],
              "under_force_rate": ds_exp["under_force_rate"], "mean_force_N": ds_exp["mean_force_N"],
              "realized_utility": ds_exp["realized_utility"], "archived_downstream_SR_proxy": ds_exp["downstream_SR"],
              "downstream_proxy_scope": "LOTO_TARGET_ADAPTATION_ARCHIVE_B60", "downstream_proxy_n": ds_exp["n"],
              "downstream_excess_force_N": ds_exp["excess_force_N"]})
    rows.append(r)

    r = base_row("PHYS2REAL_STYLE_FUSION", "NOT_FOUND", "not_available", "primary", "NA", "NA", "NA", "friction",
                 "SCIENTIFIC_BLOCKER", "NA", "No frozen visual-prior + interaction-uncertainty fusion checkpoint, prediction file, or matched downstream policy artifact was found; no claim is made.", "NA")
    rows.append(r)

    r = base_row("GT_PHYSICS_DIAGNOSTIC", "GT", "LOTO_TARGET_ADAPTATION_ARCHIVE", "secondary", explicit_macro["n"],
                 len({x["root_id"] for x in explicit}), "1", "privileged_GT_friction", "ORACLE_DIAGNOSTIC", str(SOURCES["loto_predictions"]),
                 "Privileged ground truth; diagnostic only, not deployable identification or uncertainty.", "context-pooled")
    r.update({"MAE": 0.0, "RMSE": 0.0, "median_AE": 0.0, "bias": 0.0, "Spearman": 1.0,
              "pairwise_ranking_accuracy": 1.0, "pairwise_n": explicit_macro["pairwise_n"], "low_mid_high_accuracy": 1.0,
              "force_choice_agreement": ds_gt["force_choice_agreement"], "selected_force_N": ds_gt["selected_force_N"],
              "under_force_rate": ds_gt["under_force_rate"], "mean_force_N": ds_gt["mean_force_N"],
              "realized_utility": ds_gt["realized_utility"], "archived_downstream_SR_proxy": ds_gt["downstream_SR"],
              "downstream_proxy_scope": "LOTO_TARGET_ADAPTATION_ARCHIVE_B60", "downstream_proxy_n": ds_gt["n"],
              "downstream_excess_force_N": ds_gt["excess_force_N"]})
    rows.append(r)

    table_path = OUT / "TABLE_E2_PHYSICAL_IDENTIFICATION.csv"
    write_csv(table_path, rows)

    # Small task-level audit makes the macro-vs-pooled choice inspectable.
    task_rows = []
    for task, g in sorted(group(explicit, ("target_task",)).items()):
        m = point_metrics(g, "mu_GT", "mu_hat", ("root_id",))
        task_rows.append({"method": "ExplicitSysID", "target_task": task[0], **m})
    write_csv(OUT / "E2_EXPLICIT_SYSID_TASK_AUDIT.csv", task_rows)

    checks = {
        "disk_free_GiB_at_generation": shutil.disk_usage(OUT).free / (1024 ** 3),
        "no_rollout_collection": True,
        "no_checkpoint_mutation": True,
        "p4_rows": len(p4), "p4_ensemble_mean_rows": len(p4_point), "p4_member_rows": len(p4) - len(p4_point),
        "p4_unique_contexts": len({r["context_id"] for r in p4_point}), "p4_unique_roots": len({r["root_id"] for r in p4_point}),
        "p4_rows_per_context": sorted(set(Counter(r["context_id"] for r in p4).values())),
        "p4_point_metrics": p4_m, "p4_member_only_metrics": p4_raw,
        "p4_pairwise_definition": "all within-root low/mid/high pairs; one ensemble mean per context",
        "physical_belief_split_counts": dict(Counter(r["split"] for r in pb)),
        "physical_belief_dev_metrics": pb_m, "physical_belief_uncertainty": unc,
        "loto_method_counts": dict(Counter(r["method"] for r in loto)),
        "explicit_sysid_raw_pooled": explicit_pooled, "explicit_sysid_source_registered_macro": {k: explicit_registered[k] for k in ("MAE", "RMSE", "bias", "Spearman", "pair_ranking")},
        "explicit_sysid_task_macro_recomputed": explicit_macro,
        "direct_method_counts": dict(Counter(r["method"] for r in direct)),
        "visual_model_counts": dict(Counter(r["model"] for r in visual)),
        "visual_full_metrics": v_full, "visual_joint_metrics": v_joint,
        "phys2real_style_fusion_found": False,
        "test_contexts_loaded_in_belief_manifest": manifest.get("test_contexts_loaded"),
        "warning_gate_GiB": 15.0, "hard_stop_gate_GiB": 10.0,
    }
    (OUT / "VALIDATION_AUDIT.json").write_text(json.dumps(checks, indent=2, allow_nan=True), encoding="utf-8")

    provenance = {
        "artifact": "E2_PHYSICAL_IDENTIFICATION_FINAL",
        "status": "E2_PHYSICAL_IDENTIFICATION_COMPLETE_SCIENTIFIC_NEGATIVE_FOR_UNAVAILABLE_FUSION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {"mode": "CPU/offline", "rollout_collection": False, "gpu_training": False,
                       "e5_touched": False, "mass_joint_boundary_forte_tabero_e7_touched": False,
                       "output_root": str(OUT), "script": str(Path(__file__).resolve())},
        "sources": source_manifest(),
        "methodology": {"primary_split": "same-task root-heldout where frozen artifact exists",
                         "secondary_split": "LOTO task/object heldout; task and object family are confounded",
                         "p4_point_unit": "ENSEMBLE_MEAN row, one row per context; three trained seed members",
                         "explicit_sysid_unit": "frozen source task-balanced macro; raw pooled alternative retained in VALIDATION_AUDIT.json",
                         "uncertainty": "Gaussian NLL and empirical coverage from archived total variance; Brier NA without archived band probabilities; AURC is mean cumulative absolute-error risk after sorting by total sigma"},
        "outputs": {"table": str(table_path), "report": str(OUT / "E2_PHYSICAL_IDENTIFICATION_FINAL_REPORT.md"),
                    "status": str(OUT / "STATUS.md"), "validation": str(OUT / "VALIDATION_AUDIT.json")},
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, allow_nan=True), encoding="utf-8")

    report = f"""# E2 Physical Identification Final

**Status: E2_PHYSICAL_IDENTIFICATION_COMPLETE_SCIENTIFIC_NEGATIVE_FOR_UNAVAILABLE_FUSION**  
Generated: `{provenance['generated_at_utc']}`. This is a CPU/offline closure from frozen artifacts. No rollout was recollected and no checkpoint, E5 process, scheduler, or protected campaign was touched.

## Bottom line

The strongest valid primary result is the root-heldout P4-B physical-history estimator. At the context grain (72 contexts, 24 root families, 4 tasks, 720 force branches), the frozen ensemble-mean point estimate obtains **MAE {fmt(p4_m['MAE'],4)}**, **RMSE {fmt(p4_m['RMSE'],4)}**, median AE **{fmt(p4_m['median_AE'],4)}**, bias **{fmt(p4_m['bias'],4)}**, Spearman **{fmt(p4_m['Spearman'],4)}**, pairwise ranking **{fmt(p4_m['pairwise_ranking_accuracy'],4)}** over **{p4_m['pairwise_n']}** within-root friction pairs, and LOW/MID/HIGH accuracy **{fmt(p4_m['low_mid_high_accuracy'],4)}**.

The independent three-member PhysicalBelief diagnostic is weaker on its DEV root-heldout split (24 contexts, 8 roots): MAE **{fmt(pb_m['MAE'],4)}**, RMSE **{fmt(pb_m['RMSE'],4)}**, Spearman **{fmt(pb_m['Spearman'],4)}**, pairwise ranking **{fmt(pb_m['pairwise_ranking_accuracy'],4)}**. Its nominal 68/90/95% empirical coverages are **{fmt(unc['coverage_68'],4)} / {fmt(unc['coverage_90'],4)} / {fmt(unc['coverage_95'],4)}**, with Gaussian NLL **{fmt(unc['predictive_NLL'],4)}**, uncertainty/error Spearman **{fmt(unc['uncertainty_error_Spearman'],4)}**, and AURC **{fmt(unc['AURC'],4)}**. Brier is **NA** because no archived calibrated band-probability vector exists. These are diagnostic uncertainty statistics, not a calibration claim.

## Paper-ready comparison

The CSV is the machine-readable table: [TABLE_E2_PHYSICAL_IDENTIFICATION.csv](./TABLE_E2_PHYSICAL_IDENTIFICATION.csv). `MAE`–`low_mid_high_accuracy` use the `metric_aggregation` column. Downstream columns are archived same-controller proxies only when a matched Direct + candidate set + Expected Utility artifact exists.

| variant | split / priority | μ metrics | uncertainty / visual diagnostic | downstream proxy |
|---|---|---|---|---|
| PRIOR | E1 720 archive / secondary | not an identifier | — | SR {fmt(d_no['downstream_SR'],4)}, agreement {fmt(d_no['force_choice_agreement'],4)}, force {fmt(d_no['mean_force_N'],3)} N, under {fmt(d_no['under_force_rate'],4)}, utility {fmt(d_no['realized_utility'],4)} |
| VISION ONLY | root-heldout / primary | not a μ estimator | NLL {fmt(v_full['NLL'],4)}, Brier {fmt(v_full['Brier'],4)}, frontier MAE {fmt(v_full['frontier_MAE_N'],4)} N | unavailable |
| PHYSICAL HISTORY ONLY | P4-B root-heldout / primary | MAE {fmt(p4_m['MAE'],4)}, RMSE {fmt(p4_m['RMSE'],4)}, rank {fmt(p4_m['pairwise_ranking_accuracy'],4)} | point OOF; no predictive band | SR {fmt(d_probe['downstream_SR'],4)}, agreement {fmt(d_probe['force_choice_agreement'],4)}, force {fmt(d_probe['mean_force_N'],3)} N, under {fmt(d_probe['under_force_rate'],4)}, utility {fmt(d_probe['realized_utility'],4)} |
| PHYSICAL HISTORY ONLY (3-member) | DEV root-heldout / primary | MAE {fmt(pb_m['MAE'],4)}, RMSE {fmt(pb_m['RMSE'],4)}, rank {fmt(pb_m['pairwise_ranking_accuracy'],4)} | coverage 68/90/95 = {fmt(unc['coverage_68'],4)}/{fmt(unc['coverage_90'],4)}/{fmt(unc['coverage_95'],4)}; AURC {fmt(unc['AURC'],4)} | unavailable |
| VISION + PHYSICAL | root-heldout / primary | not a μ estimator | NLL {fmt(v_joint['NLL'],4)}, Brier {fmt(v_joint['Brier'],4)}, frontier MAE {fmt(v_joint['frontier_MAE_N'],4)} N | unavailable |
| EXPLICIT SYSID | LOTO task/object-heldout / secondary | MAE {fmt(explicit_macro['MAE'],4)}, RMSE {fmt(explicit_macro['RMSE'],4)}, median AE {fmt(explicit_macro['median_AE'],4)}, rank {fmt(explicit_macro['pairwise_ranking_accuracy'],4)} | point only; no validated predictive uncertainty | B60 SR {fmt(ds_exp['downstream_SR'],4)}, agreement {fmt(ds_exp['force_choice_agreement'],4)}, force {fmt(ds_exp['mean_force_N'],3)} N, under {fmt(ds_exp['under_force_rate'],4)}, utility {fmt(ds_exp['realized_utility'],4)} |
| Phys2Real-style fusion | not available / primary | blocked | no frozen fusion artifact | blocked |
| GT physics | privileged LOTO diagnostic / secondary | exact GT diagnostic | oracle, non-deployable | B60 SR {fmt(ds_gt['downstream_SR'],4)}, agreement {fmt(ds_gt['force_choice_agreement'],4)}, force {fmt(ds_gt['mean_force_N'],3)} N, under {fmt(ds_gt['under_force_rate'],4)}, utility {fmt(ds_gt['realized_utility'],4)} |

## Validation findings

- P4-B contains 216 seed-level rows plus 72 derived `ENSEMBLE_MEAN` rows. The primary point metrics above use the 72 derived rows once per context; the previous lane artifact mixed context-level MAE with an all-row RMSE. The corrected context-pooled RMSE is **{fmt(p4_m['RMSE'],4)}**; the member-only 216-row RMSE is **{fmt(p4_raw['RMSE'],4)}**, while including the derived rows gives **{fmt(point_metrics(p4, 'mu_GT', 'mu_hat', ('root_id',))['RMSE'],4)}**.
- Explicit SysID's frozen source table reports task-balanced macro metrics: MAE **{explicit_registered['MAE']}**, RMSE **{explicit_registered['RMSE']}**, Spearman **{explicit_registered['Spearman']}**, pair ranking **{explicit_registered['pair_ranking']}**. The raw pooled audit is MAE **{fmt(explicit_pooled['MAE'],4)}**, RMSE **{fmt(explicit_pooled['RMSE'],4)}**, median AE **{fmt(explicit_pooled['median_AE'],4)}**, Spearman **{fmt(explicit_pooled['Spearman'],4)}**; both are preserved in `VALIDATION_AUDIT.json`, and the table labels the registered macro choice.
- Vision-only and vision-plus-physical rows are valid root-heldout feasibility/frontier diagnostics. Their NLL/Brier/frontier scores must not be relabeled as friction μ identification, and no same-policy downstream utility run is present.
- LOTO changes task and object family together. It is secondary evidence and cannot by itself establish pure semantic task transfer.
- The PhysicalBelief manifest reports `test_contexts_loaded = {manifest.get('test_contexts_loaded')}`; no original TEST rows were used. The manifest's train-fitted interval scale is recorded in the provenance, but the archived row intervals use the source `total_variance` convention; no calibration claim is made.
- No frozen Phys2Real-style visual-prior + interaction-uncertainty fusion checkpoint/prediction/downstream artifact was found. This is a scientific blocker for that requested comparison, not a reason to invent a value.

## Reproducibility and exact artifacts

- Runner: [run_e2_final.py](./run_e2_final.py)
- Table: [TABLE_E2_PHYSICAL_IDENTIFICATION.csv](./TABLE_E2_PHYSICAL_IDENTIFICATION.csv)
- Task audit: [E2_EXPLICIT_SYSID_TASK_AUDIT.csv](./E2_EXPLICIT_SYSID_TASK_AUDIT.csv)
- Validation: [VALIDATION_AUDIT.json](./VALIDATION_AUDIT.json)
- Provenance: [PROVENANCE.json](./PROVENANCE.json)
- Status: [STATUS.md](./STATUS.md)

Overall E2 status is complete for the strongest valid existing evidence, with a scientific negative/blocked result for commensurate visual μ comparison and Phys2Real-style fusion.
"""
    (OUT / "E2_PHYSICAL_IDENTIFICATION_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    status = f"""# E2 status

**Overall:** `E2_PHYSICAL_IDENTIFICATION_COMPLETE_SCIENTIFIC_NEGATIVE_FOR_UNAVAILABLE_FUSION`

Generated `{provenance['generated_at_utc']}` in independent result root `{OUT}`.

- Primary root-heldout physical-history result: P4-B, 72 contexts / 24 roots / 720 force branches; MAE `{fmt(p4_m['MAE'],4)}`, RMSE `{fmt(p4_m['RMSE'],4)}`, pairwise ranking `{fmt(p4_m['pairwise_ranking_accuracy'],4)}`.
- Secondary LOTO explicit SysID: 144 contexts / 48 roots; task/object confounded; registered task-balanced MAE `{explicit_registered['MAE']}`, RMSE `{explicit_registered['RMSE']}`.
- Uncertainty: 3-member DEV diagnostic only; coverage 68/90/95 `{fmt(unc['coverage_68'],4)}/{fmt(unc['coverage_90'],4)}/{fmt(unc['coverage_95'],4)}`; not calibrated.
- Visual-only and visual-plus-physical: feasibility/frontier diagnostics only; no μ or matched downstream claim.
- Phys2Real-style fusion: no frozen artifact; blocked, no claim.
- Rollout collection: none. E5/Mass/Joint/Boundary/FORTE/Tabero/E7: untouched.
- Disk free at generation: `{checks['disk_free_GiB_at_generation']:.2f} GiB`; warning gate 15 GiB, hard-stop gate 10 GiB.

Exact artifacts are listed in `PROVENANCE.json`.
"""
    (OUT / "STATUS.md").write_text(status, encoding="utf-8")
    output_paths = {
        "table": OUT / "TABLE_E2_PHYSICAL_IDENTIFICATION.csv",
        "report": OUT / "E2_PHYSICAL_IDENTIFICATION_FINAL_REPORT.md",
        "status": OUT / "STATUS.md",
        "validation": OUT / "VALIDATION_AUDIT.json",
        "task_audit": OUT / "E2_EXPLICIT_SYSID_TASK_AUDIT.csv",
        "runner": Path(__file__).resolve(),
    }
    provenance["outputs_sha256"] = {name: sha256(path) for name, path in output_paths.items()}
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
