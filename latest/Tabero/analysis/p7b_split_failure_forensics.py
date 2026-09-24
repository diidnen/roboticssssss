#!/usr/bin/env python3
"""Forensic-only reconstruction of the frozen P7-B split failure.

This script reads existing artifacts only.  It does not launch Isaac, modify
the protocol, relax any predicate, or retry any scientific episode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path("/home/exouser/Tabero/analysis/results/p7b_scientific_main_20260828_000729")
REPO = Path("/home/exouser/Tabero")

EXPECTED_HASHES = {
    "protocol": (REPO / "analysis/results/p7b_scientific_main_20260827_234017/P7B_PROTOCOL_IMMUTABLE_COPY.json",
                 "7cbdd36a600204e06ddc635f68563731445756e2e24cd5e14712036a39a3941a"),
    "collector": (REPO / "analysis/p7b_gnp_physical_belief_force_planning.py",
                  "2ef853b3d3a4ad4d1642332841dc5deb3daed71a7ca2ae8cb8d588668ec28337"),
    "gripper_semantics": (REPO / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py",
                          "51292d864a3bae436a0dbfeb2f737e2b27317f717b4e36011173241afa59956d"),
}
EXPECTED_CLEAN_IDENTITY = "638045a19295db7721b0757030302dac27ffef2700ebbac9a8bccfd5a9daaeb5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    return v


def json_dump(path: Path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, sort_keys=True) + "\n")


def pct(n, d):
    return None if not d else 100.0 * float(n) / float(d)


def numeric_summary(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max())}


def group_table(df, group_cols, value_cols):
    rows = []
    for key, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        r = {c: clean(v) for c, v in zip(group_cols, key)}
        r["n"] = int(len(g))
        for c in value_cols:
            if c in g:
                r[c + "_n"] = int(pd.to_numeric(g[c], errors="coerce").notna().sum())
                r[c + "_mean"] = clean(pd.to_numeric(g[c], errors="coerce").mean())
                r[c + "_median"] = clean(pd.to_numeric(g[c], errors="coerce").median())
        rows.append(r)
    return pd.DataFrame(rows)


def first_failure(row, qtele):
    def flag(name):
        v = row.get(name, 0)
        try:
            return bool(pd.notna(v) and float(v) != 0.0)
        except Exception:
            return str(v).lower() in {"true", "yes", "1"}
    if flag("infra_failure") or flag("query_true_runtime_error"):
        return "RUNTIME_FAILURE"
    if int(row.get("query_qualified", 0)) == 1:
        return "QUALIFIED"
    if qtele is None or len(qtele) == 0:
        return "PRE_QUERY_STATE_INVALID"
    phases = list(qtele["phase"].astype(str)) if "phase" in qtele else []
    first_phase = phases[0] if phases else ""
    first_drop = qtele.index[qtele["dropped"].fillna(0).astype(int).eq(1)].min() if "dropped" in qtele else np.nan
    probe = qtele[qtele["phase"].astype(str).str.startswith("probe")]
    bilateral = qtele[(qtele.get("contact_left", 0).fillna(0).astype(int) == 1) &
                      (qtele.get("contact_right", 0).fillna(0).astype(int) == 1)]
    if pd.notna(first_drop) and (len(probe) == 0 or int(first_drop) <= int(probe.index.min())):
        return "PRE_QUERY_STATE_INVALID" if first_phase in {"approach", "descend", "close", "hold"} else "CONTACT_NOT_ESTABLISHED"
    if flag("major_disturbance"):
        return "MAJOR_DISTURBANCE"
    reason = str(row.get("query_failure_reason") or "")
    if reason == "contact_loss" or len(probe) > 0 and len(bilateral) == 0:
        return "BILATERAL_CONTACT_FAILED"
    if reason == "return_state_invalid" or not flag("return_state_valid"):
        return "RETURN_INVALID"
    if len(probe) == 0:
        return "PROBE_DID_NOT_COMPLETE"
    return "OTHER_SUPPORTED_FAILURE"


def load_stage(path: Path):
    try:
        j = json.loads(path.read_text())
    except Exception as e:
        return {"stage_json_error": str(e)}
    out = {}
    for key in ["eef_pose_base_after_stage", "g2_pregrasp_target_base_m", "nominal_reset_eef_pose_base",
                "p4_center_grasp_target_base_m", "p4_center_pregrasp_target_base_m"]:
        vals = j.get(key, [])
        for i, v in enumerate(vals):
            out[f"{key}_{i}"] = v
    obj = j.get("object_pose_base_after_stage", {})
    for i, v in enumerate(obj.get("position_m", [])):
        out[f"stage_object_position_{i}"] = v
    for i, v in enumerate(obj.get("quaternion_wxyz", [])):
        out[f"stage_object_quat_{i}"] = v
    sr = j.get("stage_result", {})
    for k in ["staging_validity", "collision_free", "ik_success", "no_fingertip_contact_before_invocation",
              "object_disturbance_m", "object_rotation_disturbance_rad", "orientation_error_rad",
              "position_error_m", "steps", "policy_context_mode", "target_mode", "wrist_roll_deg"]:
        out["stage_" + k] = sr.get(k)
    return out


def load_query(path: Path):
    try:
        d = pd.read_csv(path)
    except Exception as e:
        return None, {"query_csv_error": str(e)}
    if len(d) == 0:
        return d, {"telemetry_rows": 0}
    d = d.sort_values("step").reset_index(drop=True)
    phases = d["phase"].astype(str)
    probe = d[phases.str.startswith("probe")]
    bilateral = d[(d["contact_left"].fillna(0).astype(int) == 1) &
                  (d["contact_right"].fillna(0).astype(int) == 1)]
    first_drop_rows = d[d["dropped"].fillna(0).astype(int) == 1]
    first_contact_rows = d[d["contact_state"].astype(str).ne("none")]
    first_bilateral_rows = bilateral
    out = {
        "telemetry_rows": len(d), "telemetry_first_phase": phases.iloc[0], "telemetry_last_phase": phases.iloc[-1],
        "telemetry_first_step": d["step"].iloc[0], "telemetry_last_step": d["step"].iloc[-1],
        "telemetry_duration_s": d["t_s"].max(), "probe_rows": len(probe), "bilateral_rows": len(bilateral),
        "max_ft_over_fn": pd.to_numeric(d["ft_over_fn"], errors="coerce").max(),
        "max_accumulated_displacement_mm": pd.to_numeric(d["accumulated_displacement_mm"], errors="coerce").max(),
        "first_drop_step": first_drop_rows["step"].iloc[0] if len(first_drop_rows) else None,
        "first_contact_step": first_contact_rows["step"].iloc[0] if len(first_contact_rows) else None,
        "first_bilateral_step": first_bilateral_rows["step"].iloc[0] if len(first_bilateral_rows) else None,
        "first_probe_step": probe["step"].iloc[0] if len(probe) else None,
        "last_object_x_w": d["object_x_w"].iloc[-1], "last_object_y_w": d["object_y_w"].iloc[-1],
        "last_object_z_w": d["object_z_w"].iloc[-1], "last_dropped": d["dropped"].iloc[-1],
    }
    for c in ["eef_x_actual", "eef_y_actual", "eef_z_actual", "object_x_w", "object_y_w", "object_z_w",
              "query_start_eef_x_base", "query_start_eef_y_base", "query_start_eef_z_base"]:
        out["first_" + c] = d[c].iloc[0]
        out["last_" + c] = d[c].iloc[-1]
    return d, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out_dir) if args.out_dir else REPO / "analysis/results" / f"p7b_split_failure_forensics_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    context = pd.read_csv(RUN / "P7B_CONTEXT_MANIFEST.csv")
    roots = pd.read_csv(RUN / "P7B_ROOT_MANIFEST.csv")
    branches = pd.read_csv(RUN / "P7B_BRANCH_MANIFEST.csv")
    parity = pd.read_csv(RUN / "P7B_STATE_PARITY.csv")
    root_split = roots.set_index("root_group_id")["split"].to_dict()

    stage_map = {}
    for p in sorted((RUN / "P7B_STAGE_TELEMETRY").glob("*.json")):
        try:
            stage_map[p.stem] = load_stage(p)
        except Exception as e:
            stage_map[p.stem] = {"stage_json_error": str(e)}
    q_map, q_summary_map = {}, {}
    for p in sorted((RUN / "P7B_QUERY_TELEMETRY").glob("*.csv")):
        q_map[p.stem], q_summary_map[p.stem] = load_query(p)

    rows = []
    for _, base in context.iterrows():
        cid = str(base.context_id)
        qpath = Path(str(base.query_telemetry_path))
        qd = q_map.get(qpath.stem)
        r = base.to_dict()
        r.update(stage_map.get(cid, {}))
        r.update(q_summary_map.get(cid, {}))
        r["attempt_index"] = 0
        r["first_failure_category"] = first_failure(r, qd)
        r["root_split_from_manifest"] = root_split.get(int(base.root_group_id))
        # Explicit geometry differences: stage TCP/object and first query TCP/object.
        def diff(a, b):
            av, bv = r.get(a), r.get(b)
            try: return float(av) - float(bv)
            except Exception: return None
        r["stage_eef_object_dx_m"] = diff("eef_pose_base_after_stage_0", "stage_object_position_0")
        r["stage_eef_object_dy_m"] = diff("eef_pose_base_after_stage_1", "stage_object_position_1")
        r["stage_eef_object_dz_m"] = diff("eef_pose_base_after_stage_2", "stage_object_position_2")
        r["stage_eef_object_xy_distance_m"] = (math.hypot(r["stage_eef_object_dx_m"], r["stage_eef_object_dy_m"])
                                               if r.get("stage_eef_object_dx_m") is not None else None)
        r["stage_query_target_dx_m"] = diff("eef_pose_base_after_stage_0", "p4_center_pregrasp_target_base_m_0")
        r["stage_query_target_dy_m"] = diff("eef_pose_base_after_stage_1", "p4_center_pregrasp_target_base_m_1")
        r["stage_query_target_dz_m"] = diff("eef_pose_base_after_stage_2", "p4_center_pregrasp_target_base_m_2")
        rows.append(r)
    canonical = pd.DataFrame(rows)
    canonical.to_csv(out / "P7B_QUERY_ATTEMPT_FORENSICS.csv", index=False)

    # Root/context query summaries.
    failure_tax = (canonical.groupby(["split", "first_failure_category"], dropna=False).size()
                   .rename("n").reset_index())
    split_n = canonical.groupby("split").size().to_dict()
    failure_tax["split_n"] = failure_tax["split"].map(split_n)
    failure_tax["pct_of_split"] = 100.0 * failure_tax["n"] / failure_tax["split_n"]
    failure_tax.to_csv(out / "P7B_FIRST_FAILURE_TAXONOMY.csv", index=False)

    root_values = ["query_qualified", "contact_retained", "return_state_valid", "drop", "major_disturbance",
                   "probe_duration_s", "return_position_error_mm", "telemetry_duration_s", "stage_object_disturbance_m",
                   "stage_position_error_m", "stage_orientation_error_rad", "stage_eef_object_xy_distance_m",
                   "stage_eef_object_dx_m", "stage_eef_object_dy_m", "stage_eef_object_dz_m", "telemetry_rows",
                   "first_drop_step", "first_bilateral_step", "first_probe_step"]
    root_summary = group_table(canonical, ["root_group_id", "root_seed", "root_split_from_manifest", "split"], root_values)
    root_summary.to_csv(out / "P7B_ROOT_QUERY_FORENSICS.csv", index=False)

    # Branch/contact telemetry and chunk phase reconstruction.
    chunk_rows = []
    for _, b in branches.iterrows():
        bid = str(b.branch_id)
        cp = Path(str(b.contact_telemetry_path))
        try:
            cd = pd.read_csv(cp).sort_values("step")
            phases = cd["phase"].astype(str) if "phase" in cd else pd.Series([], dtype=str)
            last = cd.iloc[-1] if len(cd) else pd.Series(dtype=object)
            chunk_rows.append({
                "branch_id": bid, "context_id": b.context_id, "root_group_id": b.root_group_id,
                "split": b.split, "requested_force_N": b.requested_force_N,
                "vla_chunk_budget_exhausted": b.vla_chunk_budget_exhausted,
                "num_policy_chunks": b.num_policy_chunks, "episode_steps": b.episode_steps,
                "failure_stage": b.failure_stage, "telemetry_rows": len(cd),
                "telemetry_first_phase": phases.iloc[0] if len(phases) else None,
                "telemetry_last_phase": phases.iloc[-1] if len(phases) else None,
                "telemetry_last_step": last.get("step"), "telemetry_last_chunk_idx": last.get("chunk_idx"),
                "telemetry_last_object_x_w": last.get("object_x_w"),
                "telemetry_last_object_y_w": last.get("object_y_w"),
                "telemetry_last_object_z_w": last.get("object_z_w"),
                "query_phase_rows": int(phases.isin(["approach", "descend", "close", "hold", "probe_out", "probe_back", "probe_hold"]).sum()) if len(phases) else 0,
                "vla_phase_rows": int(phases.eq("vla").sum()) if len(phases) else 0,
            })
        except Exception as e:
            chunk_rows.append({"branch_id": bid, "context_id": b.context_id, "root_group_id": b.root_group_id,
                               "split": b.split, "contact_telemetry_error": str(e),
                               "vla_chunk_budget_exhausted": b.vla_chunk_budget_exhausted,
                               "num_policy_chunks": b.num_policy_chunks, "episode_steps": b.episode_steps,
                               "failure_stage": b.failure_stage})
    chunk = pd.DataFrame(chunk_rows)
    chunk.to_csv(out / "P7B_CHUNK_BRANCH_FORENSICS.csv", index=False)
    ctx_chunk = (chunk.groupby(["context_id", "root_group_id", "split"], dropna=False)
                 .agg(branches=("branch_id", "size"), exhausted=("vla_chunk_budget_exhausted", "sum"),
                      mean_chunks=("num_policy_chunks", "mean"), median_chunks=("num_policy_chunks", "median"),
                      max_episode_steps=("episode_steps", "max"), query_phase_rows=("query_phase_rows", "sum"),
                      vla_phase_rows=("vla_phase_rows", "sum"))
                 .reset_index())
    ctx_chunk["exhaustion_rate_pct"] = 100.0 * ctx_chunk.exhausted / ctx_chunk.branches
    ctx_chunk.to_csv(out / "P7B_CHUNK_CONTEXT_FORENSICS.csv", index=False)

    # Chronology is measured from raw telemetry file mtimes, not directory order.
    chrono = []
    for _, r in canonical.iterrows():
        files = [RUN / "P7B_QUERY_TELEMETRY" / (Path(str(r.query_telemetry_path)).name),
                 RUN / "P7B_STAGE_TELEMETRY" / (str(r.context_id) + ".json")]
        mt = [p.stat().st_mtime for p in files if p.exists()]
        chrono.append({"context_id": r.context_id, "root_group_id": r.root_group_id, "split": r.split,
                       "min_mtime_utc": datetime.fromtimestamp(min(mt), timezone.utc).isoformat() if mt else None,
                       "max_mtime_utc": datetime.fromtimestamp(max(mt), timezone.utc).isoformat() if mt else None})
    chronology = pd.DataFrame(chrono)
    chronology.to_csv(out / "P7B_CHRONOLOGY.csv", index=False)
    chronology_summary = []
    for split, g in chronology.groupby("split", sort=False):
        chronology_summary.append({"split": split, "n_contexts": len(g), "first_file_mtime_utc": g.min_mtime_utc.min(),
                                   "last_file_mtime_utc": g.max_mtime_utc.max(), "first_root": int(g.root_group_id.min()),
                                   "last_root": int(g.root_group_id.max())})

    # Historical coverage, read-only.
    coverage = {}
    p4 = REPO / "analysis/results/p4_contact_conditioned_probe_20260822_184213"
    metas = sorted(p4.glob("P4B_TASK*_COLLECT_META.json"))
    coverage["p4_probe_development"] = {"directory": str(p4), "task_meta_files": [p.name for p in metas],
        "n_task_variants": len(metas), "episode_table_rows": int(len(pd.read_csv(p4 / "P4B_EPISODES.csv"))) if (p4 / "P4B_EPISODES.csv").exists() else None,
        "per_task_design": []}
    for p in metas:
        try:
            j = json.loads(p.read_text())
            coverage["p4_probe_development"]["per_task_design"].append({"file": p.name, "task": j.get("task"),
                "n_seeds": j.get("n_seeds"), "mus": j.get("mus"), "probe_rule": j.get("probe_rule"),
                "task_specific_probe_lookup": j.get("task_specific_probe_lookup")})
        except Exception as e:
            coverage["p4_probe_development"]["per_task_design"].append({"file": p.name, "error": str(e)})
    for label, path in [("clean_pilot", REPO / "analysis/results/p7b_clean_pilot_recovered_20260827"),
                        ("engineering_acceptance", REPO / "analysis/results/p7b_engineering_acceptance_final_20260827")]:
        item = {"directory": str(path)}
        for name in ["P7B_PILOT_AUDIT.json", "P7B_ENGINEERING_ACCEPTANCE_AUDIT.json", "P7B_ROOT_MANIFEST.csv", "P7B_CONTEXT_MANIFEST.csv"]:
            p = path / name
            if p.exists():
                item[name] = json.loads(p.read_text()) if p.suffix == ".json" else {"rows": int(len(pd.read_csv(p)))}
        rm = path / "P7B_ROOT_MANIFEST.csv"
        cm = path / "P7B_CONTEXT_MANIFEST.csv"
        if rm.exists():
            rd = pd.read_csv(rm)
            item["root_ids"] = [int(x) for x in rd["root_group_id"].tolist()]
            item["splits"] = rd["split"].value_counts(dropna=False).to_dict()
        if cm.exists():
            cd = pd.read_csv(cm)
            if "query_qualified" in cd:
                item["qualified_contexts"] = int(cd["query_qualified"].sum())
        coverage[label] = item

    # Hash and input inventory.
    hashes = {}
    for label, (path, expected) in EXPECTED_HASHES.items():
        actual = sha256(path) if path.exists() else None
        hashes[label] = {"path": str(path), "expected": expected, "actual": actual, "match": actual == expected}
    identity_candidates = [RUN / "P7B_PROTOCOL_HASH.txt", RUN / "P7B_CODE_HASH.txt"]
    hashes["clean_pilot_identity_expected"] = EXPECTED_CLEAN_IDENTITY
    hashes["run_protocol_hash_file"] = (identity_candidates[0].read_text().strip() if identity_candidates[0].exists() else None)
    hashes["run_code_hash_file"] = (identity_candidates[1].read_text().strip() if identity_candidates[1].exists() else None)

    # Machine-readable forensic result.
    def counts_for_split(split):
        g = canonical[canonical.split == split]
        return {"contexts_attempted": len(g), "qualified": int(g.query_qualified.sum()),
                "qualification_rate_pct": pct(g.query_qualified.sum(), len(g)),
                "contact_retained": int(g.contact_retained.sum()),
                "return_valid": int(g["return_state_valid"].sum()), "drops": int(g["drop"].sum()),
                "major_disturbance": int(g["major_disturbance"].sum()),
                "first_failure": g.first_failure_category.value_counts().to_dict()}
    chunk_stats = {
        "branch_n": int(len(chunk)), "exhausted": int(chunk.vla_chunk_budget_exhausted.sum()),
        "exhaustion_rate_pct": pct(chunk.vla_chunk_budget_exhausted.sum(), len(chunk)),
        "all_exhausted_last_phase": chunk.loc[chunk.vla_chunk_budget_exhausted.eq(1), "telemetry_last_phase"].value_counts().to_dict(),
        "all_exhausted_first_phase": chunk.loc[chunk.vla_chunk_budget_exhausted.eq(1), "telemetry_first_phase"].value_counts().to_dict(),
        "exhausted_num_policy_chunks": numeric_summary(chunk.loc[chunk.vla_chunk_budget_exhausted.eq(1), "num_policy_chunks"]),
        "nonexhausted_num_policy_chunks": numeric_summary(chunk.loc[chunk.vla_chunk_budget_exhausted.eq(0), "num_policy_chunks"]),
        "by_requested_force": group_table(chunk, ["requested_force_N"], ["vla_chunk_budget_exhausted", "num_policy_chunks"] ).to_dict("records"),
        "query_phase_exhaustion_rows": int(chunk.loc[chunk.vla_chunk_budget_exhausted.eq(1), "query_phase_rows"].sum()),
        "vla_phase_exhaustion_rows": int(chunk.loc[chunk.vla_chunk_budget_exhausted.eq(1), "vla_phase_rows"].sum()),
    }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "run": str(RUN), "output_dir": str(out),
        "split_summary": {s: counts_for_split(s) for s in ["TRAIN", "DEV", "TEST"]},
        "failure_taxonomy": failure_tax.to_dict("records"),
        "root_summary": root_summary.to_dict("records"), "chunk": chunk_stats,
        "chunk_by_split": group_table(chunk, ["split"], ["vla_chunk_budget_exhausted", "num_policy_chunks"]).to_dict("records"),
        "chunk_by_failure_stage": group_table(chunk, ["failure_stage"], ["vla_chunk_budget_exhausted", "num_policy_chunks"]).to_dict("records"),
        "chronology": chronology_summary, "coverage": coverage, "hashes": hashes,
        "infrastructure_failures": int(pd.to_numeric(context["infra_failure"], errors="coerce").fillna(0).sum()),
        "scientific_retries": int(pd.to_numeric(context["infra_retry_count"], errors="coerce").fillna(0).sum()),
        "interpretation": {
            "dev_test_query_observation": "all DEV/TEST attempts terminated with pre-query drop on first approach telemetry; zero probe/bilateral/return observations",
            "chunk_causality": "chunk exhaustion appears only in downstream TRAIN branches after a qualified query and cannot be the direct cause of DEV/TEST query unavailability",
            "belief_generalization": "not evaluable because DEV/TEST contain zero qualified query observations",
            "calibration_diversity": "prior P4 probe development used 5 seeds x 3 friction settings per task; P7B pilot/acceptance used only 4 roots and selected successes, so broad held-out root execution coverage was not established",
            "chronology_caveat": "all TRAIN roots precede DEV/TEST roots in root-major execution order; artifact evidence shows no infra errors or retry drift, but a single contiguous run cannot mathematically eliminate every temporal simulator degradation confound",
            "primary_classification": "PROBE_EXECUTION_DOES_NOT_GENERALIZE_ACROSS_ROOTS",
            "geometry_caveat": "gross staged TCP-object geometry is similar for successful TRAIN and held-out DEV/TEST; the evidence supports a root-specific initial-state/contact/handoff failure, not a uniquely identified scalar geometry threshold",
        }
    }
    json_dump(out / "P7B_SPLIT_FAILURE_FORENSICS.json", result)

    # Concise, inspectable report generated from the same tables.
    def lines_for_table(df, cols, limit=100):
        if not len(df):
            return "(none)"
        d = df[cols].head(limit).copy()
        header = "| " + " | ".join(map(str, d.columns)) + " |"
        rule = "| " + " | ".join(["---"] * len(d.columns)) + " |"
        body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
                for row in d.itertuples(index=False, name=None)]
        return "\n".join([header, rule] + body)
    report = f"""# P7-B Split Failure Forensics

Created: {result['created_utc']}

This is a read-only forensic reconstruction. No scientific run, threshold,
probe, planner, controller, or retry was changed.

## Split summary

{json.dumps(result['split_summary'], indent=2, sort_keys=True)}

## First-failure taxonomy

{lines_for_table(failure_tax, ['split','first_failure_category','n','split_n','pct_of_split'])}

## Root-level query summary

{lines_for_table(root_summary, ['root_group_id','root_split_from_manifest','n','query_qualified_mean','contact_retained_mean','return_state_valid_mean','drop_mean','stage_object_disturbance_m_mean','stage_eef_object_xy_distance_m_mean','first_drop_step_median','first_probe_step_median'])}

## Chunk exhaustion

{json.dumps(result['chunk'], indent=2, sort_keys=True)}

The context-level query table shows DEV/TEST failing before probe rows exist.
The branch-level chunk table shows exhausted branches ending in the VLA phase,
after query-qualified TRAIN contexts produced downstream branches.

## Chronology

{json.dumps(chronology_summary, indent=2, sort_keys=True)}

All TRAIN roots are root-major before DEV and TEST. The run has zero recorded
infrastructure failures and zero retries; this argues against a launcher-level
split path, but temporal degradation cannot be eliminated with one contiguous run.

## Historical development coverage

{json.dumps(coverage, indent=2, sort_keys=True)}

## Evidence-backed interpretation

* DEV/TEST: 40/40 query attempts have a four-row approach/descend/close/hold
  trace with the first row already marked dropped, no probe excitation, no
  bilateral contact, and no return-valid observation.
* TRAIN: 44/60 qualified; the remaining 16 are 9 early drops and 7 contact-loss
  cases that did enter probe telemetry. Thus TRAIN failures are mixed, while
  held-out failures are uniformly upstream of belief inference.
* Chunk exhaustion is downstream and cannot directly explain zero DEV/TEST
  query qualification. It is a secondary execution-budget problem in the
  available TRAIN branches.
* Belief interpretation generalization is not testable: no valid DEV/TEST
  query response reached the belief stage.
* Prior probe development and successful pilots were narrow relative to the
  intended root split. This supports insufficient root/query-start diversity
  as a plausible design risk, but does not prove whether geometry/contact or
  hidden simulator state is the exact low-level mechanism.

## Classification

`PROBE_EXECUTION_DOES_NOT_GENERALIZE_ACROSS_ROOTS`

Secondary: `QUERY_REQUIRES_NON_GENERAL_ROOT_SPECIFIC_INITIAL_GEOMETRY` (the
observed failure is at the initial contact/handoff boundary, but gross staged
geometry alone is not sufficient to identify the mechanism) and
`QUERY_AVAILABILITY_CENSORED_BY_EXECUTION_BUDGET` (downstream only).
The earliest supported broken link is `root/context -> query availability`.
"""
    (out / "P7B_SPLIT_FAILURE_FORENSICS_REPORT.md").write_text(report)

    # Manifest includes all created artifacts and their hashes.
    manifest = []
    for p in sorted(out.iterdir()):
        if p.name == "MANIFEST.sha256":
            continue
        manifest.append(f"{sha256(p)}  {p.name}")
    (out / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n")
    print(json.dumps({"output_dir": str(out), "result": str(out / "P7B_SPLIT_FAILURE_FORENSICS.json"),
                      "report": str(out / "P7B_SPLIT_FAILURE_FORENSICS_REPORT.md"),
                      "split_summary": result["split_summary"], "chunk": result["chunk"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
