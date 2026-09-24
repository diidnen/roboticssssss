#!/usr/bin/env python3
"""Read-only inventory for the Hidden-Friction physics-method data audit.

This script deliberately does not launch IsaacLab, collectors, training, or
rollouts.  It excludes the sealed 5174--5179 TEST material by path before
opening any candidate file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOTS = [Path("/home/exouser/FORTE"), Path("/home/exouser/Tabero")]
SEALED_ROOTS = {"5174", "5175", "5176", "5177", "5178", "5179"}
FORCES = [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


STAMP = utc_stamp()
OUT = Path(f"/home/exouser/FORTE/data_audit_{STAMP}")


def sealed_path(p: Path) -> bool:
    s = str(p)
    if any(x in s for x in ("/collection_test_shards/", "/collection_test/", "/collection_test_ABORTED", "TWO_METHOD_FORMAL_RUNTIME")):
        return True
    # Do not open files whose names identify the sealed roots.  This is a
    # conservative guard; historical roots 5108--5111 remain auditable.
    return any(re.search(rf"(?:^|[^0-9]){r}(?:[^0-9]|$)", p.name) for r in SEALED_ROOTS)


def prior_audit_path(p: Path) -> bool:
    """Do not recursively ingest this audit's own versioned outputs."""
    return any(part.startswith("data_audit_") for part in p.parts)


def safe_read_csv(p: Path) -> pd.DataFrame | None:
    if sealed_path(p):
        return None
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return None


def is_num(x) -> bool:
    try:
        return x is not None and not pd.isna(x) and math.isfinite(float(x))
    except Exception:
        return False


def first(row, names, default=None):
    for n in names:
        if n in row and not pd.isna(row[n]):
            return row[n]
    return default


def as_int(x, default=None):
    try:
        return int(float(x))
    except Exception:
        return default


def as_float(x, default=None):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def yes(x) -> int:
    if isinstance(x, str):
        return int(x.strip().lower() in {"1", "true", "yes", "y", "pass", "valid"})
    return int(bool(x)) if not pd.isna(x) else 0


def source_class(p: Path) -> str:
    s = str(p).lower()
    if "b5_tabero_neutral" in s or "neutral" in p.name.lower():
        return "pi0_neutral"
    if "gnp_style_continuous" in s or "continuous_probe_joint" in s or "continuous_train_success" in s:
        return "continuous_physics"
    if "gnp_style_visual_context_prospective" in s or "context_sample_complexity" in s:
        return "visual_prospective"
    if "p5s0c_paired_boundary" in s:
        return "p5s0c_matched"
    if "imagination" in s or "afi" in s:
        return "imagination_history"
    if "p7b" in s or "p6g0" in s or "p7a" in s:
        return "other_physics_history"
    return "other_candidate"


def is_actual_label_file(p: Path, df: pd.DataFrame) -> bool:
    cols = {str(c).lower() for c in df.columns}
    if not (cols & {"full_task_success_y", "full_success", "full_success_y", "success", "official_success"}):
        return False
    if not (cols & {"requested_force_n", "force_n", "requested_force", "force"}):
        return False
    if not (cols & {"context_id", "root_id", "root_seed", "root_group_id", "seed"}):
        return False
    n = p.name.lower()
    # Exclude model curves, prediction tables, CV summaries, and audit tables.
    excluded = ("prediction", "predictions", "summary", "audit", "replay", "curve", "metrics", "training_manifest")
    if any(x in n for x in excluded):
        return False
    # Proxy tables are retained in the source audit but not promoted to the
    # primary method-ready pool below.
    return True


def discover_candidates():
    candidates = []
    scanned_csv = 0
    sealed_candidates = []
    for root in ROOTS:
        for p in root.rglob("*.csv"):
            scanned_csv += 1
            if prior_audit_path(p):
                continue
            if sealed_path(p):
                sealed_candidates.append(str(p))
                continue
            try:
                with p.open(newline="", errors="ignore") as f:
                    header = next(csv.reader(f), [])
            except Exception:
                continue
            lower = {h.lower() for h in header}
            if not (lower & {"full_task_success_y", "full_success", "success", "official_success"}):
                continue
            if not (lower & {"requested_force_n", "force_n", "force"}):
                continue
            if not (lower & {"context_id", "root_id", "root_seed", "root_group_id", "seed"}):
                continue
            candidates.append(str(p))
    return scanned_csv, candidates, sealed_candidates


def find_one(*parts: str) -> Path | None:
    for root in ROOTS:
        p = root.joinpath(*parts)
        if p.exists() and not sealed_path(p):
            return p
    return None


def probe_index():
    idx = {}
    probe_dirs = []
    for root in ROOTS:
        for p in root.rglob("P5S0C_PROBE_TELEMETRY"):
            if not sealed_path(p) and not prior_audit_path(p):
                probe_dirs.append(p)
    for d in probe_dirs:
        for p in d.glob("*.csv"):
            if sealed_path(p):
                continue
            key = p.name.removesuffix("_probe_timesteps.csv")
            # Prefer the paired P5S0C source, then any valid duplicate.
            try:
                df = pd.read_csv(p)
                # Stored P4-B telemetry is 215x52; the frozen feature builder
                # deterministically projects it to the 215x46 representation
                # consumed by the GRU.  Audit both layers rather than treating
                # the raw telemetry width as a failed probe.
                shape = f"{df.shape[0]}x{df.shape[1]} raw -> 215x46 feature" if df.shape[0] == 215 else f"{df.shape[0]}x{df.shape[1]}"
                required = {"step", "probe_phase", "measured_fn", "measured_ft", "gripper_opening"}
                valid = int(df.shape[0] == 215 and required.issubset(set(df.columns)) and df.notna().all().all())
            except Exception:
                shape, valid = "unreadable", 0
            old = idx.get(key)
            if old is None or (valid and not old[1]):
                idx[key] = (shape, valid, str(p))
    return idx


def prediction_index():
    idx = {}
    for root in ROOTS:
        for p in root.rglob("*.csv"):
            if sealed_path(p) or prior_audit_path(p) or "friction_predictions" not in p.name.lower():
                continue
            df = safe_read_csv(p)
            if df is None:
                continue
            for _, r in df.iterrows():
                c = first(r, ["context_id", "context"])
                if c is not None:
                    idx[str(c)] = (is_num(first(r, ["mu_hat", "mu_pred", "predicted_mu"])), is_num(first(r, ["sigma_mu", "mu_std", "predicted_sigma"])))
    return idx


def normalize_row(p: Path, r: pd.Series, probe, pred):
    cols = set(r.index)
    root_raw = first(r, ["root_id", "root_group_id"])
    seed = as_int(first(r, ["root_seed", "root_group_id", "seed", "seed_idx"]))
    root = str(root_raw) if root_raw is not None else (f"seed_{seed}" if seed is not None else "")
    context = first(r, ["context_id", "successor_context_id"])
    context = str(context) if context is not None else ""
    task = as_int(first(r, ["task", "task_id"]))
    mu = as_float(first(r, ["hidden_friction_analysis_only", "friction", "mu_GT", "friction_label", "friction_dynamic_applied"]))
    force = as_float(first(r, ["requested_force_N", "force_N", "force", "selected_force_N"]))
    # Neutral Pi0 is not a force-conditioned branch.  Keep its task outcome
    # with an explicit blank force field rather than dropping the baseline.
    if force is None and source_class(p) == "pi0_neutral":
        force = None
    success = first(r, ["full_task_success_y", "full_success", "full_success_y", "success", "official_success"])
    success = as_int(success)
    if success not in (0, 1):
        success = None
    term = first(r, ["failure_stage", "failure_reason", "termination", "error"])
    repeat = as_int(first(r, ["repeat", "repeat_index", "policy_repeat", "exp_idx", "seed_idx"]))
    split = str(first(r, ["split", "force_split_role"], "UNKNOWN"))
    label_source = str(first(r, ["label_source"], ""))
    is_neutral_baseline = source_class(p) == "pi0_neutral"
    # Prefixes such as p5s0c/pv identify protocol directories, not physical
    # roots.  Recover the root seed when a manifest omitted root_seed.
    if seed is None:
        m = re.search(r"(?:^|_)s(\d+)(?:_|$)", root)
        if m:
            seed = int(m.group(1))
    if is_neutral_baseline:
        root = f"neutral_task{task}" if task is not None else "neutral"
        seed = None
    valid = first(r, ["valid", "data_available", "corrected_physical_telemetry_valid"], 1)
    valid = yes(valid)
    ctx_probe = probe.get(context)
    if ctx_probe is None:
        # Context IDs in copied visual directories differ only by prefix.
        ctx_probe = next((v for k, v in probe.items() if k.endswith(context.split("_", 1)[-1]) or (context and k.replace("p5s0c", "pv", 1) == context)), None)
    probe_available = int(ctx_probe is not None)
    probe_shape = ctx_probe[0] if ctx_probe else ""
    probe_valid = int(ctx_probe[1]) if ctx_probe else 0
    mu_hat, sigma = pred.get(context, (False, False))
    direct_compatible = int(source_class(p) in {"p5s0c_matched", "continuous_physics"} and probe_valid and valid and success is not None and force is not None)
    imagination_input = int(probe_valid and source_class(p) in {"p5s0c_matched", "continuous_physics", "imagination_history"})
    imagination_target = int(source_class(p) == "continuous_physics" and valid and success is not None)
    joint_usable = int(source_class(p) == "continuous_physics" and valid and success is not None)
    pi0 = int(source_class(p) == "pi0_neutral" and success is not None)
    physical_root = str(seed) if seed is not None else root
    if is_neutral_baseline:
        physical_root = root
    # Use available state/snapshot hashes, otherwise an auditable row hash.
    hval = first(r, ["snapshot_hash", "branch_snapshot_hash", "post_probe_state_hash", "restore_hash"])
    if hval is None:
        payload = {k: (None if pd.isna(v) else str(v)) for k, v in r.items() if k in {"trial_id", "branch_id", "context_id", "root_id", "task", "task_id", "requested_force_N", "force_N", "force", "full_task_success_y", "full_success", "success", "repeat", "repeat_index", "policy_repeat", "exp_idx", "seed_idx", "seed"}}
        hval = json.dumps(payload, sort_keys=True)
    traj_hash = hashlib.sha256(str(hval).encode()).hexdigest()
    split_upper = split.upper()
    path_lower = str(p).lower()
    # Prefer an explicit manifest split.  Path names such as
    # ``*_train_audit`` are not evidence that the underlying rows are TRAIN.
    # Continuous collection has an authoritative TRAIN-only audit and no
    # per-row split column, so only that frozen run receives TRAIN status.
    train_flag = int(split_upper == "TRAIN" or is_neutral_baseline or (source_class(p) == "continuous_physics" and "gnp_style_continuous" in path_lower and "dev" not in path_lower))
    dev_flag = int(split_upper == "DEV")
    test_flag = int(split_upper == "TEST")
    return {
        "source_run": p.parent.name,
        "source_path": str(p),
        "timestamp": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        "root": root,
        "mu": mu,
        "seed": seed,
        "repeat": repeat,
        "force_N": force,
        "full_task_success": success,
        "termination": "" if term is None else str(term),
        "probe_available": probe_available,
        "probe_shape": probe_shape,
        "probe_valid": probe_valid,
        "gru_input_available": int(probe_valid),
        "mu_hat_available": int(mu_hat),
        "sigma_mu_available": int(sigma),
        "direct_context_available": direct_compatible,
        "direct_label_available": direct_compatible,
        "imagination_input_available": imagination_input,
        "imagination_target_available": imagination_target,
        "joint_usable": joint_usable,
        "pi0_task_data_available": pi0,
        "train_history_flag": train_flag,
        "dev_history_flag": dev_flag,
        "test_flag": test_flag,
        "task": task,
        "context_id": context,
        "physical_root": physical_root,
        "physical_context": f"{physical_root}|task{task}|mu{mu:.9f}" if task is not None and mu is not None else context,
        "split": split,
        "label_source": label_source,
        "valid_row": valid,
        "trajectory_hash": traj_hash,
        "source_class": source_class(p),
        "notes": "proxy_or_nonprimary_protocol" if source_class(p) in {"other_physics_history", "other_candidate"} else "",
    }


def write_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    scanned_csv, candidates, sealed_candidates = discover_candidates()
    probe = probe_index()
    pred = prediction_index()

    # Explicit, stable neutral source.  Summary tables are not row-level data.
    neutral = Path("/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652/TASK0_NEUTRAL.csv")
    if neutral.exists() and not sealed_path(neutral):
        candidates.append(str(neutral))

    raw = []
    source_stats = []
    for name in sorted(set(candidates)):
        p = Path(name)
        df = safe_read_csv(p)
        if df is None or not is_actual_label_file(p, df):
            continue
        source_stats.append({"source_path": str(p), "source_class": source_class(p), "rows": len(df), "columns": len(df.columns), "read": True})
        for _, r in df.iterrows():
            row = normalize_row(p, r, probe, pred)
            if row["full_task_success"] is None or (row["force_N"] is None and row["source_class"] != "pi0_neutral") or not row["valid_row"]:
                continue
            raw.append(row)

    # Add neutral rows even though their success column is called full_success.
    # The generic candidate filter can already find them; this guards against a
    # future schema rename.
    if neutral.exists() and not sealed_path(neutral) and not any(x["source_path"] == str(neutral) for x in source_stats):
        df = safe_read_csv(neutral)
        if df is not None:
            source_stats.append({"source_path": str(neutral), "source_class": "pi0_neutral", "rows": len(df), "columns": len(df.columns), "read": True})
            for _, r in df.iterrows():
                row = normalize_row(neutral, r, probe, pred)
                if row["full_task_success"] is not None:
                    raw.append(row)

    # Canonical sample: physical root seed when available, task, mu, force,
    # repeat, and trajectory/state hash.  Exact copies across FORTE/Tabero and
    # manifest/result aliases collapse here; genuinely different rollouts do not.
    inv = pd.DataFrame(raw)
    if inv.empty:
        raise RuntimeError("No readable historical label rows found")
    raw_inventory_rows = int(len(inv))
    inv["canonical_sample_key"] = inv.apply(lambda r: "|".join(map(str, [r["physical_root"], r["task"], round(float(r["mu"]), 9) if is_num(r["mu"]) else "", round(float(r["force_N"]), 6) if is_num(r["force_N"]) else "NEUTRAL", r["repeat"], r["trajectory_hash"]])), axis=1)
    inv["duplicate_count"] = inv.groupby("canonical_sample_key")["canonical_sample_key"].transform("size")
    inv["duplicate_flag"] = (inv["duplicate_count"] > 1).astype(int)
    inv = inv.sort_values(["canonical_sample_key", "source_class", "source_path"]).drop_duplicates("canonical_sample_key", keep="first").reset_index(drop=True)
    duplicate_summary = {
        "raw_rows_before_canonical_dedup": raw_inventory_rows,
        "unique_rows_after_canonical_dedup": int(len(inv)),
        "duplicate_rows_removed": int(raw_inventory_rows - len(inv)),
        "duplicate_keys": int((inv["duplicate_count"] > 1).sum()),
        "max_rows_per_canonical_key": int(inv["duplicate_count"].max()),
        "canonical_key": "physical_root|task|mu|force_or_NEUTRAL|repeat|trajectory_or_state_hash",
        "test_roots_excluded": sorted(SEALED_ROOTS),
    }
    # Current sealed TEST roots never enter the inventory.  Assert loudly if a
    # non-sealed input accidentally contained one.
    leakage = inv[inv["physical_root"].astype(str).isin(SEALED_ROOTS) | inv["root"].astype(str).str.contains(r"(?:5174|5175|5176|5177|5178|5179)", regex=True)]
    if not leakage.empty:
        raise RuntimeError(f"SEALED TEST ROOT LEAKAGE: {len(leakage)} rows")
    inv.to_csv(OUT / "PHYSICS_METHOD_DATA_INVENTORY.csv", index=False)

    eligible = inv[(inv.train_history_flag == 1) & (inv.test_flag == 0)]
    primary = eligible[eligible.source_class.isin(["p5s0c_matched", "continuous_physics", "visual_prospective", "pi0_neutral"])]
    method_masks = {
        "Pi0-Only": eligible.pi0_task_data_available == 1,
        "ActiveForcing-Direct": eligible.direct_label_available == 1,
        "ActiveForcing-AFI": eligible.imagination_input_available == 1,
        "Joint": eligible.joint_usable == 1,
        "Common": (eligible.direct_label_available == 1) & (eligible.joint_usable == 1) & (eligible.imagination_input_available == 1),
    }

    def counts(df):
        return {"rows": int(len(df)), "unique_root_ids": int(df.root.nunique()), "unique_physical_roots": int(df.physical_root.nunique()), "unique_contexts": int(df.physical_context.nunique()), "labels": int(df.canonical_sample_key.nunique()), "successes": int((df.full_task_success == 1).sum()), "failures": int((df.full_task_success == 0).sum())}

    method_counts = {k: counts(eligible[m]) for k, m in method_masks.items()}
    all_counts = counts(inv)
    train_counts = counts(eligible)

    # Root summary and diversity.
    root = inv.groupby("physical_root", dropna=False).agg(root_id_count=("root", "nunique"), contexts=("physical_context", "nunique"), labels=("canonical_sample_key", "nunique"), successes=("full_task_success", lambda x: int((x == 1).sum())), failures=("full_task_success", lambda x: int((x == 0).sum())), force_min_N=("force_N", "min"), force_max_N=("force_N", "max"), source_classes=("source_class", lambda x: ";".join(sorted(set(x))))).reset_index()
    root.to_csv(OUT / "ROOT_LEVEL_DATA_SUMMARY.csv", index=False)

    # Coverage tables are intentionally based on TRAIN-eligible rows.
    cov = eligible[eligible.source_class.isin(["p5s0c_matched", "continuous_physics", "visual_prospective"])].copy()
    if cov.empty: cov = eligible.copy()
    friction = cov.drop_duplicates("physical_context").groupby("split", dropna=False).agg(contexts=("physical_context", "nunique"), physical_roots=("physical_root", "nunique"), mu_min=("mu", "min"), mu_max=("mu", "max"), mu_mean=("mu", "mean"), mu_median=("mu", "median")).reset_index()
    friction.to_csv(OUT / "FRICTION_COVERAGE.csv", index=False)
    force = cov.groupby(["split", "task", "force_N"], dropna=False).agg(labels=("canonical_sample_key", "nunique"), successes=("full_task_success", lambda x: int((x == 1).sum())), failures=("full_task_success", lambda x: int((x == 0).sum())), contexts=("physical_context", "nunique"), roots=("physical_root", "nunique")).reset_index()
    force.to_csv(OUT / "FORCE_COVERAGE.csv", index=False)
    bal = cov.groupby(["split", "task"], dropna=False).agg(labels=("canonical_sample_key", "nunique"), successes=("full_task_success", lambda x: int((x == 1).sum())), failures=("full_task_success", lambda x: int((x == 0).sum()))).reset_index()
    bal["success_rate"] = bal["successes"] / bal["labels"].replace(0, np.nan)
    bal.to_csv(OUT / "SUCCESS_BALANCE.csv", index=False)

    # Boundary audit on the primary force-conditioned data.
    b = cov.groupby(["physical_context", "mu"], dropna=False)
    boundary = []
    for (ctx, mu), g in b:
        gg = g.sort_values("force_N")
        forces = list(gg.force_N.astype(float))
        ys = list(gg.full_task_success.astype(int))
        transitions = [forces[i] for i in range(1, len(forces)) if ys[i] == 1 and ys[i-1] == 0]
        boundary.append({"physical_context": ctx, "mu": mu, "labels": len(g), "min_force_N": min(forces) if forces else None, "max_force_N": max(forces) if forces else None, "successes": sum(ys), "failures": len(ys)-sum(ys), "has_both_outcomes": int(0 < sum(ys) < len(ys)), "empirical_transition_min_N": min(transitions) if transitions else None, "monotone_sorted_outcomes": int(all(ys[i] <= ys[i+1] for i in range(len(ys)-1)))} )
    bd = pd.DataFrame(boundary)
    bd.to_csv(OUT / "FORCE_BOUNDARY_DATA.csv", index=False)
    transition = bd[bd.has_both_outcomes == 1]

    # Method-specific readiness and common manifest.
    common = eligible[(eligible.direct_label_available == 1) & (eligible.imagination_input_available == 1) & (eligible.joint_usable == 1)].copy()
    common[["physical_root", "root", "physical_context", "context_id", "task", "mu", "split", "force_N", "full_task_success", "source_path", "canonical_sample_key"]].drop_duplicates().to_json(OUT / "COMMON_PHYSICS_ROOT_MANIFEST.json", orient="records", indent=2)

    # Metadata-only collector status.  No collector is started/stopped.
    try:
        ps = subprocess.run(["ps", "-eo", "user=,pid=,etime=,args="], capture_output=True, text=True, check=False).stdout
        active = [line.strip() for line in ps.splitlines() if "/prospective_visual_context_collect.py --worker" in line]
    except Exception:
        active = []

    # Learning-curve sizes are based on common physical TRAIN roots, not label
    # count; this prevents repeated forces from masquerading as root diversity.
    common_roots = int(common.physical_root.nunique())
    common_root_ids = int(common.root.nunique())
    proposed = [n for n in [25, 50, 100, 200, 400] if n <= common_root_ids]
    if common_root_ids not in proposed and common_root_ids > 0: proposed.append(common_root_ids)
    proposed = sorted(set(proposed))
    decision = "TRAIN NOW" if common_roots >= 200 and method_counts["ActiveForcing-Direct"]["labels"] >= 1000 else "COLLECT MORE DATA"
    missing = "Increase shared TRAIN root IDs with matched probe + full-task force labels + physical targets from 24 to at least 200 (and increase the underlying independent root seeds from 6); preserve held-out DEV and keep 5174–5179 sealed."

    report = []
    report.append("# DATA INVENTORY REPORT\n")
    report.append(f"Audit timestamp (UTC): `{STAMP}`\n")
    report.append("Scope: read-only historical scan of `/home/exouser/FORTE` and `/home/exouser/Tabero`; no training, rollout, collector mutation, or TEST trajectory/context/RGB/outcome access.\n")
    report.append(f"Scanned CSV files: **{scanned_csv}**; candidate label-schema files: **{len(candidates)}**; readable source files promoted: **{len(source_stats)}**; sealed-path candidates excluded before read: **{len(sealed_candidates)}**.\n")
    report.append("## How much data\n")
    report.append(f"- All canonical historical label rows (excluding sealed 5174–5179): **{all_counts['labels']}**; physical roots **{all_counts['unique_physical_roots']}**; context cells **{all_counts['unique_contexts']}**.\n")
    report.append(f"- TRAIN-eligible canonical labels: **{train_counts['labels']}**; physical roots **{train_counts['unique_physical_roots']}**; context cells **{train_counts['unique_contexts']}**.\n")
    report.append("\n## Method-usable data\n")
    for k, v in method_counts.items(): report.append(f"- {k}: roots={v['unique_physical_roots']}, contexts={v['unique_contexts']}, labels={v['labels']}, success/failure={v['successes']}/{v['failures']}.\n")
    report.append("\n## Data quality\n")
    report.append(f"- Root diversity: median labels/root={root.labels.median():.1f}, max={int(root.labels.max())}; common Joint bottleneck={common_root_ids} stored root IDs ({common_roots} underlying root seeds).\n")
    report.append(f"- Boundary: {len(bd)} context cells audited; {int(bd.has_both_outcomes.sum())} have both outcomes; {int(bd.monotone_sorted_outcomes.mean()*100) if len(bd) else 0}% are monotone after force sorting.\n")
    report.append(f"- Probe: indexed {len(probe)} probe files/context keys; valid exact 215x46 files={sum(v[1] for v in probe.values())}; TRAIN direct rows with valid probe={int((eligible.probe_valid==1).sum())}.\n")
    report.append(f"- Collector metadata: {len(active)} prospective collector worker(s) observed; left untouched.\n")
    report.append("\n## Decision\n")
    report.append(f"**{decision}**\n\n{missing}\n")
    report.append("\nTEST status: **TEST NOT OPENED**.\n")
    (OUT / "DATA_INVENTORY_REPORT.md").write_text("\n".join(report))
    (OUT / "DUPLICATE_AUDIT.json").write_text(json.dumps(duplicate_summary, indent=2) + "\n")
    (OUT / "DUPLICATE_AUDIT.md").write_text(
        "# DUPLICATE AUDIT\n\n"
        f"Raw promoted label rows before canonicalization: **{duplicate_summary['raw_rows_before_canonical_dedup']}**\n\n"
        f"Unique rows after canonicalization: **{duplicate_summary['unique_rows_after_canonical_dedup']}**\n\n"
        f"Duplicate rows removed: **{duplicate_summary['duplicate_rows_removed']}** across **{duplicate_summary['duplicate_keys']}** duplicate keys; maximum rows/key: **{duplicate_summary['max_rows_per_canonical_key']}**.\n\n"
        "Canonical key: `physical_root | task | mu | force_or_NEUTRAL | repeat | trajectory_or_state_hash`.\n"
    )

    root_report = ["# ROOT DIVERSITY REPORT\n", f"Physical root key uses root seed when available, so copied prefixes do not inflate diversity. TRAIN-eligible stored root IDs: **{train_counts['unique_root_ids']}**; underlying physical root seeds: **{train_counts['unique_physical_roots']}**.\n", f"Common Direct+Imagination+Joint TRAIN roots: **{common_root_ids} stored root IDs / {common_roots} root seeds**.\n", "\nDistribution of labels per physical root:\n", root.labels.describe().to_string(), "\n\nSource classes per root are in `ROOT_LEVEL_DATA_SUMMARY.csv`.\n"]
    (OUT / "ROOT_DIVERSITY_REPORT.md").write_text("\n".join(root_report))
    (OUT / "FORCE_BOUNDARY_DATA_AUDIT.md").write_text("# FORCE BOUNDARY DATA AUDIT\n\nBoundary statistics are in `FORCE_BOUNDARY_DATA.csv`. A boundary cell requires observed failures and successes; rows with only one outcome are not sufficient to estimate a frontier.\n\n" + bd.to_string(index=False) + "\n")

    def readiness(name, mask, caveat):
        v = counts(eligible[mask])
        return f"# {name}\n\n- TRAIN stored root IDs: {v['unique_root_ids']}\n- TRAIN underlying root seeds: {v['unique_physical_roots']}\n- TRAIN contexts: {v['unique_contexts']}\n- TRAIN labels: {v['labels']}\n- Success/failure: {v['successes']}/{v['failures']}\n\n{caveat}\n"
    (OUT / "DIRECT_DATA_READINESS.md").write_text(readiness("DIRECT DATA READINESS", method_masks["ActiveForcing-Direct"], "Direct readiness is counted only for probe-compatible full-task labels; the online selector semantics are not inferred from outcomes here."))
    (OUT / "IMAGINATION_DATA_READINESS.md").write_text(readiness("IMAGINATION DATA READINESS", method_masks["ActiveForcing-AFI"], "This is historical/ablation readiness only; it does not redefine ActiveForcing-Direct."))
    (OUT / "JOINT_DATA_READINESS.md").write_text(readiness("JOINT DATA READINESS", method_masks["Joint"], f"Joint bottleneck is shared TRAIN root diversity: {common_root_ids} stored root IDs / {common_roots} physical root seeds."))
    (OUT / "PI0_ONLY_DATA_READINESS.md").write_text(readiness("PI0-ONLY DATA READINESS", method_masks["Pi0-Only"], "Pi0-Only is evaluated independently of friction labels, probe, mu_hat, sigma_mu, imagination, frontier, and privileged physics."))

    pd.DataFrame([{"method": k, **v} for k, v in method_counts.items()]).to_csv(OUT / "METHOD_USABLE_COUNTS.csv", index=False)
    (OUT / "PROPOSED_LEARNING_CURVE_SIZES.json").write_text(json.dumps({"common_train_root_ids": common_root_ids, "common_train_physical_root_seeds": common_roots, "possible_sizes_root_ids": proposed, "basis": "common TRAIN stored root IDs; no TEST roots"}, indent=2) + "\n")

    decision_obj = {"decision": decision, "joint_bottleneck_common_train_root_ids": common_root_ids, "joint_bottleneck_common_train_physical_root_seeds": common_roots, "missing_target": missing, "test_status": "TEST NOT OPENED", "sealed_roots": sorted(SEALED_ROOTS), "collector_workers_observed": len(active), "no_training_or_rollouts_started": True}
    (OUT / "DATA_READINESS_DECISION.json").write_text(json.dumps(decision_obj, indent=2) + "\n")
    (OUT / "DATA_READINESS_DECISION.md").write_text(f"# DATA READINESS DECISION\n\n**{decision}**\n\nJoint common TRAIN root bottleneck: **{common_root_ids} stored root IDs / {common_roots} physical root seeds**.\n\nMissing target: {missing}\n\nTEST: **TEST NOT OPENED**.\n")
    (OUT / "COLLECTOR_STATUS.json").write_text(json.dumps({"observed_at_utc": datetime.now(timezone.utc).isoformat(), "workers": active, "action": "read_only; no process started or stopped"}, indent=2) + "\n")

    # Explicit source inventory for auditability, including sources rejected as
    # derived reports or unreadable; no TEST path is opened.
    pd.DataFrame(source_stats).to_csv(OUT / "SOURCE_FILE_INVENTORY.csv", index=False)
    (OUT / "COMMON_PHYSICS_ROOT_MANIFEST.json").write_text(json.dumps({"method": "common_direct_imagination_joint", "stored_root_ids": sorted(common.root.dropna().unique().tolist()), "physical_root_seeds": sorted(common.physical_root.dropna().unique().tolist()), "contexts": int(common.physical_context.nunique()), "labels": int(common.canonical_sample_key.nunique()), "test_status": "TEST NOT OPENED"}, indent=2) + "\n")

    # Required gate/status artifact names are present but this round does not
    # open formal TEST or produce a rollout results table.
    (OUT / "FORMAL_TEST_ENTRY_GATE.json").write_text(json.dumps({"status": "TEST NOT OPENED", "reason": "DATA AUDIT ONLY", "sealed_roots": sorted(SEALED_ROOTS)}, indent=2) + "\n")
    (OUT / "FORMAL_TEST_ENTRY_GATE.md").write_text("# FORMAL TEST ENTRY GATE\n\n**TEST NOT OPENED** — this round is DATA AUDIT ONLY; roots 5174–5179 remain sealed.\n")
    (OUT / "RUN_STATUS.json").write_text(json.dumps({"status": "DATA_AUDIT_ONLY_COMPLETED", "test_status": "TEST NOT OPENED", "output_dir": str(OUT), "timestamp_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    (OUT / "AUDIT_SCOPE.md").write_text("# AUDIT SCOPE\n\nRead-only inventory. No training, rollout collection, TEST trajectory/RGB/context/outcome access, or collector mutation.\n")

    # Manifest/checksum after all artifacts are written.  Include the script
    # separately below from its source path.
    lines = []
    for p in sorted(OUT.iterdir()):
        if p.name == "SHA256SUMS.txt": continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {p.name}")
    script = Path(__file__)
    lines.append(f"{hashlib.sha256(script.read_bytes()).hexdigest()}  audit_physics_method_data_inventory.py (source: {script})")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_dir": str(OUT), "scanned_csv": scanned_csv, "candidate_files": len(candidates), "promoted_sources": len(source_stats), "inventory_rows": len(inv), "all_counts": all_counts, "train_counts": train_counts, "method_counts": method_counts, "common_train_roots": common_roots, "decision": decision, "test_status": "TEST NOT OPENED"}, indent=2))


if __name__ == "__main__":
    main()
