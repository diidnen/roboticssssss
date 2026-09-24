#!/usr/bin/env python3
"""Assemble the final ActiveForcing-Direct closure bundle from frozen evidence.

This is an evidence assembler, not a method-training script.  It reads the
authoritative 720-row TRAIN archive and already-frozen OOF/transfer artifacts,
recomputes the audit counts and basic aggregate tables, and writes a new,
timestamped closure directory.  It never reads the sealed challenge outcomes
or changes a frozen checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FORTE = Path("/home/exouser/FORTE")
TABERO = Path("/home/exouser/Tabero")
ARCHIVE = FORTE / "gnp_style_continuous_20260830_125107"
DATA = ARCHIVE / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv"
CONTEXT_AUDIT = ARCHIVE / "CONTINUOUS_TRAIN_CONTEXT_AUDIT.json"
PROBE = FORTE / "activeforcing_probe_conditioned_wm_20260901_064627"
TRANSFER = FORTE / "activeforcing_shared_physical_transfer_20260901_094722"
JOINT = FORTE / "joint_mechanism_20260831/pooled_matched"
PROTOCOL_SOURCE = FORTE / "hidden_friction_baseline_20260831/ACTIVEFORCING_DIRECT_SELECTOR_CONTRACT.md"
PRETEST_SOURCE = FORTE / "hidden_friction_baseline_20260831/ACTIVEFORCING_DIRECT_PRETEST_FREEZE.md"
P5_MANIFEST = TABERO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_BRANCH_MANIFEST.csv"
P5_CONTEXT = TABERO / "analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542/P5S0C_CONTEXT_MANIFEST.csv"

TASKS = [0, 1, 5, 6]
DIRECT_GRID = [3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00]
NO_PROBE_PRIOR = [0.30, 0.56, 0.92]
SEEDS = [0, 1, 2]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        w.writerows(rows)


def cmd(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"ERROR: {e!r}"


def make_out() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = FORTE / f"activeforcing_final_closure_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def audit(out: Path, df: pd.DataFrame) -> dict:
    required = {"branch_id", "context_id", "root_id", "task", "friction_band", "friction",
                "stratum_index", "repeat", "requested_force_N", "valid", "full_task_success_y",
                "state_parity", "strict_preprobe_hash", "branch_snapshot_hash", "nominal_motion_hash",
                "telemetry_path"}
    missing = sorted(required - set(df.columns))
    task_counts = {str(k): int(v) for k, v in df.task.value_counts().sort_index().items()}
    ctx_counts = df.groupby("context_id").size()
    root_counts = df.groupby("root_id").size()
    force_counts = df.groupby(["context_id", "requested_force_N"]).size()
    context_root = df.groupby("context_id").root_id.nunique()
    context_task = df.groupby("context_id").task.nunique()
    context_probe_hash = df.groupby("context_id").strict_preprobe_hash.nunique()
    context_branch_hash = df.groupby("context_id").branch_snapshot_hash.nunique()
    context_motion_hash = df.groupby("context_id").nominal_motion_hash.nunique()
    train_roots = sorted(df.root_id.astype(str).unique())
    # Three deterministic folds.  The key is the task-specific physical root;
    # the same root seed in another task is not assumed to be the same state.
    split_rows = []
    for rid in train_roots:
        seed = int(str(rid).rsplit("_s", 1)[1])
        fold = seed % 3
        q = df[df.root_id.astype(str).eq(rid)].iloc[0]
        split_rows.append({"root_id": rid, "task": int(q.task), "root_seed": seed,
                           "fold": fold, "split": "OOF_TEST_FOLD", "sibling_branches_kept_together": 1})
    split_rows.sort(key=lambda x: (x["fold"], x["task"], x["root_seed"]))
    write_json(out / "ACTIVEFORCING_FINAL_SPLIT_MANIFEST.json", {
        "status": "FROZEN_ROOT_HELDOUT_OOF",
        "unit": "task-specific physical root family; all friction contexts, force cells, and repeats stay together",
        "fold_rule": "root_seed % 3",
        "folds": {str(f): [r["root_id"] for r in split_rows if r["fold"] == f] for f in range(3)},
        "rows": split_rows,
        "source_dataset": str(DATA), "source_dataset_sha256": sha(DATA),
    })
    force_grid_by_task = {
        str(t): sorted({round(float(x), 12) for x in df.loc[df.task == t, "requested_force_N"]})
        for t in TASKS
    }
    ctx_audit = json.loads(CONTEXT_AUDIT.read_text()) if CONTEXT_AUDIT.exists() else {}
    context_hash_ok = bool(len(ctx_counts) == 72 and ctx_counts.eq(10).all() and context_root.eq(1).all()
                           and context_task.eq(1).all() and context_probe_hash.eq(1).all()
                           and context_branch_hash.eq(1).all() and context_motion_hash.eq(1).all())
    audit_payload = {
        "status": "PASS_WITH_FORCE_GRID_GAP" if not missing and context_hash_ok else "FAIL",
        "authoritative_dataset": str(DATA), "dataset_sha256": sha(DATA),
        "rows": int(len(df)), "valid_rows": int(df.valid.sum()), "state_parity_rows": int(df.state_parity.sum()),
        "successes": int(df.full_task_success_y.sum()), "failures": int(len(df) - df.full_task_success_y.sum()),
        "tasks": TASKS, "task_counts": task_counts,
        "contexts": int(df.context_id.nunique()), "root_families": int(df.root_id.nunique()),
        "branches_per_context": sorted(set(int(x) for x in ctx_counts)),
        "force_cells_per_context": sorted(set(int(x) for x in force_counts)),
        "repeats_per_force_cell": sorted(set(int(x) for x in force_counts)),
        "force_grid_by_task_observed": force_grid_by_task,
        "continuous_strata": int(df.stratum_index.nunique()),
        "strict_preprobe_shared_within_context": bool(context_probe_hash.eq(1).all()),
        "branch_snapshot_shared_within_context": bool(context_branch_hash.eq(1).all()),
        "nominal_motion_shared_within_context": bool(context_motion_hash.eq(1).all()),
        "same_query_before_force_branching": True,
        "query_semantics": "archive branches are post-P4-B query branches; offline no-physical row is Query-Ignored, not true zero-query",
        "task1_label_caveat": "140/180 task1 terminal labels are reconstructed in the authoritative lineage; retain caveat in primary outcome tables",
        "missing_required_columns": missing,
        "context_audit_source": str(CONTEXT_AUDIT),
        "context_audit_status": ctx_audit.get("status", "unknown"),
        "frozen_direct_grid": DIRECT_GRID,
        "force_grid_gap": "720 archive has five context-specific continuous stratum draws, not the frozen 0.25N 3.00–5.00 Direct candidate grid; no nearest-force substitution is allowed",
        "sealed_challenge": str(FORTE / "activeforcing_final_experiment_20260901_045000"),
        "sealed_challenge_read": False,
        "root_split": "3-fold grouped root-heldout; siblings never cross fold",
    }
    write_json(out / "ACTIVEFORCING_FINAL_DATA_GAPS.json", {
        "status": "GAPS_IDENTIFIED",
        "gaps": [
            {"id": "FIXED_GRID_NOT_ARCHIVED", "severity": "material", "impact": "Exact frozen 9-point Direct policy cannot be evaluated on all archived candidates without new matched branches", "allowed_action": "report archive-compatible continuous-stratum offline results; do not nearest-force substitute"},
            {"id": "TRUE_NO_QUERY_NOT_IDENTIFIABLE_OFFLINE", "severity": "material", "impact": "All 720 branches share a P4-B query before branching", "allowed_action": "reserve true NoQuery-Prior for fresh E2E"},
            {"id": "TASK1_RECONSTRUCTED_TERMINALS", "severity": "material", "impact": "140/180 task1 terminal outcomes are reconstructed", "allowed_action": "retain in primary with explicit caveat and direct-label sensitivity"},
            {"id": "VISUAL_COVERAGE", "severity": "minor", "impact": "visual embeddings are available for current TRAIN contexts, but not every historical context lineage", "allowed_action": "use existing matched visual OOF experiment only; no recollection"},
        ],
        "supplementary_collection_trigger": "only if exact frozen-grid E1 or true NoQuery/E5 cannot be answered from existing trajectories; never recollect 720",
    })
    md = f"""# ActiveForcing final data audit

Status: **{audit_payload['status']}**

## Authoritative archive

- Source: `{DATA}`
- SHA-256: `{audit_payload['dataset_sha256']}`
- Rows: **{audit_payload['rows']}**; valid **{audit_payload['valid_rows']}**; state parity **{audit_payload['state_parity_rows']}**
- Tasks: `{TASKS}`; task counts: `{task_counts}`
- Contexts: **{audit_payload['contexts']}**; task-specific root families: **{audit_payload['root_families']}**
- Each context: **5** force cells × **2** repeats; all contexts have one strict pre-probe hash, branch snapshot hash, and nominal-motion hash.

The 720 rows are post-query force branches. Therefore the offline no-information row is named **Query-Ignored / No-Physical-Information**. It is not reported as a true zero-query rollout.

## Frozen-method compatibility

The current frozen Direct selector uses the 0.25 N grid `{DIRECT_GRID}` and threshold 0.5 with max-force fallback. The authoritative 720 archive contains five continuous, context-specific stratum draws per context. Its observed force supports are recorded in `ACTIVEFORCING_FINAL_DATA_GAPS.json`; no nearest-force replacement is used.

Task1 retains the lineage caveat: 140/180 terminal labels are reconstructed. Exact label sensitivity is retained from the existing transfer artifact.

## Split integrity

The accompanying JSON assigns each task-specific root family to exactly one of three root-heldout folds. All sibling force branches, friction contexts, and repeats remain together.
"""
    (out / "ACTIVEFORCING_FINAL_DATA_AUDIT.md").write_text(md, encoding="utf-8")
    return audit_payload


def protocol(out: Path, audit_payload: dict) -> dict:
    p = {
        "status": "FROZEN_FOR_CLOSURE",
        "core_method": "ACTIVEFORCING_DIRECT",
        "tasks": TASKS,
        "archive": {"rows": 720, "contexts": 72, "root_families": 24, "source": str(DATA), "sha256": sha(DATA)},
        "split": {"name": "ROOT_HELDOUT", "folds": 3, "group": "task-specific root family", "manifest": "ACTIVEFORCING_FINAL_SPLIT_MANIFEST.json"},
        "query": {"name": "P4-B", "count": 1, "sequence_shape": [215, 46], "fixed_before_branching": True},
        "identifier": {"type": "point_estimate", "source": "frozen FRICTION_GRU", "uncertainty": "diagnostic sigma only", "no_ensemble_claim": True},
        "direct": {"architecture": "Shared Direct full-task feasibility model; 17-step command + 54D condition", "candidate_grid_N": DIRECT_GRID, "decision": "minimum candidate with ensemble mean p_success >= 0.5", "fallback": "5.00 N when no candidate passes", "controller": "selected force only as low-level setpoint; unchanged frozen controller law"},
        "utility": {"equation": "U(F|x)=p_success(F|x)*(Fmax-F)/Fmax + (1-p_success(F|x))*(-1)", "Fmax": "task/archive maximum; frozen Direct runtime uses candidate-grid max", "tie_break": "lower force", "future_outcome_access": False},
        "baselines": ["Frozen VLA Default", "Fixed-Max/Fixed-Robust", "Query-Ignored/No-Physical-Information", "ActiveForcing-Direct", "GT-Physics + same Direct + same utility", "Hindsight Grid Oracle (diagnostic only)"],
        "labels": {"primary": "full_task_success_y", "local_lift": "recovered from trajectory phase reaching post-lift transit/placement when available", "delayed_failure": "lift_success=1 AND full_task_success=0 AND downstream grip-related failure", "task1": "140/180 reconstructed terminal labels retained and flagged"},
        "metrics": ["full_task_success_rate", "selected_force_N", "realized_force_N", "under_force_rate", "excess_force_N", "delayed_failure_rate", "failure_stage", "query_count/cost only when semantically valid", "utility/regret diagnostic"],
        "statistics": {"cluster_unit": "physical root family", "ci": "95% cluster bootstrap", "paired_comparison": "same task/root/friction/repeat/initial condition", "seeds": SEEDS, "model_selection": "train-only / grouped OOF; no test tuning"},
        "additional_task_policy": "recover existing benchmark tasks first; mass/multi-property is optional and not required for core closure",
        "source_freezes": {"selector_contract": str(PROTOCOL_SOURCE), "selector_contract_sha256": sha(PROTOCOL_SOURCE), "pretest_freeze": str(PRETEST_SOURCE), "pretest_freeze_sha256": sha(PRETEST_SOURCE)},
        "data_gap_reference": "ACTIVEFORCING_FINAL_DATA_GAPS.json",
    }
    write_json(out / "ACTIVEFORCING_FINAL_PROTOCOL.json", p)
    (out / "ACTIVEFORCING_FINAL_PROTOCOL.md").write_text("""# ActiveForcing-Direct final protocol

**Frozen status: `FROZEN_FOR_CLOSURE`**

The final method is **Frozen VLA → one fixed P4-B diagnostic query → point friction estimate → Shared Direct full-task feasibility model → frozen force candidates → expected-utility selection → unchanged low-level controller**.

The machine-readable contract is in `ACTIVEFORCING_FINAL_PROTOCOL.json`. The authoritative archive and its compatibility gaps are in `ACTIVEFORCING_FINAL_DATA_AUDIT.md` and `ACTIVEFORCING_FINAL_DATA_GAPS.json`.

No final test result may alter the query, candidate grid, threshold, fallback, utility, split, or controller semantics.
""", encoding="utf-8")
    return p


def e2_identification(out: Path) -> None:
    src = pd.read_csv(TRANSFER / "PHYSICS_ESTIMATOR_LOTO.csv")
    # Keep the source rows visible and add an aggregate table with the exact
    # metrics requested by the closure protocol.
    rows = []
    for method, g in src.groupby("method"):
        q = g[g.scope.astype(str).eq("TASK")]
        if not len(q):
            continue
        rows.append({"experiment": "E2_PHYSICAL_IDENTIFICATION", "split": "LOTO_TASK_OBJECT_HELDOUT", "method": method,
                     "seeds_or_members": int(q.seed.nunique()), "contexts": int(q.contexts.sum()),
                     "friction_MAE": float(q.MAE.mean()), "median_absolute_error": float(q.MAE.median()),
                     "RMSE": float(q.RMSE.mean()), "Spearman": float(q.Spearman.mean()),
                     "pairwise_friction_ranking_accuracy": float(q.pair_ranking.mean()),
                     "force_choice_agreement": math.nan, "downstream_selected_force_regret": math.nan,
                     "downstream_success": math.nan, "source_artifact": str(TRANSFER / "PHYSICS_ESTIMATOR_LOTO.csv")})
    # The existing pooled OOF Probe result is the primary root-heldout point
    # estimate.  Include it as a separate, not conflated, row.
    p = pd.read_csv(PROBE / "POOLED_OOF_PROBE_PREDICTIONS.csv")
    if len(p):
        rows.append({"experiment": "E2_PHYSICAL_IDENTIFICATION", "split": "ROOT_HELDOUT_OOF", "method": "Probe-PhysicalHistory",
                     "seeds_or_members": int(p.seed.nunique()), "contexts": int(p.context_id.nunique()),
                     "friction_MAE": float(p.groupby("context_id").mu_hat.mean().sub(p.groupby("context_id").mu_GT.first()).abs().mean()),
                     "median_absolute_error": float(p.groupby("context_id").apply(lambda g: abs(g.mu_hat.mean()-g.mu_GT.iloc[0]), include_groups=False).median()),
                     "RMSE": float(np.sqrt(np.mean((p.mu_hat.to_numpy(float)-p.mu_GT.to_numpy(float))**2))),
                     "Spearman": float(pd.Series(p.mu_GT).corr(pd.Series(p.mu_hat), method="spearman")),
                     "pairwise_friction_ranking_accuracy": math.nan, "force_choice_agreement": math.nan,
                     "downstream_selected_force_regret": math.nan, "downstream_success": math.nan,
                     "source_artifact": str(PROBE / "POOLED_OOF_PROBE_PREDICTIONS.csv")})
    write_csv(out / "TABLE_PHYSICAL_IDENTIFICATION.csv", rows)
    src.to_csv(out / "PHYSICAL_IDENTIFICATION_FULL_RESULTS.csv", index=False)
    write_json(out / "PHYSICAL_IDENTIFICATION_FULL_RESULTS.json", {"status": "COMPLETE_FROM_EXISTING_MATCHED_OOF", "rows": rows, "source_sha256": sha(TRANSFER / "PHYSICS_ESTIMATOR_LOTO.csv"), "visual_ablation_source": str(PROBE / "PROBE_SCALAR_VS_RICH_DIRECT.csv")})
    lines = ["# Physical identification report", "", "The frozen physical-history estimator has been re-summarized from the existing grouped-root OOF artifact. Vision-only and vision+physical are retained only where the existing matched visual artifact exists; no full-task recollection was performed.", "", "| split | method | contexts | MAE | RMSE | Spearman | pair ranking |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        pair = r["pairwise_friction_ranking_accuracy"]
        pair_text = "NA" if math.isnan(pair) else f"{pair:.4f}"
        lines.append(f"| {r['split']} | {r['method']} | {r['contexts']} | {r['friction_MAE']:.4f} | {r['RMSE']:.4f} | {r['Spearman']:.4f} | {pair_text} |")
    lines += ["", "Caveat: LOTO is task/object-family heldout because the archive confounds task and object family. The current frozen estimator is a point estimate; sigma is diagnostic only."]
    (out / "TABLE_PHYSICAL_IDENTIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "PHYSICAL_IDENTIFICATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def e1_main(out: Path) -> None:
    d = pd.read_csv(PROBE / "PROBE_SCALAR_VS_RICH_DIRECT.csv")
    rows = []
    for method, g in d.groupby("method"):
        r = {"experiment": "E1_720_ARCHIVE_PAIRED", "method": method, "scope": "72 contexts × 2 repeats", "n": len(g),
             "full_task_success_rate": float(g.success.mean()), "mean_selected_force_N": float(g.selected_force_N.mean()),
             "under_force_rate": float(g.under_force.mean()), "mean_excess_force_N": float(g.excess_force_N.mean()),
             "mean_regret": float(g.regret.mean()), "mean_realized_utility": float(g.realized_utility.mean()),
             "task1_reconstructed_label_caveat": "140/180 terminal labels reconstructed", "source": str(PROBE / "PROBE_SCALAR_VS_RICH_DIRECT.csv")}
        rows.append(r)
        for task, q in g.groupby("task"):
            rows.append({**r, "method": method, "scope": f"task{int(task)}", "n": len(q),
                         "full_task_success_rate": float(q.success.mean()), "mean_selected_force_N": float(q.selected_force_N.mean()),
                         "under_force_rate": float(q.under_force.mean()), "mean_excess_force_N": float(q.excess_force_N.mean()),
                         "mean_regret": float(q.regret.mean()), "mean_realized_utility": float(q.realized_utility.mean())})
    # Archive-native baselines.  These are evaluated on the exact 720 rows,
    # with the observed five force cells per context; no interpolation or
    # nearest-grid replacement is performed.
    raw = pd.read_csv(ARCHIVE / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv")
    base_rows = []
    for (cid, rep), g in raw.groupby(["context_id", "repeat"], sort=True):
        g = g.sort_values("requested_force_N")
        mx = g.iloc[-1]
        success_rows = g[g.full_task_success_y.astype(int).eq(1)]
        oracle = success_rows.iloc[0] if len(success_rows) else None
        for name, q, status in [("Fixed-Max", mx, "available"),
                                ("Hindsight-Grid-Oracle", oracle, "available" if oracle is not None else "no_success"),
                                ("Frozen-VLA-Default", None, "not_identifiable_on_720_archive")]:
            base_rows.append({"experiment": "E1_720_ARCHIVE_PAIRED", "method": name,
                              "scope": "archive-compatible paired episodes", "n": 1,
                              "context_id": cid, "repeat": int(rep), "status": status,
                              "full_task_success_rate": (float(q.full_task_success_y) if q is not None else math.nan),
                              "mean_selected_force_N": (float(q.requested_force_N) if q is not None else math.nan),
                              "under_force_rate": math.nan, "mean_excess_force_N": math.nan,
                              "mean_regret": math.nan, "mean_realized_utility": math.nan,
                              "task1_reconstructed_label_caveat": "140/180 terminal labels reconstructed",
                              "source": str(ARCHIVE / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv")})
    write_csv(out / "E1_ARCHIVE_BASELINE_PER_EPISODE.csv", base_rows)
    for name in ["Fixed-Max", "Hindsight-Grid-Oracle", "Frozen-VLA-Default"]:
        q = pd.DataFrame([r for r in base_rows if r["method"] == name])
        if name == "Frozen-VLA-Default":
            rows.append({"experiment": "E1_720_ARCHIVE_PAIRED", "method": name, "scope": "archive-compatible paired episodes", "n": 0,
                         "status": "NOT_IDENTIFIABLE_ON_720_ARCHIVE", "full_task_success_rate": math.nan,
                         "mean_selected_force_N": math.nan, "under_force_rate": math.nan, "mean_excess_force_N": math.nan,
                         "mean_regret": math.nan, "mean_realized_utility": math.nan,
                         "task1_reconstructed_label_caveat": "140/180 terminal labels reconstructed", "source": str(ARCHIVE / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv")})
        else:
            q = q[q.status.eq("available")]
            rows.append({"experiment": "E1_720_ARCHIVE_PAIRED", "method": name, "scope": "archive-compatible paired episodes", "n": len(q),
                         "status": "available", "full_task_success_rate": float(q.full_task_success_rate.mean()),
                         "mean_selected_force_N": float(q.mean_selected_force_N.mean()), "under_force_rate": math.nan,
                         "mean_excess_force_N": math.nan, "mean_regret": math.nan, "mean_realized_utility": math.nan,
                         "task1_reconstructed_label_caveat": "140/180 terminal labels reconstructed", "source": str(ARCHIVE / "CONTINUOUS_TRAIN_SUCCESS_DATA.csv")})
    # Direct contrasts are computed only on paired rows present in the same
    # current archive artifact.  No branch is treated as independent in CI.
    piv = d.pivot_table(index=["context_id", "repeat"], columns="method", values="success")
    contrasts = []
    for a, b, label in [("ProbeScalar-Direct", "NoProbe-Direct", "Direct-NoPhysicalInfo"),
                        ("ProbeScalar-Direct", "GT-Direct", "GTPhysics-Direct")]:
        if a in piv and b in piv:
            z = piv[a].dropna().to_frame("a").join(piv[b].dropna().rename("b"), how="inner")
            contrasts.append({"experiment": "E1_CONTRAST", "method": label, "scope": "paired episodes", "n": len(z),
                              "full_task_success_rate": float((z.a-z.b).mean()), "mean_selected_force_N": math.nan,
                              "under_force_rate": math.nan, "mean_excess_force_N": math.nan, "mean_regret": math.nan, "mean_realized_utility": math.nan,
                              "paired_difference": float((z.a-z.b).mean()), "wins_a_minus_b": int((z.a>z.b).sum()), "wins_b_minus_a": int((z.b>z.a).sum())})
    rows += contrasts
    write_csv(out / "TABLE_MAIN_FORCE_ADAPTATION.csv", rows)
    lines = ["# Main 720 paired force-adaptation benchmark", "", "Archive-compatible offline results are summarized from the frozen grouped-root OOF selector artifact. Probe rows use the point friction estimate; GT-Direct is offline oracle; NoProbe-Direct is Query-Ignored and is not zero-query.", "", "| method | scope | n | SR | mean force (N) | under-force | excess (N) | utility |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["experiment"] != "E1_720_ARCHIVE_PAIRED": continue
        lines.append(f"| {r['method']} | {r['scope']} | {r['n']} | {r['full_task_success_rate']:.4f} | {r['mean_selected_force_N']:.3f} | {r['under_force_rate']:.4f} | {r['mean_excess_force_N']:.3f} | {r['mean_realized_utility']:.4f} |")
    (out / "TABLE_MAIN_FORCE_ADAPTATION.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(out / "MAIN_PAIRED_BENCHMARK_RESULTS.json", {"status": "COMPLETE_ARCHIVE_COMPATIBLE_WITH_GRID_GAP", "rows": rows, "source": str(PROBE / "PROBE_SCALAR_VS_RICH_DIRECT.csv"), "cluster_unit": "root family", "sealed_test_read": False})
    (out / "MAIN_PAIRED_BENCHMARK_REPORT.md").write_text("\n".join(lines)+"\n\nExact frozen 0.25N grid evaluation remains a data gap because the 720 archive contains continuous context-specific force draws. No nearest-force substitution was used.\n", encoding="utf-8")


def e4_transfer(out: Path) -> None:
    src = TRANSFER / "NEW_TASK_FEWSHOT_TRANSFER_AGG.csv"
    d = pd.read_csv(src)
    d.to_csv(out / "TABLE_SHARED_TRANSFER.csv", index=False)
    cols = list(d.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False, name=None):
        md.append("| " + " | ".join(str(x) for x in row) + " |")
    (out / "TABLE_SHARED_TRANSFER.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    report = (TRANSFER / "FINAL_SHARED_PHYSICAL_TRANSFER_REPORT.md").read_text(encoding="utf-8")
    (out / "SHARED_TRANSFER_FINAL_REPORT.md").write_text("# Shared transfer final report\n\n" + report + "\n\nSource reused without rerun: `" + str(src) + "`.\n", encoding="utf-8")


def task_recovery(out: Path) -> None:
    task_names = {0: "alphabet soup → basket", 1: "cream cheese → basket", 5: "tomato sauce → basket", 6: "butter → basket"}
    rows = []
    for t in TASKS:
        rows.append({"task": t, "task_name": task_names[t], "status": "EXISTING_AND_READY", "role": "current hidden-friction core", "evidence": str(P5_CONTEXT)})
    rows += [
        {"task": "task groups", "task_name": "short pick-place / long transport / turning-acceleration / precise placement", "status": "EXISTING_BUT_NOT_CURRENTLY_USED", "role": "taxonomy recovered from force-critical planning and LIBERO assets", "evidence": "FORTE planning notes; no claim of completed breadth result"},
        {"task": "mass extension", "task_name": "mass-sensitive or friction+mass joint physics", "status": "PLANNED_BUT_NOT_IMPLEMENTED", "role": "optional extension; not core closure", "evidence": "current frozen protocol excludes mass"},
        {"task": "new proposal", "task_name": "none", "status": "NEW_PROPOSAL", "role": "none introduced", "evidence": "no new task proposed"},
    ]
    write_csv(out / "ADDITIONAL_TASK_CANDIDATES.csv", rows)
    (out / "TASK_BENCHMARK_RECOVERY.md").write_text("""# Task benchmark recovery

## Existing and ready

The authoritative current set is task0 alphabet soup, task1 cream cheese, task5 tomato sauce, and task6 butter, each placed in a basket. They are the hidden-friction core and have current P4-B/probe/full-task data.

## Existing but not currently used

Historical planning separates short pick-place, long post-lift transport, turning/acceleration, and constrained placement. Existing force-critical/LIBERO-style assets were audited, but no additional task is promoted into the core closure without a fresh qualification table.

## Planned but not implemented

Mass-sensitive and friction+mass joint-physics tasks remain optional extensions. They are not inserted into the main friction paper.

## New proposal

None. No planned task is described as completed.
""", encoding="utf-8")
    (out / "ADDITIONAL_TASK_SELECTION_REPORT.md").write_text("""# Additional task selection report

No secondary breadth task is promoted in this closure bundle. The recovered candidates are recorded, but the inclusion criteria require nonzero frozen-VLA reach, friction sensitivity, force sensitivity, and preferably delayed failure. Existing artifacts do not provide a clean, current-Direct, paired qualification table for an extra task without a new collection. This is a documented data gap, not a positive breadth claim.
""", encoding="utf-8")


def reproducibility(out: Path, audit_payload: dict) -> None:
    env = {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "python": cmd("python3", "--version"), "torch": cmd("python3", "-c", "import torch; print(torch.__version__)"), "cuda": cmd("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")}
    write_json(out / "REPRODUCIBILITY_BUNDLE.json", {"environment": env, "git": {"forte_commit": cmd("git", "-C", str(FORTE), "rev-parse", "HEAD"), "tabero_commit": cmd("git", "-C", str(TABERO), "rev-parse", "HEAD"), "forte_status": cmd("git", "-C", str(FORTE), "status", "--short"), "tabero_status": cmd("git", "-C", str(TABERO), "status", "--short")}, "dataset_sha256": audit_payload["dataset_sha256"], "seed": SEEDS, "commands": ["python3 /home/exouser/FORTE/assemble_activeforcing_closure.py"], "sealed_test_read": False})
    (out / "ENGINEERING_FIX_LOG.md").write_text("""# Engineering fix log

No scientific method change was made by the closure assembler. The only engineering action is aggregation/validation of existing artifacts into a timestamped bundle. Existing historical fixes remain in the source experiment directories.

| symptom | root cause | change | scientific-method impact | validation |
|---|---|---|---|---|
| required closure files absent | prior experiments wrote scoped artifacts only | recompute/assemble auditable tables and manifests | none | source hashes and row-count checks recorded |
| archive/grid mismatch | 720 collection sampled continuous force strata; frozen Direct uses 0.25N grid | record data gap; do not nearest-force substitute | preserves frozen method | audit reports exact observed supports |
""", encoding="utf-8")


def main() -> None:
    out = make_out()
    df = pd.read_csv(DATA)
    ap = audit(out, df)
    protocol(out, ap)
    e2_identification(out)
    e1_main(out)
    e4_transfer(out)
    task_recovery(out)
    # Required placeholders are explicit only where the current evidence is
    # not semantically sufficient; later stages may replace them with actual
    # results without overwriting this directory.
    (out / "TABLE_FULLTASK_VS_LOCALLIFT.csv").write_text("experiment,status,reason\nE3,IN_PROGRESS,matched LocalLift training/evaluation still running\n", encoding="utf-8")
    (out / "TABLE_FULLTASK_VS_LOCALLIFT.md").write_text("# FullTask vs LocalLift\n\nStatus: **IN_PROGRESS**. No result is claimed until matched label-only training completes.\n", encoding="utf-8")
    (out / "FULLTASK_LOCALLIFT_MATCHED_REPORT.md").write_text("# FullTask vs LocalLift matched report\n\nStatus: IN_PROGRESS.\n", encoding="utf-8")
    (out / "DELAYED_FAILURE_CASES.csv").write_text("status,reason\nIN_PROGRESS,awaiting model-independent stage recovery\n", encoding="utf-8")
    (out / "TABLE_FRESH_E2E.csv").write_text("experiment,status,reason\nE5,IN_PROGRESS,existing online evidence is task0/two-method only; current Direct all-task paired E2E not yet complete\n", encoding="utf-8")
    (out / "TABLE_FRESH_E2E.md").write_text("# Fresh E2E\n\nStatus: **IN_PROGRESS**.\n", encoding="utf-8")
    (out / "FRESH_E2E_RESULTS.json").write_text(json.dumps({"status": "IN_PROGRESS", "reason": "current all-task paired E2E still requires execution", "existing": str(FORTE / "hidden_friction_baseline_20260831/TWO_METHOD_FORMAL_RUNTIME_RERUN1")}, indent=2)+"\n", encoding="utf-8")
    (out / "FRESH_E2E_REPORT.md").write_text("# Fresh E2E report\n\nStatus: IN_PROGRESS; no fresh-E2E claim is made from the task0-only predecessor.\n", encoding="utf-8")
    (out / "FINAL_FAILURE_TAXONOMY.csv").write_text("taxonomy,status\nUPSTREAM_VLA_FAILURE,IN_PROGRESS\nQUERY_REACH_FAILURE,IN_PROGRESS\nQUERY_INVALID,IN_PROGRESS\nPHYSICS_IDENTIFICATION_ERROR,IN_PROGRESS\nUNDER_FORCE,IN_PROGRESS\nPOST_LIFT_DROP,IN_PROGRESS\nTRANSPORT_FAILURE,IN_PROGRESS\nPLACEMENT_FAILURE,IN_PROGRESS\nOVER_FORCE_WITHOUT_BENEFIT,IN_PROGRESS\nFEASIBILITY_MODEL_ERROR,IN_PROGRESS\nUTILITY_SELECTION_ERROR,IN_PROGRESS\nCONTROLLER_TRACKING_ERROR,IN_PROGRESS\nOTHER,IN_PROGRESS\n", encoding="utf-8")
    (out / "FINAL_FAILURE_ANALYSIS.md").write_text("# Final failure analysis\n\nStatus: IN_PROGRESS; taxonomy schema is frozen, case transitions will be populated from E3/E5 outputs.\n", encoding="utf-8")
    reproducibility(out, ap)
    print(json.dumps({"status": "PHASE_A_B_E1_E2_E4_ASSEMBLED", "output": str(out), "audit": ap}, indent=2))


if __name__ == "__main__":
    main()
