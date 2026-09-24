#!/usr/bin/env python3
"""Freeze task-specific visual-context models before task-specific DEV is read.

This generalizes the completed task0 early run to tasks 1, 5, and 6.  It is
CPU-only, reads TRAIN artifacts only, and writes outside the live Tabero
collection.  The task1 cumulative CSV was interrupted, so missing task1 labels
use the same terminal-height fallback frozen in prospective_visual_context_pipeline.py;
the label audit is always persisted and the downstream evaluator must run the
declared sensitivity analysis.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import task0_visual_context_early as early


SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
TRAIN_MANIFEST = SOURCE / "PROSPECTIVE_TRAIN_RUN_MANIFEST.csv"
CONTEXT_MANIFEST = SOURCE / "PROSPECTIVE_CONTEXT_MANIFEST.csv"
VISUAL_CSV = SOURCE / "collection_train/visual_alignment_worker.csv"
SUPPORTED_TASKS = (1, 5, 6)
TASK_NAMES = {0: "alphabet soup", 1: "cream cheese", 5: "tomato sauce", 6: "butter"}


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(newline="", encoding="utf-8"))) if path.exists() else []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, np.float32).tobytes()).hexdigest()


def telemetry_path(task: int, context_id: str, branch_label: str) -> Path | None:
    d = SOURCE / f"collection_train/task{task}/P5S0C_BRANCH_TELEMETRY"
    matches = sorted(d.glob(f"{context_id}_{branch_label}_F*_trajectory.csv"))
    return matches[-1] if matches else None


def fallback_success(td: pd.DataFrame) -> int:
    """Exact fallback frozen by prospective_visual_context_pipeline.py."""
    z = td["object_z_analysis_only"].to_numpy(float)
    return int(float(z[-1] - z[0]) > 0.005)


def audit_and_load(task: int, out: Path, tpi):
    context_rows = [r for r in read_csv(CONTEXT_MANIFEST) if r["split"] == "TRAIN" and int(r["task"]) == task]
    target_rows = [r for r in read_csv(TRAIN_MANIFEST) if int(r["task"]) == task]
    direct_context_rows = read_csv(SOURCE / f"collection_train/task{task}/task{task}/context.csv")
    direct_branch_rows = read_csv(SOURCE / f"collection_train/task{task}/task{task}/branches.csv")
    direct_context = {r["context_id"]: r for r in direct_context_rows}
    direct_branch = {(r["context_id"], r["branch_label"]): r for r in direct_branch_rows}
    visual_rows = [r for r in read_csv(VISUAL_CSV) if int(r["task"]) == task and r["context_id"] in {x["context_id"] for x in context_rows}]
    visual = {r["context_id"]: r for r in visual_rows}

    failures: list[str] = []
    warnings: list[str] = []
    if len(context_rows) != 18 or len({r["context_id"] for r in context_rows}) != 18:
        failures.append(f"contexts={len(context_rows)} unique={len({r['context_id'] for r in context_rows})}; expected 18")
    if len(target_rows) != 180 or len({(r["context_id"], r["branch_label"]) for r in target_rows}) != 180:
        failures.append(f"target branches={len(target_rows)}; expected 180 unique")
    if len(visual) != 18:
        failures.append(f"visual contexts={len(visual)}; expected 18")
    if len(direct_context_rows) != 18:
        warnings.append(f"cumulative context.csv contains {len(direct_context_rows)}/18 rows; frozen manifest + aligned captures control")
    if len(direct_branch_rows) != 180:
        warnings.append(f"cumulative branches.csv contains {len(direct_branch_rows)}/180 rows; frozen manifest + telemetry control")

    by_context: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in target_rows:
        by_context[r["context_id"]].append(r)
    for cid, rows in by_context.items():
        cells = Counter((int(r["stratum_index"]), int(r["repeat"])) for r in rows)
        if cells != Counter((s, rep) for s in range(5) for rep in (1, 2)):
            failures.append(f"{cid}: incomplete 5x2 stratum/repeat cells")

    cmap: dict[str, dict[str, Any]] = {}
    feature_rows: list[dict[str, Any]] = []
    raw_features: list[np.ndarray] = []
    for r in sorted(context_rows, key=lambda q: q["context_id"]):
        cid = r["context_id"]
        vr = visual.get(cid)
        if vr is None:
            failures.append(f"{cid}: missing aligned visual capture")
            continue
        paths = [Path(vr[k]) for k in ("camera0_rgb_path", "camera1_rgb_path", "visual_feature_path", "snapshot_path")]
        if not all(p.exists() for p in paths):
            failures.append(f"{cid}: missing visual/snapshot file")
            continue
        x = np.load(vr["visual_feature_path"], allow_pickle=False)
        if x.shape != (4096,) or x.dtype != np.float32:
            failures.append(f"{cid}: visual shape/dtype={x.shape}/{x.dtype}")
        if array_sha256(x) != vr["visual_feature_sha256"]:
            failures.append(f"{cid}: visual feature hash mismatch")
        if not (vr.get("restore_exact") == "1" and vr.get("restored_state_hash") == vr.get("snapshot_state_hash") and vr.get("second_restore_hash") == vr.get("snapshot_state_hash")):
            failures.append(f"{cid}: snapshot restore audit failed")
        dc = direct_context.get(cid)
        if dc and dc.get("post_probe_state_hash") != vr.get("snapshot_state_hash"):
            failures.append(f"{cid}: direct context/visual snapshot hash mismatch")
        probe = SOURCE / f"collection_train/task{task}/P5S0C_PROBE_TELEMETRY/{cid}_probe_timesteps.csv"
        if not probe.exists():
            failures.append(f"{cid}: missing probe telemetry")
            continue
        state, mask, premeta = early.strict_preprobe_state(probe)
        cmap[cid] = {
            "context_id": cid,
            "root_id": r["root_id"],
            "task": task,
            "friction": float(r["mu_GT"]),
            "friction_band": r["friction_band"],
            "preprobe_state": state,
            "preprobe_mask": mask,
            "visual_raw": np.asarray(x, np.float32),
        }
        raw_features.append(np.asarray(x, np.float32))
        feature_rows.append({
            "context_id": cid,
            "root_id": r["root_id"],
            "task": task,
            "visual_feature_path": vr["visual_feature_path"],
            "visual_feature_raw_sha256": array_sha256(x),
            "snapshot_state_hash": vr["snapshot_state_hash"],
            "restore_exact": int(vr["restore_exact"]),
            "probe_telemetry_path": str(probe),
            **premeta,
        })

    corrected = {
        "left_normal_force_N", "right_normal_force_N", "left_tangential_force_N",
        "right_tangential_force_N", "object_vx_mps", "object_vy_mps", "object_vz_mps",
    }
    traces = []
    meta: dict[str, early.Meta] = {}
    canonical_rows: list[dict[str, Any]] = []
    direct_fallback_checks: list[dict[str, Any]] = []
    for r in sorted(target_rows, key=lambda q: (q["context_id"], int(q["stratum_index"]), int(q["repeat"]))):
        cid, label = r["context_id"], r["branch_label"]
        path = telemetry_path(task, cid, label)
        if path is None:
            failures.append(f"{cid}/{label}: missing branch telemetry")
            continue
        d = pd.read_csv(path)
        required = corrected | {"phase", "object_z_analysis_only", "cmd_x", "cmd_y", "cmd_z"}
        if len(d) < early.H + 1 or not required <= set(d.columns) or not np.isfinite(d[list(corrected)].to_numpy(float)).all():
            failures.append(f"{cid}/{label}: invalid corrected telemetry")
            continue
        fallback = fallback_success(d)
        direct = direct_branch.get((cid, label))
        if direct is not None and direct.get("full_task_success_y", "") != "":
            outcome = int(float(direct["full_task_success_y"]))
            label_source = "DIRECT_CUMULATIVE_BRANCH_LABEL"
            direct_fallback_checks.append({
                "branch_id": direct["branch_id"], "context_id": cid, "task": task,
                "direct_label": outcome, "fallback_label": fallback,
                "agree": int(outcome == fallback), "trajectory_rows": len(d),
                "final_phase": str(d.iloc[-1]["phase"]),
            })
        else:
            outcome = fallback
            label_source = "FROZEN_PIPELINE_TERMINAL_HEIGHT_FALLBACK"

        state, mask = tpi.state_from(d)
        state, mask = state.copy(), mask.copy()
        state[0] = cmap[cid]["preprobe_state"]
        mask[0] = cmap[cid]["preprobe_mask"]
        force, mu = float(r["force_N"]), float(r["mu_GT"])
        nominal = tpi.nominal_from(d, 0, force, mu, state, mask)
        branch_id = f"{cid}_{label}"
        tr = tpi.Trace(
            branch_id, cid, r["root_id"], task, "TRAIN", force, mu, outcome,
            "continuous", path, state, mask, nominal, d.phase.astype(str).tolist(),
            1.0, f"PROSPECTIVE_TASK{task}",
        )
        traces.append(tr)
        meta[branch_id] = early.Meta(
            branch_id, cid, task, r["root_id"], r["friction_band"], outcome,
            force, int(r["repeat"]), int(r["stratum_index"]),
        )
        canonical_rows.append({
            "branch_id": branch_id, "context_id": cid, "task": task, "root_id": r["root_id"],
            "root_index": int(r["root_index"]), "friction_band": r["friction_band"],
            "mu_GT": mu, "stratum_index": int(r["stratum_index"]), "repeat": int(r["repeat"]),
            "force_N": force, "success": outcome, "label_source": label_source,
            "telemetry_path": str(path), "telemetry_sha256": sha256(path),
            "visual_snapshot_restore_exact": int(visual[cid]["restore_exact"]),
            "direct_branch_state_parity": int(direct.get("state_parity", 1)) if direct else "",
        })

    if len(traces) != 180:
        failures.append(f"valid traces={len(traces)}; expected 180")
    if len(cmap) != 18:
        failures.append(f"valid visual contexts={len(cmap)}; expected 18")
    if failures:
        early.write_json(out / f"TASK{task}_DATA_AUDIT.json", {"status": "FAIL", "failures": failures, "warnings": warnings})
        raise RuntimeError("task audit failed: " + "; ".join(failures[:6]))

    xraw = np.stack(raw_features).astype(np.float32)
    xmean = xraw.mean(0)
    _, singular, vt = np.linalg.svd(xraw - xmean, full_matrices=False)
    rank = min(64, len(cmap) - 1, int(np.sum(singular > singular[0] * 1e-7)))
    components = vt[:rank].astype(np.float32)
    xpca = ((xraw - xmean) @ components.T).astype(np.float32)
    pmean, pstd = xpca.mean(0).astype(np.float32), xpca.std(0).astype(np.float32)
    pstd[pstd < 1e-6] = 1.0
    for cid, z in zip(sorted(cmap), (xpca - pmean) / pstd):
        cmap[cid]["visual"] = z.astype(np.float32)

    early.write_csv(out / f"TASK{task}_FROZEN_VISUAL_ALIGNMENT.csv", feature_rows)
    early.write_csv(out / f"TASK{task}_CANONICAL_TRAIN_BRANCHES.csv", canonical_rows)
    early.write_csv(out / f"TASK{task}_DIRECT_VS_FALLBACK_LABEL_AUDIT.csv", direct_fallback_checks)
    pca_path = out / f"TASK{task}_PCA{rank}_TRAIN_ONLY.npz"
    np.savez(pca_path, raw_mean=xmean, components=components, projected_mean=pmean,
             projected_std=pstd, singular_values=singular)

    mismatches = sum(1 - int(r["agree"]) for r in direct_fallback_checks)
    reconstructed = sum(r["label_source"] != "DIRECT_CUMULATIVE_BRANCH_LABEL" for r in canonical_rows)
    label_risk = "NONE_DIRECT_LABELS_COMPLETE" if reconstructed == 0 else "MATERIAL_RECONSTRUCTED_LABEL_CAVEAT"
    audit = {
        "status": "PASS_WITH_CAVEAT" if reconstructed else "PASS",
        "scope": f"task{task} TRAIN only", "task": task, "object": TASK_NAMES[task],
        "contexts": 18, "branches": 180,
        "successes": int(sum(r["success"] for r in canonical_rows)),
        "failures": int(180 - sum(r["success"] for r in canonical_rows)),
        "direct_labels": 180 - reconstructed, "reconstructed_labels": reconstructed,
        "direct_vs_fallback_compared": len(direct_fallback_checks),
        "direct_vs_fallback_mismatches": mismatches,
        "label_risk": label_risk,
        "task1_required_sensitivity": bool(task == 1 and reconstructed),
        "task1_sensitivity_rule": "final classification must be stable to plausible correction of rare long-episode transient basket contacts; primary frozen label remains the preregistered pipeline fallback",
        "five_forces_two_repeats_each": True,
        "visual_alignment": "18/18", "snapshot_restore_exact": "18/18",
        "corrected_physical_telemetry": "180/180", "DEV_read": False, "TEST_read": False,
        "pca_requested_components": 64, "pca_effective_components": rank,
        "pca_deviation_reason": "18 task-specific contexts imply centered PCA rank at most 17",
        "warnings": warnings,
        "source_hashes": {
            "PROSPECTIVE_CONTEXT_MANIFEST.csv": sha256(CONTEXT_MANIFEST),
            "PROSPECTIVE_TRAIN_RUN_MANIFEST.csv": sha256(TRAIN_MANIFEST),
            "visual_alignment_worker.csv": sha256(VISUAL_CSV),
        },
    }
    early.write_json(out / f"TASK{task}_DATA_AUDIT.json", audit)
    return cmap, traces, meta, rank, pca_path, canonical_rows, audit


def build_pairs(task: int, cf, traces, meta):
    by: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for tr in traces:
        by[(tr.context_id, meta[tr.branch_id].repeat)].append(tr)
    pairs = []
    for (cid, repeat), rows in sorted(by.items()):
        rows = sorted(rows, key=lambda tr: meta[tr.branch_id].stratum)
        if len(rows) != 5:
            raise RuntimeError(f"expected five strata: {cid}/R{repeat}")
        for a, b in zip(rows[:-1], rows[1:]):
            ma, mb = meta[a.branch_id], meta[b.branch_id]
            pairs.append(cf.Pair(
                f"task{task}:{cid}:R{repeat}:S{ma.stratum}_vs_S{mb.stratum}",
                f"task{task}:{cid}:R{repeat}", "TRAIN", f"PROSPECTIVE_TASK{task}", cid,
                a.root_id, task, ma.friction_band, a.mu, a.force, b.force, "adjacent",
                False, a, b,
            ))
    expected_pairs = len(by) * 4
    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"expected {expected_pairs} adjacent pairs from {len(by)} context-repeat groups, "
            f"got {len(pairs)}"
        )
    return pairs


def save_checkpoint(path: Path, model, variant: str, seed: int, task: int, norm,
                    pca_path: Path, extra: dict[str, Any] | None = None) -> None:
    obj = {
        "state_dict": model.state_dict(), "variant": variant, "seed": seed,
        "task": task, "scope": f"TASK{task}_TRAIN_ONLY_FROZEN_BEFORE_DEV",
        "epochs": early.EPOCHS, "device": "cpu",
        "normalization": {k: v.tolist() for k, v in zip(["x_mean", "x_std", "y_mean", "y_std"], norm)},
        "pca_path": str(pca_path), "pca_sha256": sha256(pca_path),
        "DEV_used": False, "TEST_used": False,
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def write_notebook(task: int, out: Path, summary: dict[str, Any]) -> None:
    def md(text: str):
        return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in text.splitlines()]}
    def code(source: str, output: str):
        return {"cell_type": "code", "execution_count": 1, "metadata": {},
                "source": [x + "\n" for x in source.splitlines()],
                "outputs": [{"output_type": "stream", "name": "stdout", "text": [output + "\n"]}]}
    audit = (out / f"TASK{task}_DATA_AUDIT.json").read_text(encoding="utf-8")
    metrics = pd.read_csv(out / f"TASK{task}_TRAIN_IN_SAMPLE_METRICS.csv").to_string(index=False)
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            md(f"# task{task} visual-context TRAIN freeze\n\nThis is TRAIN-only and cannot support a final scientific classification."),
            md("## Data and label quality\nThe task-specific audit checks frozen-manifest coverage, telemetry, visual alignment, PCA rank, and direct-versus-fallback labels."),
            code(f"import json\nprint(json.dumps(json.load(open('TASK{task}_DATA_AUDIT.json')), indent=2))", audit),
            md("## In-sample diagnostics\nThese values are retained only to measure later TRAIN-to-DEV gaps."),
            code(f"import pandas as pd\nprint(pd.read_csv('TASK{task}_TRAIN_IN_SAMPLE_METRICS.csv').to_string(index=False))", metrics),
            md("## Decision boundary\nNo model selection or visual/Joint conclusion is permitted until the frozen task-specific DEV evaluation completes."),
            code(f"summary = {json.dumps(summary, sort_keys=True)}\nprint(summary)", json.dumps(summary, indent=2)),
        ],
    }
    early.write_json(out / f"TASK{task}_EARLY_RUN.ipynb", nb)


def freeze_protocol(task: int, out: Path, rank: int, audit: dict[str, Any]) -> None:
    protocol = {
        "status": f"TASK{task}_TRAIN_ONLY_FROZEN_BEFORE_DEV",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_scientific_goal": f"test whether task{task} has task0-like visual context memorization versus held-out generalization, and whether x_visual×F or Joint supervision adds safe independent value",
        "source": str(SOURCE), "task": task, "object": TASK_NAMES[task],
        "split": "TRAIN", "contexts": 18, "branches": 180,
        "models": ["PROSPECTIVE_BASE_FEAS", "VISUAL_INTERCEPT_RESIDUAL", "VISUAL_CONTEXT_FULL_FEAS", "VISUAL_CONTEXT_JOINT"],
        "seeds": early.SEEDS, "epochs": early.EPOCHS, "optimizer": "AdamW",
        "lr": early.LR, "weight_decay": early.WEIGHT_DECAY,
        "loss": "same authoritative task0 architecture/loss family; Joint adds trajectory + IE and lambda_feas=0.3",
        "device": "CPU_ONLY", "torch_threads": 1, "launches_Isaac_or_pi0": False,
        "DEV_used": False, "TEST_used": False, "visual_pca_components": rank,
        "classification_thresholds": {
            "visual_probability_MAE_relative_improvement_min": 0.20,
            "frontier_MAE_improvement_min_N": 0.05,
            "under_force_nonworse": True,
            "monotonic_context_fraction_min": 0.90,
            "joint_probability_MAE_relative_improvement_min": 0.10,
            "joint_brier_ratio_max": 1.05,
            "multiple_DEV_context_benefit_required": True,
        },
        "single_context_task6_rule": "task6 has one frozen DEV context; report metrics and TRAIN root-heldout CV but do not claim multi-context visual generalization or independent Joint value from task6 alone",
        "task1_label_rule": {
            "primary": "same frozen terminal-height fallback as prospective_visual_context_pipeline.py for missing cumulative rows",
            "sensitivity_required": bool(task == 1 and audit["reconstructed_labels"]),
            "reason": "basket contact was not logged in branch telemetry; rare transient-contact successes can finish at low terminal height",
        },
        "deviation_from_pooled_protocol": "task-specific rank-limited PCA and model fits; used to compare mechanism consistency across tasks, not to replace pooled 72-context result",
        "source_code_hashes": {
            str(Path(early.__file__).resolve()): sha256(Path(early.__file__).resolve()),
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(early.TPI_CODE): sha256(early.TPI_CODE),
            str(early.CF_CODE): sha256(early.CF_CODE),
            str(early.FULL_CODE): sha256(early.FULL_CODE),
        },
    }
    early.write_json(out / f"TASK{task}_EARLY_PROTOCOL.json", protocol)


def run(task: int, out: Path, audit_only: bool = False) -> None:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"task must be one of {SUPPORTED_TASKS}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    tpi = early.load_module(f"tpi_task{task}_early", early.TPI_CODE)
    cf = early.load_module(f"cf_task{task}_early", early.CF_CODE)
    full = early.load_module(f"full_task{task}_early", early.FULL_CODE)
    cmap, traces, meta, visual_dim, pca_path, canonical, audit = audit_and_load(task, out, tpi)
    if audit_only:
        print(json.dumps({"status": "AUDIT_ONLY_PASS", "task": task, "audit": audit}, indent=2), flush=True)
        return

    pairs = build_pairs(task, cf, traces, meta)
    segs, norm = early.build_segments_and_norm(cf, tpi, traces)
    norm_path = out / f"TASK{task}_TRAIN_NORMALIZATION.npz"
    np.savez(norm_path, x_mean=norm[0], x_std=norm[1], y_mean=norm[2], y_std=norm[3])
    freeze_protocol(task, out, visual_dim, audit)

    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metrics: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for seed in early.SEEDS:
        print(f"[task{task}] seed={seed} starting base/residual/full", flush=True)
        base, bh, bsteps = early.train_base(full, traces, segs, norm, cmap, meta, device, seed)
        bp = out / f"PROSPECTIVE_BASE_FEAS_task{task}_seed{seed}.pt"
        save_checkpoint(bp, base, "PROSPECTIVE_BASE_FEAS", seed, task, norm, pca_path)
        residual, rh, rsteps = early.train_residual(base, traces, segs, norm, cmap, meta, device, seed, visual_dim)
        rp = out / f"VISUAL_INTERCEPT_RESIDUAL_task{task}_seed{seed}.pt"
        save_checkpoint(rp, residual, "VISUAL_INTERCEPT_RESIDUAL", seed, task, norm, pca_path,
                        {"base_checkpoint": str(bp), "base_checkpoint_sha256": sha256(bp), "force_input_to_residual": False})
        full_model, fh, fsteps = early.train_full(traces, segs, norm, cmap, meta, device, seed, visual_dim)
        fp = out / f"VISUAL_CONTEXT_FULL_FEAS_task{task}_seed{seed}.pt"
        save_checkpoint(fp, full_model, "VISUAL_CONTEXT_FULL_FEAS", seed, task, norm, pca_path)
        histories["PROSPECTIVE_BASE_FEAS"].extend(bh)
        histories["VISUAL_INTERCEPT_RESIDUAL"].extend(rh)
        histories["VISUAL_CONTEXT_FULL_FEAS"].extend(fh)
        for name, model, path, steps, base_model in [
            ("PROSPECTIVE_BASE_FEAS", base, bp, bsteps, None),
            ("VISUAL_INTERCEPT_RESIDUAL", residual, rp, rsteps, base),
            ("VISUAL_CONTEXT_FULL_FEAS", full_model, fp, fsteps, None),
        ]:
            kind = "BASE" if name.startswith("PROSPECTIVE") else "RESIDUAL" if "RESIDUAL" in name else "FULL"
            logits = early.logits_for(model, kind, traces, segs, norm, cmap, device, base=base_model)
            metrics.append({**early.train_metrics(name, seed, traces, logits), "task": task})
            manifest.append({"variant": name, "task": task, "seed": seed, "checkpoint": str(path),
                             "sha256": sha256(path), "optimizer_steps": steps})

        print(f"[task{task}] seed={seed} starting joint", flush=True)
        joint, jh, jsteps, units, initial = early.train_joint(
            full, cf, tpi, traces, pairs, segs, norm, cmap, meta, device, seed, visual_dim,
        )
        jp = out / f"VISUAL_CONTEXT_JOINT_task{task}_seed{seed}.pt"
        save_checkpoint(jp, joint, "VISUAL_CONTEXT_JOINT", seed, task, norm, pca_path,
                        {"initial_physics_checkpoint": str(initial), "initial_physics_checkpoint_sha256": sha256(initial),
                         "physical_units": units, "adjacent_ie_pairs": len(pairs), "lambda_feas": early.LAMBDA_FEAS})
        histories["VISUAL_CONTEXT_JOINT"].extend(jh)
        metrics.append({**early.train_metrics("VISUAL_CONTEXT_JOINT", seed, traces,
                                             early.logits_for(joint, "JOINT", traces, segs, norm, cmap, device)),
                        "task": task})
        manifest.append({"variant": "VISUAL_CONTEXT_JOINT", "task": task, "seed": seed,
                         "checkpoint": str(jp), "sha256": sha256(jp),
                         "optimizer_steps": jsteps, "physical_units": units})
        print(f"[task{task}] seed={seed} complete", flush=True)

    for variant, rows in histories.items():
        early.write_csv(out / f"{variant}_TASK{task}_TRAINING_MANIFEST.csv", rows)
    early.write_csv(out / f"TASK{task}_TRAIN_IN_SAMPLE_METRICS.csv", metrics)
    early.write_json(out / f"ALL_PROSPECTIVE_VISUAL_MODELS_FROZEN_TASK{task}_ONLY.json", {
        "status": f"TASK{task}_ONLY_12_FROZEN_BEFORE_DEV", "checkpoint_count": len(manifest),
        "checkpoints": manifest, "DEV_used": False, "TEST_used": False,
        "eligible_for_task_specific_DEV_evaluation": True,
        "eligible_for_pooled_model_selection": False,
    })
    summary = {
        "status": f"COMPLETE_TASK{task}_TRAIN_ONLY_FROZEN", "task": task,
        "contexts": 18, "branches": 180,
        "successes": int(sum(r["success"] for r in canonical)),
        "failures": int(180 - sum(r["success"] for r in canonical)),
        "direct_labels": audit["direct_labels"], "reconstructed_labels": audit["reconstructed_labels"],
        "checkpoints": len(manifest), "device": "cpu", "visual_pca_components": visual_dim,
        "DEV_used": False, "final_scientific_classification": "NOT_PERMITTED_WITH_TRAIN_ONLY",
    }
    early.write_json(out / f"TASK{task}_RUN_SUMMARY.json", summary)
    write_notebook(task, out, summary)
    hashes = [f"{sha256(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"]
    (out / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=SUPPORTED_TASKS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    run(args.task, args.out.resolve(), args.audit_only)


if __name__ == "__main__":
    main()
