#!/usr/bin/env python3
"""Frozen-model independent-root scaling train/evaluate pipeline.

Training never opens TEST.  Evaluation refuses to open TEST until all 72
task/N/model/seed checkpoints are frozen and both TEST commits pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import per_task_visual_context_early as taskwise
import task0_visual_context_early as early
import task0_visual_generalization as gen


ROOT = Path("/home/exouser/FORTE")
OUT = ROOT / "root_scaling_20260831"
SOURCE = Path("/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000")
TASK0_OLD = ROOT / "task0_context_sample_complexity_20260831/collection_train_new"
TASKS = [0, 5]
LEVELS = [6, 15, 30, 50]
MODELS = ["Base", "Full Visual", "Visual Joint"]
SEEDS = [0, 1, 2]
FORCES_DENSE = np.round(np.arange(3.0, 5.0001, 0.05), 2)
RHO = 0.80
S6_FROZEN = {
    0: ROOT / "task0_visual_context_early_20260831_025000",
    5: ROOT / "taskwise_visual_context_20260831_074000/task5_train_frozen",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, frame_or_rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame_or_rows, pd.DataFrame):
        frame_or_rows.to_csv(path, index=False)
    else:
        rows = list(frame_or_rows); fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields or ["status"]); w.writeheader(); w.writerows(rows)


def verify_freeze() -> tuple[dict, dict]:
    freeze = json.loads((OUT / "ROOT_SCALING_FREEZE_SHA256.json").read_text())
    for raw, expected in freeze["hashes"].items():
        if sha256(Path(raw)) != expected:
            raise RuntimeError(f"frozen input changed: {raw}")
    train = json.loads((OUT / "ROOT_SCALING_TRAIN_MANIFEST.json").read_text())
    return freeze, train


def modules():
    suffix = str(__import__("os").getpid())
    return (early.load_module("scale_tpi_" + suffix, early.TPI_CODE),
            early.load_module("scale_cf_" + suffix, early.CF_CODE),
            early.load_module("scale_full_" + suffix, early.FULL_CODE))


def addition_root(task: int) -> Path:
    if task == 0 and (TASK0_OLD / "task0/task0/context.csv").exists():
        return TASK0_OLD
    return OUT / "collection_train"


def source_tables(task: int, n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = SOURCE / "collection_train"
    c1 = pd.read_csv(base / f"task{task}/task{task}/context.csv")
    b1 = pd.read_csv(base / f"task{task}/task{task}/branches.csv")
    v1 = pd.read_csv(base / "visual_alignment_worker.csv")
    v1 = v1[v1.context_id.astype(str).isin(c1.context_id.astype(str))]
    if n == 6:
        return c1, b1, v1
    add = addition_root(task)
    c2 = pd.read_csv(add / f"task{task}/task{task}/context.csv")
    b2 = pd.read_csv(add / f"task{task}/task{task}/branches.csv")
    v2 = pd.read_csv(add / "visual_alignment_worker.csv")
    v2 = v2[v2.context_id.astype(str).isin(c2.context_id.astype(str))]
    return (pd.concat([c1, c2], ignore_index=True),
            pd.concat([b1, b2], ignore_index=True),
            pd.concat([v1, v2], ignore_index=True))


def pca_path(task: int, n: int) -> Path:
    if n == 6:
        return S6_FROZEN[task] / ("TASK0_PCA17_PROVISIONAL.npz" if task == 0 else "TASK5_PCA17_TRAIN_ONLY.npz")
    return OUT / f"preprocessing/TASK{task}_S{n}_PCA17_TRAIN_ONLY.npz"


def norm_path(task: int, n: int) -> Path:
    if n == 6:
        return S6_FROZEN[task] / f"TASK{task}_TRAIN_NORMALIZATION.npz"
    return OUT / f"preprocessing/TASK{task}_S{n}_TRAIN_NORMALIZATION.npz"


def fit_or_transform_pca(task: int, n: int, cids: list[str], vmap: dict) -> tuple[np.ndarray, np.ndarray, Path]:
    path = pca_path(task, n)
    raw = np.stack([np.load(vmap[c]["visual_feature_path"], allow_pickle=False).astype(np.float32) for c in cids])
    if n != 6 and not path.exists():
        mean = raw.mean(0); _, singular, vt = np.linalg.svd(raw - mean, full_matrices=False)
        components = vt[:17].astype(np.float32)
        projected = ((raw - mean) @ components.T).astype(np.float32)
        pmean = projected.mean(0).astype(np.float32)
        pstd = projected.std(0).astype(np.float32); pstd[pstd < 1e-6] = 1.0
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, raw_mean=mean, components=components, projected_mean=pmean,
                 projected_std=pstd, singular_values=singular)
    p = np.load(path)
    if p["components"].shape[0] != 17:
        raise RuntimeError(f"task{task}/S{n}: frozen PCA is not 17-D")
    z = (raw - p["raw_mean"]) @ p["components"].T
    z = ((z - p["projected_mean"]) / p["projected_std"]).astype(np.float32)
    return raw, z, path


def build_pairs(task: int, cf, traces, meta):
    by = defaultdict(list)
    for tr in traces:
        by[(tr.context_id, meta[tr.branch_id].repeat)].append(tr)
    pairs = []
    for (cid, repeat), rows in sorted(by.items()):
        rows = sorted(rows, key=lambda tr: meta[tr.branch_id].stratum)
        if len(rows) != 5:
            raise RuntimeError(f"task{task}/{cid}/R{repeat}: expected five strata")
        for a, b in zip(rows[:-1], rows[1:]):
            ma, mb = meta[a.branch_id], meta[b.branch_id]
            pairs.append(cf.Pair(f"scale:t{task}:{cid}:R{repeat}:S{ma.stratum}_vs_{mb.stratum}",
                                 f"scale:t{task}:{cid}:R{repeat}", "TRAIN", "ROOT_SCALING",
                                 cid, a.root_id, task, ma.friction_band, a.mu, a.force, b.force,
                                 "adjacent", False, a, b))
    if len(pairs) != len(by) * 4:
        raise RuntimeError("adjacent IE pair count mismatch")
    return pairs


def load_train(task: int, n: int, tpi, cf):
    _, manifest = verify_freeze()
    ids = manifest["nested_sets"][f"S{n}"][str(task)]["context_ids"]
    expected_roots = manifest["nested_sets"][f"S{n}"][str(task)]["root_ids"]
    contexts, branches, visual = source_tables(task, n)
    contexts = contexts[contexts.context_id.astype(str).isin(ids)].sort_values("context_id").copy()
    branches = branches[branches.context_id.astype(str).isin(ids)].copy()
    visual = visual[visual.context_id.astype(str).isin(ids)].copy()
    expected_contexts = len(ids)
    if len(contexts) != expected_contexts or contexts.context_id.nunique() != expected_contexts:
        raise RuntimeError(f"task{task}/S{n}: context count mismatch")
    if contexts.root_id.nunique() != n or set(contexts.root_id.astype(str)) != set(expected_roots):
        raise RuntimeError(f"task{task}/S{n}: independent root identity mismatch")
    if len(branches) != expected_contexts * 10 or branches.branch_id.nunique() != expected_contexts * 10:
        raise RuntimeError(f"task{task}/S{n}: branch count/identity mismatch")
    if len(visual) != expected_contexts or visual.context_id.nunique() != expected_contexts:
        raise RuntimeError(f"task{task}/S{n}: visual identity mismatch")
    if int(branches.state_parity.sum()) != len(branches) or int(contexts.strict_matched.sum()) != len(contexts):
        raise RuntimeError(f"task{task}/S{n}: state parity mismatch")
    vmap = visual.set_index("context_id").to_dict("index")
    cids = contexts.context_id.astype(str).tolist()
    raw, z, pp = fit_or_transform_pca(task, n, cids, vmap)
    cmap = {}
    for row, rv, zv in zip(contexts.itertuples(index=False), raw, z):
        state0, mask0, _ = early.strict_preprobe_state(Path(str(row.probe_telemetry_path)))
        cmap[str(row.context_id)] = {"visual_raw": rv, "visual": zv,
            "preprobe_state": state0, "preprobe_mask": mask0, "root_id": str(row.root_id),
            "friction": float(row.hidden_friction_analysis_only), "friction_band": str(row.friction_band)}
    corrected = {"left_normal_force_N", "right_normal_force_N", "left_tangential_force_N",
                 "right_tangential_force_N", "object_vx_mps", "object_vy_mps", "object_vz_mps"}
    traces, meta = [], {}
    for r in branches.sort_values("branch_id").itertuples(index=False):
        cid = str(r.context_id); path = Path(str(r.telemetry_path)); d = pd.read_csv(path)
        if len(d) < early.H + 1 or not corrected <= set(d) or not np.isfinite(d[list(corrected)].to_numpy(float)).all():
            raise RuntimeError(f"task{task}/S{n}: bad telemetry {r.branch_id}")
        state, mask = tpi.state_from(d); state, mask = state.copy(), mask.copy()
        state[0] = cmap[cid]["preprobe_state"]; mask[0] = cmap[cid]["preprobe_mask"]
        force = float(r.requested_force_N); mu = float(r.hidden_friction_analysis_only)
        nominal = tpi.nominal_from(d, 0, force, mu, state, mask)
        tr = tpi.Trace(str(r.branch_id), cid, str(r.root_id), task, "TRAIN", force, mu,
                       int(r.full_task_success_y), "continuous", path, state, mask, nominal,
                       d.phase.astype(str).tolist(), 1.0, f"ROOT_SCALING_TASK{task}")
        stratum, repeat = early.parse_cell(str(r.branch_label))
        traces.append(tr)
        meta[tr.branch_id] = early.Meta(tr.branch_id, cid, task, str(r.root_id),
                                        str(r.friction_band), int(r.full_task_success_y), force, repeat, stratum)
    pairs = build_pairs(task, cf, traces, meta)
    segs, norm = early.build_segments_and_norm(cf, tpi, traces)
    npth = norm_path(task, n)
    if n == 6:
        frozen = np.load(npth)
        fnorm = tuple(frozen[k].astype(np.float32) for k in ("x_mean", "x_std", "y_mean", "y_std"))
        if not all(np.allclose(a, b, rtol=0, atol=1e-7) for a, b in zip(norm, fnorm)):
            raise RuntimeError(f"task{task}/S6 normalization differs from authoritative freeze")
        norm = fnorm
    elif not npth.exists():
        npth.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npth, x_mean=norm[0], x_std=norm[1], y_mean=norm[2], y_std=norm[3])
    else:
        frozen = np.load(npth)
        norm = tuple(frozen[k].astype(np.float32) for k in ("x_mean", "x_std", "y_mean", "y_std"))
    return contexts, branches, cmap, traces, meta, pairs, segs, norm, pp, npth


def ck_path(task: int, n: int, model: str, seed: int) -> Path:
    slug = model.upper().replace(" ", "_")
    return OUT / f"checkpoints/TASK{task}_S{n}_{slug}_seed{seed}.pt"


def s6_source(task: int, model: str, seed: int) -> Path:
    prefix = {"Base": "PROSPECTIVE_BASE_FEAS", "Full Visual": "VISUAL_CONTEXT_FULL_FEAS",
              "Visual Joint": "VISUAL_CONTEXT_JOINT"}[model]
    return S6_FROZEN[task] / f"{prefix}_task{task}_seed{seed}.pt"


def save_ck(path: Path, task: int, n: int, model: str, seed: int, state, pca: Path, norm: Path, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": task, "independent_roots": n, "model": model, "seed": seed,
                "state_dict": state, "epochs": early.EPOCHS, "optimizer": "AdamW",
                "lr": early.LR, "weight_decay": early.WEIGHT_DECAY, "batch_size": early.BATCH,
                "PCA_dimension": 17, "PCA_path": str(pca), "PCA_sha256": sha256(pca),
                "normalization_path": str(norm), "normalization_sha256": sha256(norm),
                "lambda_physics": 1.0 if model == "Visual Joint" else None,
                "lambda_IE": 1.0 if model == "Visual Joint" else None,
                "lambda_feasibility": early.LAMBDA_FEAS if model == "Visual Joint" else None,
                "TRAIN_only": True, "TEST_used": False, "architecture_or_loss_changed": False,
                **(extra or {})}, path)


def load_model(path: Path, model: str, tpi, full, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if model == "Base":
        m = full.FeasibilityOnly().to(device)
    elif model == "Full Visual":
        m = early.VisualFullFeas(17).to(device)
    else:
        ps = {k[len("physics."):]: v for k, v in ck["state_dict"].items() if k.startswith("physics.")}
        m = early.VisualJoint(tpi, ps, 17).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


def train_all() -> None:
    verify_freeze()
    # TRAIN only: this phase intentionally does not check or open TEST files.
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    tpi, cf, full = modules(); device = torch.device("cpu")
    rows = []
    for task in TASKS:
        for n in LEVELS:
            contexts, branches, cmap, traces, meta, pairs, segs, norm, pca, npth = load_train(task, n, tpi, cf)
            print(f"[root-scale] task={task} S{n} roots={contexts.root_id.nunique()} contexts={len(contexts)}", flush=True)
            for seed in SEEDS:
                for model in MODELS:
                    path = ck_path(task, n, model, seed)
                    extra = {}
                    if path.exists():
                        trained = load_model(path, model, tpi, full, device)
                    elif n == 6:
                        src = s6_source(task, model, seed)
                        q = torch.load(src, map_location=device, weights_only=False)
                        if model == "Base": trained = full.FeasibilityOnly().to(device)
                        elif model == "Full Visual": trained = early.VisualFullFeas(17).to(device)
                        else:
                            ps = {k[len("physics."):]: v for k, v in q["state_dict"].items() if k.startswith("physics.")}
                            trained = early.VisualJoint(tpi, ps, 17).to(device)
                        trained.load_state_dict(q["state_dict"]); trained.eval()
                        extra = {"reused_authoritative_S6": str(src), "source_sha256": sha256(src)}
                        save_ck(path, task, n, model, seed, trained.state_dict(), pca, npth, extra)
                    elif model == "Base":
                        trained, _, steps = early.train_base(full, traces, segs, norm, cmap, meta, device, seed)
                        extra = {"optimizer_steps": steps}; save_ck(path, task, n, model, seed, trained.state_dict(), pca, npth, extra)
                    elif model == "Full Visual":
                        trained, _, steps = early.train_full(traces, segs, norm, cmap, meta, device, seed, 17)
                        extra = {"optimizer_steps": steps}; save_ck(path, task, n, model, seed, trained.state_dict(), pca, npth, extra)
                    else:
                        trained, _, steps, units, initial = early.train_joint(full, cf, tpi, traces, pairs, segs,
                                                                            norm, cmap, meta, device, seed, 17)
                        extra = {"optimizer_steps": steps, "physical_units": units,
                                 "initial_physics_checkpoint": str(initial), "initial_physics_sha256": sha256(initial)}
                        save_ck(path, task, n, model, seed, trained.state_dict(), pca, npth, extra)
                    rows.append({"task": task, "N_root": n, "friction_conditioned_contexts": len(contexts),
                                 "model": model, "seed": seed, "checkpoint": str(path),
                                 "checkpoint_sha256": sha256(path), "PCA_path": str(pca),
                                 "PCA_sha256": sha256(pca), "normalization_path": str(npth),
                                 "normalization_sha256": sha256(npth), "epochs": early.EPOCHS,
                                 "optimizer": "AdamW", "lr": early.LR, "batch_size": early.BATCH,
                                 "lambda_physics": 1.0 if model == "Visual Joint" else "",
                                 "lambda_IE": 1.0 if model == "Visual Joint" else "",
                                 "lambda_feasibility": early.LAMBDA_FEAS if model == "Visual Joint" else "",
                                 "TEST_used": 0})
                    write_json(OUT / "ROOT_SCALING_CHECKPOINT_MANIFEST_PARTIAL.json",
                               {"status": "IN_PROGRESS_TRAIN_ONLY", "checkpoints": rows, "TEST_used": False})
    if len(rows) != len(TASKS) * len(LEVELS) * len(MODELS) * len(SEEDS):
        raise RuntimeError("checkpoint count mismatch")
    write_csv(OUT / "ROOT_SCALING_MODEL_HASHES.csv", rows)
    write_json(OUT / "ROOT_SCALING_CHECKPOINT_MANIFEST.json", {
        "status": "ALL_72_FROZEN_BEFORE_TEST_TOUCH", "checkpoint_count": len(rows),
        "models": MODELS, "root_levels": LEVELS, "seeds": SEEDS, "TEST_used": False,
        "architecture_loss_representation_changes": False,
        "model_hash_csv": str(OUT / "ROOT_SCALING_MODEL_HASHES.csv"),
        "model_hash_csv_sha256": sha256(OUT / "ROOT_SCALING_MODEL_HASHES.csv"),
    })
    print(json.dumps({"status": "ALL_72_FROZEN_BEFORE_TEST_TOUCH", "checkpoints": len(rows)}, indent=2))


def transform_pca(paths: list[Path], path: Path) -> tuple[np.ndarray, np.ndarray]:
    p = np.load(path); raw = np.stack([np.load(x, allow_pickle=False).astype(np.float32) for x in paths])
    z = (raw - p["raw_mean"]) @ p["components"].T
    return raw, ((z - p["projected_mean"]) / p["projected_std"]).astype(np.float32)


def load_test(task: int, tpi, pca: Path):
    base = OUT / "collection_test"
    c = pd.read_csv(base / f"task{task}/task{task}/context.csv").sort_values("context_id")
    b = pd.read_csv(base / f"task{task}/task{task}/branches.csv")
    v = pd.read_csv(base / "visual_alignment_worker.csv")
    expected = json.loads((OUT / "ROOT_SCALING_TEST_MANIFEST.json").read_text())["contexts"][str(task)]
    ids = {x["context_id"] for x in expected}
    c = c[c.context_id.astype(str).isin(ids)]; b = b[b.context_id.astype(str).isin(ids)]
    v = v[v.context_id.astype(str).isin(ids)]
    if len(c) != 10 or len(b) != 450 or len(v) != 10:
        raise RuntimeError(f"task{task}: committed TEST count mismatch")
    vm = v.set_index("context_id").to_dict("index")
    raw, z = transform_pca([Path(vm[cid]["visual_feature_path"]) for cid in c.context_id.astype(str)], pca)
    cmap, templates = {}, {}
    for row, rv, zv in zip(c.itertuples(index=False), raw, z):
        cid = str(row.context_id); state0, mask0, _ = early.strict_preprobe_state(Path(str(row.probe_telemetry_path)))
        q = b[b.context_id.astype(str) == cid].sort_values(["requested_force_N", "branch_label"])
        path = Path(str(q.iloc[0].telemetry_path)); d = pd.read_csv(path)
        state, mask = tpi.state_from(d); state, mask = state.copy(), mask.copy()
        state[0] = state0; mask[0] = mask0; mu = float(row.hidden_friction_analysis_only)
        nominal = tpi.nominal_from(d, 0, 3.0, mu, state, mask)
        templates[cid] = tpi.Trace(f"template:{cid}", cid, str(row.root_id), task, "TEST", 3.0, mu,
                                   0, "dense", path, state, mask, nominal, d.phase.astype(str).tolist(),
                                   1.0, f"ROOT_SCALING_TEST_TASK{task}")
        cmap[cid] = {"visual": zv, "visual_raw": rv, "root_id": str(row.root_id),
                     "friction": mu, "friction_band": str(row.friction_band)}
    return c, b, cmap, templates


def load_models(task: int, n: int, tpi, full, device):
    return {(model, seed): load_model(ck_path(task, n, model, seed), model, tpi, full, device)
            for model in MODELS for seed in SEEDS}


def predict(models, cmap, templates, norm, tpi, cf, forces_by_context):
    rows = []
    with torch.no_grad():
        for cid in sorted(templates):
            visual = torch.tensor(cmap[cid]["visual"][None], dtype=torch.float32)
            for force in forces_by_context[cid]:
                sn, cn = gen.normalized_segment(cf, tpi, templates[cid], float(force), norm)
                step = torch.tensor(sn[None], dtype=torch.float32); cond = torch.tensor(cn[None], dtype=torch.float32)
                for model in MODELS:
                    probs = []
                    for seed in SEEDS:
                        m = models[(model, seed)]
                        if model == "Base": logit = m(step, cond)
                        elif model == "Full Visual": logit = m(step, cond, visual)
                        else: logit = m(step, cond, visual)[1]
                        probs.append(float(torch.sigmoid(logit)[0]))
                    rows.append({"context_id": cid, "force_N": float(force), "model": model,
                                 **{f"seed{s}_prob": probs[s] for s in SEEDS},
                                 "ensemble_prob": float(np.mean(probs)),
                                 "seed_probability_std": float(np.std(probs, ddof=1))})
    return pd.DataFrame(rows)


def real_cells(branches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cid, force), q in branches.groupby(["context_id", "requested_force_N"]):
        rows.append({"context_id": str(cid), "force_N": float(force),
                     "successes": int(q.full_task_success_y.sum()), "repeats": len(q),
                     "p_real": float(q.full_task_success_y.mean())})
    return pd.DataFrame(rows)


def real_frontiers(real: pd.DataFrame) -> dict[str, float]:
    ans = {}
    for cid, q in real.groupby("context_id"):
        q = q.sort_values("force_N"); safe = q[q.p_real >= RHO]
        ans[str(cid)] = float(safe.force_N.min()) if len(safe) else math.nan
    return ans


def per_root(task: int, n: int, pred: pd.DataFrame, real: pd.DataFrame, cmap) -> pd.DataFrame:
    fronts = real_frontiers(real); rows = []
    aggs = [("ENSEMBLE", "ensemble_prob")] + [(f"SEED_{s}", f"seed{s}_prob") for s in SEEDS]
    for model in MODELS:
        for cid, rr in real.groupby("context_id"):
            rr = rr.sort_values("force_N")
            q = pred[(pred.model == model) & (pred.context_id == cid)].sort_values("force_N")
            for agg, col in aggs:
                qr = q[q.force_N.isin(rr.force_N)].sort_values("force_N")
                p = qr[col].to_numpy(float); y = rr.p_real.to_numpy(float)
                dense = q[col].to_numpy(float); diff = np.diff(dense)
                safe = q[q[col] >= RHO]; fp = float(safe.force_N.min()) if len(safe) else math.nan
                fr = fronts[str(cid)]; err = fp - fr if math.isfinite(fp) and math.isfinite(fr) else math.nan
                rows.append({"task": task, "N_root": n, "model": model, "aggregation": agg,
                    "context_id": cid, "root_id": cmap[cid]["root_id"],
                    "friction_band": cmap[cid]["friction_band"], "mu_GT": cmap[cid]["friction"],
                    "probability_MAE": float(np.mean(np.abs(p-y))),
                    "Brier": float(np.mean((p-y)**2)),
                    "NLL": float(-np.mean(y*np.log(np.clip(p,1e-8,1))+(1-y)*np.log(np.clip(1-p,1e-8,1)))),
                    "signed_bias": float(np.mean(p-y)), "real_frontier_N": fr,
                    "real_frontier_supported": math.isfinite(fr), "predicted_frontier_N": fp,
                    "signed_frontier_error_N": err,
                    "absolute_frontier_error_N": abs(err) if math.isfinite(err) else math.nan,
                    "finite_decision": math.isfinite(fp),
                    "under_force": bool(err < -1e-8) if math.isfinite(err) else math.nan,
                    "under_force_magnitude_N": max(0.0, -err) if math.isfinite(err) else math.nan,
                    "excess_force_N": max(0.0, err) if math.isfinite(err) else math.nan,
                    "dense_monotonic": bool(np.all(diff >= -1e-8)),
                    "nonmonotonic_steps": int(np.sum(diff < -1e-8)),
                    "safe_to_unsafe_reversals": int(np.sum((dense[:-1] >= RHO) & (dense[1:] < RHO)))})
    return pd.DataFrame(rows)


def aggregate_one(task: int, n: int, model: str, agg: str, pr: pd.DataFrame,
                  pred: pd.DataFrame, real: pd.DataFrame, train_pred: pd.DataFrame,
                  train_real: pd.DataFrame) -> tuple[dict, dict, dict]:
    col = "ensemble_prob" if agg == "ENSEMBLE" else f"seed{int(agg.split('_')[1])}_prob"
    q = pr[(pr.model == model) & (pr.aggregation == agg)].copy()
    ptest, ytest = [], []
    for cid, rr in real.groupby("context_id"):
        rr = rr.sort_values("force_N")
        z = pred[(pred.model == model) & (pred.context_id == cid) & pred.force_N.isin(rr.force_N)].sort_values("force_N")
        ptest.extend(z[col]); ytest.extend(rr.p_real)
    ptest, ytest = np.asarray(ptest, float), np.asarray(ytest, float)
    ptr, ytr = [], []
    for cid, rr in train_real.groupby("context_id"):
        rr = rr.sort_values("force_N")
        z = train_pred[(train_pred.model == model) & (train_pred.context_id == cid)].sort_values("force_N")
        ptr.extend(z[col]); ytr.extend(rr.p_real)
    ptr, ytr = np.asarray(ptr, float), np.asarray(ytr, float)
    base = pr[(pr.model == "Base") & (pr.aggregation == agg)].set_index("context_id")
    improved_prob = int(sum(float(x.probability_MAE) < float(base.loc[x.context_id].probability_MAE) for x in q.itertuples()))
    comp = [x for x in q.itertuples() if math.isfinite(float(x.absolute_frontier_error_N)) and
            math.isfinite(float(base.loc[x.context_id].absolute_frontier_error_N))]
    improved_front = int(sum(float(x.absolute_frontier_error_N) < float(base.loc[x.context_id].absolute_frontier_error_N) for x in comp))
    evaluable = q[q.real_frontier_supported & q.finite_decision]
    under = evaluable[evaluable.under_force == True]
    test_mae = float(np.mean(abs(ptest-ytest)))
    test_nll = float(-np.mean(ytest*np.log(np.clip(ptest,1e-8,1))+(1-ytest)*np.log(np.clip(1-ptest,1e-8,1))))
    train_mae = float(np.mean(abs(ptr-ytr)))
    train_nll = float(-np.mean(ytr*np.log(np.clip(ptr,1e-8,1))+(1-ytr)*np.log(np.clip(1-ptr,1e-8,1))))
    main = {"task": task, "N_root": n, "model": model, "aggregation": agg,
            "TEST_roots": len(q), "TEST_probability_MAE": test_mae,
            "TEST_Brier": float(np.mean((ptest-ytest)**2)), "TEST_NLL": test_nll,
            "TEST_signed_bias": float(np.mean(ptest-ytest)),
            "TEST_frontier_MAE_N": float(evaluable.absolute_frontier_error_N.mean()) if len(evaluable) else math.nan,
            "TEST_real_frontier_supported_rate": float(q.real_frontier_supported.mean()),
            "TEST_finite_decision_rate": float(q.finite_decision.mean()),
            "TEST_under_force_rate": float(evaluable.under_force.mean()) if len(evaluable) else math.nan,
            "TEST_excess_force_N": float(evaluable.excess_force_N.mean()) if len(evaluable) else math.nan,
            "TEST_dense_monotonic_root_fraction": float(q.dense_monotonic.mean()),
            "TEST_safe_to_unsafe_reversals": int(q.safe_to_unsafe_reversals.sum()),
            "TEST_probability_improved_roots_vs_Base": improved_prob,
            "fraction_TEST_roots_probability_improved_vs_Base": improved_prob/len(q),
            "TEST_frontier_comparable_roots_vs_Base": len(comp),
            "TEST_frontier_improved_roots_vs_Base": improved_front,
            "fraction_TEST_roots_frontier_improved_vs_Base": improved_front/len(comp) if comp else math.nan,
            "TRAIN_probability_MAE": train_mae, "TRAIN_NLL": train_nll,
            "TRAIN_to_TEST_probability_MAE_gap": test_mae-train_mae,
            "TRAIN_to_TEST_NLL_gap": test_nll-train_nll}
    safety = {"task": task, "N_root": n, "model": model, "aggregation": agg,
              "TEST_roots": len(q), "real_frontier_supported_roots": int(q.real_frontier_supported.sum()),
              "frontier_evaluable_roots": len(evaluable), "finite_decision_rate": float(q.finite_decision.mean()),
              "under_force_rate": float(evaluable.under_force.mean()) if len(evaluable) else math.nan,
              "mean_under_force_magnitude_N": float(under.under_force_magnitude_N.mean()) if len(under) else 0.0,
              "mean_excess_force_N": float(evaluable.excess_force_N.mean()) if len(evaluable) else math.nan,
              "dense_monotonic_root_fraction": float(q.dense_monotonic.mean()),
              "nonmonotonic_steps": int(q.nonmonotonic_steps.sum()),
              "safe_to_unsafe_reversals": int(q.safe_to_unsafe_reversals.sum())}
    gap = {"task": task, "N_root": n, "model": model, "aggregation": agg,
           "TRAIN_probability_MAE": train_mae, "TEST_probability_MAE": test_mae,
           "MAE_gap_TEST_minus_TRAIN": test_mae-train_mae,
           "TRAIN_NLL": train_nll, "TEST_NLL": test_nll, "NLL_gap_TEST_minus_TRAIN": test_nll-train_nll}
    return main, safety, gap


def add_seed_mean_std(frame: pd.DataFrame, keys: list[str], numeric: list[str]) -> pd.DataFrame:
    rows = [frame]
    seed = frame[frame.aggregation.str.startswith("SEED_")]
    grouped = seed.groupby(keys, dropna=False)
    for label, ddof in (("SEED_MEAN", 0), ("SEED_STD", 1)):
        x = grouped[numeric].mean().reset_index() if label == "SEED_MEAN" else grouped[numeric].std(ddof=ddof).reset_index()
        x["aggregation"] = label; rows.append(x)
    return pd.concat(rows, ignore_index=True, sort=False)


def evaluate_all() -> None:
    verify_freeze()
    ck = json.loads((OUT / "ROOT_SCALING_CHECKPOINT_MANIFEST.json").read_text())
    if ck.get("status") != "ALL_72_FROZEN_BEFORE_TEST_TOUCH" or ck.get("checkpoint_count") != 72:
        raise RuntimeError("all 72 checkpoints must freeze before TEST touch")
    for task in TASKS:
        commit = json.loads((OUT / f"TASK{task}_TEST_COLLECTION_COMMIT.json").read_text())
        if commit.get("status") != "ATOMICALLY_COMMITTED_UNTOUCHED_TEST":
            raise RuntimeError(f"task{task} TEST not atomically committed")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    tpi, cf, full = modules(); device = torch.device("cpu")
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    summary, safety, gaps, perroots = [], [], [], []
    for task in TASKS:
        for n in LEVELS:
            _, train_branches, train_cmap, train_traces, _, _, _, norm, pca, _ = load_train(task, n, tpi, cf)
            _, test_branches, test_cmap, templates = load_test(task, tpi, pca)
            models = load_models(task, n, tpi, full, device)
            test_forces = {cid: FORCES_DENSE for cid in templates}
            pred = predict(models, test_cmap, templates, norm, tpi, cf, test_forces)
            real = real_cells(test_branches)
            pr = per_root(task, n, pred, real, test_cmap); perroots.append(pr)
            train_templates = {cid: next(x for x in train_traces if x.context_id == cid) for cid in train_cmap}
            train_forces = {cid: sorted(train_branches[train_branches.context_id.astype(str) == cid].requested_force_N.astype(float).unique())
                            for cid in train_templates}
            train_pred = predict(models, train_cmap, train_templates, norm, tpi, cf, train_forces)
            train_real = real_cells(train_branches)
            for model in MODELS:
                for agg in ["ENSEMBLE"] + [f"SEED_{s}" for s in SEEDS]:
                    a, b, c = aggregate_one(task, n, model, agg, pr, pred, real, train_pred, train_real)
                    summary.append(a); safety.append(b); gaps.append(c)
            pred.to_csv(OUT / f"predictions/TASK{task}_S{n}_TEST_DENSE_PREDICTIONS.csv", index=False)
            print(f"[evaluate] task={task} S{n} complete", flush=True)
    lc = pd.DataFrame(summary); sf = pd.DataFrame(safety); gp = pd.DataFrame(gaps)
    main_numeric = [c for c in lc.columns if c not in {"task","N_root","model","aggregation"} and pd.api.types.is_numeric_dtype(lc[c])]
    safe_numeric = [c for c in sf.columns if c not in {"task","N_root","model","aggregation"} and pd.api.types.is_numeric_dtype(sf[c])]
    gap_numeric = [c for c in gp.columns if c not in {"task","N_root","model","aggregation"} and pd.api.types.is_numeric_dtype(gp[c])]
    lc = add_seed_mean_std(lc, ["task","N_root","model"], main_numeric)
    sf = add_seed_mean_std(sf, ["task","N_root","model"], safe_numeric)
    gp = add_seed_mean_std(gp, ["task","N_root","model"], gap_numeric)
    write_csv(OUT / "ROOT_SCALING_LEARNING_CURVE.csv", lc)
    write_csv(OUT / "ROOT_SCALING_PER_ROOT_METRICS.csv", pd.concat(perroots, ignore_index=True))
    write_csv(OUT / "ROOT_SCALING_SAFETY_METRICS.csv", sf)
    write_csv(OUT / "ROOT_SCALING_TRAIN_TEST_GAP.csv", gp)
    write_json(OUT / "ROOT_SCALING_EVALUATION_COMPLETE.json", {
        "status": "COMPLETE_NO_TEST_SELECTION_CALIBRATION_OR_TUNING",
        "learning_curve_rows": len(lc), "per_root_rows": sum(len(x) for x in perroots),
        "safety_rows": len(sf), "gap_rows": len(gp),
        "checkpoint_manifest_sha256": sha256(OUT / "ROOT_SCALING_CHECKPOINT_MANIFEST.json")})
    print(lc[lc.aggregation == "SEED_MEAN"].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["train", "evaluate"])
    args = ap.parse_args(); train_all() if args.phase == "train" else evaluate_all()
