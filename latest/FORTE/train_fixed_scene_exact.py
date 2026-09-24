#!/usr/bin/env python3
"""Train the existing Direct and Joint backbones on the fixed-scene split.

The network classes, input construction, losses, optimizer, and checkpoint
initialization are imported from the frozen GNP continuous implementation. No
new model or calibration is introduced here. DEV rows are not used for
training or checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FORTE = Path("/home/exouser/FORTE")
SEEDS = [0, 1, 2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def build_population(split_dir: Path, gnp, tpi, cf):
    df = pd.read_csv(split_dir / "FIXED_SCENE_COMMON_DATASET_AUDIT.csv")
    split_map = {}
    for name, split in [("FIXED_SCENE_TRAIN_MANIFEST.csv", "TRAIN"), ("FIXED_SCENE_DEV_MANIFEST.csv", "DEV")]:
        q = pd.read_csv(split_dir / name)
        split_map.update(dict(zip(q.context_id.astype(str), [split] * len(q))))
    traces, meta = [], {}
    for r in df.itertuples(index=False):
        path = Path(str(r.source)).parent / Path(str(r.telemetry_path)).name if hasattr(r, "telemetry_path") else None
        # The re-index deliberately keeps source provenance; recover the
        # original telemetry path by matching branch_id in the authoritative
        # source table instead of guessing from a namespace.
        if path is None or not path.exists():
            src = pd.read_csv(gnp.SOURCE if hasattr(gnp, "SOURCE") else FORTE / "gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv")
            q = src[src.branch_id.astype(str) == str(r.branch_id)]
            if len(q) != 1:
                raise RuntimeError(f"cannot recover telemetry for {r.branch_id}")
            path = Path(str(q.iloc[0].telemetry_path))
        d = pd.read_csv(path)
        state, mask = tpi.state_from(d)
        nominal = tpi.nominal_from(d, int(r.task), float(r.force_N), float(r.mu), state, mask)
        row_split = split_map[str(r.context_id)]
        tr = tpi.Trace(str(r.branch_id), str(r.context_id), str(r.root_id), int(r.task), row_split,
                       float(r.force_N), float(r.mu), int(r.success), "fixed_scene", path,
                       state, mask, nominal, d.phase.astype(str).tolist(), 1.0, "FIXED_SCENE_COMMON_720")
        traces.append(tr)
        meta[tr.branch_id] = gnp.Meta(tr.branch_id, tr.context_id, tr.task, tr.root_id,
                                      str(r.friction_band), tr.outcome, tr.force,
                                      "fixed_scene", "FIXED_SCENE_COMMON_720", int(r.repeat), int(r.stratum_index))
    train = [t for t in traces if t.split == "TRAIN"]
    dev = [t for t in traces if t.split == "DEV"]
    if len(traces) != 720 or len(train) != 480 or len(dev) != 240:
        raise RuntimeError(f"population mismatch all/train/dev={len(traces)}/{len(train)}/{len(dev)}")
    if set(t.context_id for t in train) & set(t.context_id for t in dev):
        raise RuntimeError("context leakage between fixed-scene TRAIN and DEV")
    # Adjacent-force IE pairs use only TRAIN contexts and the exact physical
    # trajectories from the common set. Both repeats are retained.
    grouped = {}
    for t in train:
        grouped.setdefault((t.context_id, int(meta[t.branch_id].repeat_index)), []).append(t)
    pairs = []
    for (cid, rep), q in sorted(grouped.items()):
        q = sorted(q, key=lambda t: t.force)
        for a, b in zip(q[:-1], q[1:]):
            ma = meta[a.branch_id]
            pairs.append(cf.Pair(f"fixed:{cid}:r{rep}:{a.force:.8f}:{b.force:.8f}", f"fixed:{cid}:r{rep}", "TRAIN", "FIXED_SCENE_COMMON_720", cid, a.root_id, a.task, ma.friction_band, a.mu, a.force, b.force, "adjacent", False, a, b))
    return train, dev, meta, pairs


def evaluate(backend: str, seed_models, dev, segs, norm, device, gnp):
    rows = []
    logits = {t.branch_id: [] for t in dev}
    for model in seed_models:
        z = gnp.model_logits(model, backend, dev, segs, norm, device)
        for bid, v in z.items(): logits[bid].append(float(v))
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    for t in dev:
        p = float(np.mean([sigmoid(x) for x in logits[t.branch_id]]))
        scene = {0:"scene_task0_alphabet_soup_1", 1:"scene_task1_cream_cheese_1", 5:"scene_task5_tomato_sauce_1", 6:"scene_task6_butter_1"}[int(t.task)]
        rep = 1 if "_R1_" in t.branch_id else 2 if "_R2_" in t.branch_id else ""
        rows.append({"method": backend, "branch_id": t.branch_id, "context_id": t.context_id, "scene_id": scene, "task": t.task, "mu": t.mu, "force_N": t.force, "repeat": rep, "actual_success": t.outcome, "p_success": p})
    summaries = []
    by = {}
    for r in rows: by.setdefault(r["context_id"], []).append(r)
    for cid, q in sorted(by.items()):
        q = sorted(q, key=lambda x: x["force_N"])
        candidates = [x for x in q if x["p_success"] >= 0.5]
        chosen = min(candidates, key=lambda x: x["force_N"]) if candidates else q[-1]
        same = [x for x in q if abs(x["force_N"] - chosen["force_N"]) < 1e-7]
        summaries.append({"method": backend, "context_id": cid, "task": q[0]["task"], "mu": q[0]["mu"], "selected_force_N": chosen["force_N"], "threshold": 0.5, "fallback_used": int(not candidates), "full_task_success_rate_at_selected_force": float(np.mean([x["actual_success"] for x in same])), "n_repeats_at_selected_force": len(same)})
    return rows, summaries


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--audit", type=Path, required=True); args = ap.parse_args()
    out = args.audit
    gnp = load("gnp_fixed", FORTE / "gnp_style_continuous.py")
    tpi = load("tpi_fixed", gnp.TPI_CODE)
    cf = load("cf_fixed", gnp.CF_CODE)
    full = load("full_fixed", gnp.FULL_CODE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("A100 CUDA required; CPU fallback forbidden")
    torch.set_num_threads(min(4, __import__('os').cpu_count() or 1))
    train, dev, meta, pairs = build_population(out, gnp, tpi, cf)
    all_traces = train + dev
    segs = gnp.start_segments(cf, tpi, all_traces)
    norm = gnp.fit_shared_norm(train, train, {t.branch_id: segs[t.branch_id] for t in train}, tpi)
    norm_obj = {k: v.tolist() for k, v in zip(["x_mean","x_std","y_mean","y_std"], norm)}
    (out / "FIXED_SCENE_TRAIN_NORMALIZATION.json").write_text(json.dumps({"fit_split":"TRAIN","branches":len(train),"normalization":norm_obj}, indent=2)+"\n", encoding="utf-8")
    all_results=[]; summary=[]; checkpoint_rows=[]
    for seed in SEEDS:
        fmodel, flog, fsteps = gnp.train_feas_seed(full, train, segs, norm, meta, device, seed)
        fpath = out / f"FIXED_SCENE_DIRECT_seed{seed}.pt"; torch.save({"method":"ActiveForcing-Direct","backend":"FEASIBILITY_ONLY","seed":seed,"state_dict":fmodel.state_dict(),"normalization":norm_obj,"threshold":0.5,"training_contexts":48,"training_branches":480}, fpath)
        write_csv(out / f"DIRECT_TRAINING_seed{seed}.csv", flog)
        jmodel, jlog, jsteps, units, base_path = gnp.train_joint_seed(full, cf, tpi, train, train, pairs, segs, norm, meta, device, seed)
        jpath = out / f"FIXED_SCENE_JOINT_seed{seed}.pt"; torch.save({"method":"ActiveForcing-Joint","backend":"JOINT","seed":seed,"physics_state_dict":jmodel.physics.state_dict(),"feas_head_state_dict":jmodel.feas_head.state_dict(),"normalization":norm_obj,"threshold":0.5,"training_contexts":48,"training_branches":480,"adjacent_ie_pairs":len(pairs),"initial_checkpoint":str(base_path)}, jpath)
        write_csv(out / f"JOINT_TRAINING_seed{seed}.csv", jlog)
        checkpoint_rows += [{"method":"ActiveForcing-Direct","seed":seed,"checkpoint":str(fpath),"sha256":__import__('hashlib').sha256(fpath.read_bytes()).hexdigest(),"optimizer_steps":fsteps},{"method":"ActiveForcing-Joint","seed":seed,"checkpoint":str(jpath),"sha256":__import__('hashlib').sha256(jpath.read_bytes()).hexdigest(),"optimizer_steps":jsteps,"physical_units":units}]
        dr, ds = evaluate("FEAS", [fmodel], dev, segs, norm, device, gnp); jr, js = evaluate("JOINT", [jmodel], dev, segs, norm, device, gnp)
        all_results += dr + jr; summary += ds + js
    write_csv(out / "FIXED_SCENE_DEV_PREDICTIONS.csv", all_results)
    write_csv(out / "FIXED_SCENE_DEV_RESULTS.csv", summary)
    (out / "FIXED_SCENE_CHECKPOINT_MANIFEST.json").write_text(json.dumps({"status":"TRAINED","device":torch.cuda.get_device_name(0),"train_contexts":48,"dev_contexts":24,"checkpoints":checkpoint_rows,"threshold":0.5,"dev_not_used_for_training_or_selection":True}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status":"TRAINED_DIRECT_AND_JOINT","out":str(out),"device":torch.cuda.get_device_name(0),"train_branches":len(train),"dev_branches":len(dev),"pairs":len(pairs)}, indent=2))


if __name__ == "__main__": main()
