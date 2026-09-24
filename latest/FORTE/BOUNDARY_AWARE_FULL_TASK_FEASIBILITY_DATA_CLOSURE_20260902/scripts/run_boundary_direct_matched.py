#!/usr/bin/env python3
"""Matched OLD versus OLD+BOUNDARY Direct training, fail-closed by default.

The implementation reuses the authoritative FeasibilityOnly architecture and
tensor construction, but fixes the per-epoch sampled-example count to the OLD
arm count so augmentation cannot silently receive more optimizer steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path("/home/exouser/FORTE")
sys.path.insert(0, str(ROOT))
import pooled_joint_novisual_current as pooled  # noqa: E402
import task0_visual_context_early as early  # noqa: E402

OUT = ROOT / "BOUNDARY_AWARE_FULL_TASK_FEASIBILITY_DATA_CLOSURE_20260902"
MANIFEST = OUT / "BOUNDARY_AUGMENTED_DATASET_MANIFEST.json"
PROTOCOL = OUT / "BOUNDARY_DIRECT_MATCHED_TRAINING_PROTOCOL.json"
TASKS = (0, 1, 5, 6)
SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gate() -> dict:
    reasons = []
    if not PROTOCOL.exists():
        reasons.append("matched training protocol missing")
    if not MANIFEST.exists():
        reasons.append("boundary-augmented manifest missing")
        return {"allowed": False, "reasons": reasons}
    m = json.loads(MANIFEST.read_text())
    if m.get("status") != "COMPLETE_TRAIN_DEV_BOUNDARY_AUGMENTATION":
        reasons.append(f"augmented manifest status={m.get('status')}")
    if int(m.get("new_boundary_branches", 0)) <= 0:
        reasons.append("no new boundary branches")
    if not m.get("telemetry_complete", False):
        reasons.append("new boundary telemetry incomplete")
    if not m.get("state_hash_parity_complete", False):
        reasons.append("state-hash parity incomplete")
    if set(m.get("tasks", [])) != set(TASKS):
        reasons.append("four-task coverage incomplete")
    new_path = m.get("new_data", {}).get("manifest")
    if not new_path or not Path(new_path).exists():
        reasons.append("new boundary branch manifest missing")
    return {"allowed": not reasons, "reasons": reasons, "manifest": str(MANIFEST), "new_manifest": new_path}


def load_new(tpi, cmap, old_traces, path: Path):
    d = pd.read_csv(path)
    required = {
        "branch_id", "context_id", "task", "root_id", "root_index", "friction_band", "friction",
        "force_N", "repeat", "full_task_success", "telemetry_path", "acquisition_type", "state_parity",
    }
    if not required <= set(d.columns):
        raise RuntimeError(f"new manifest missing columns: {sorted(required-set(d.columns))}")
    if set(d.acquisition_type) != {"BOUNDARY_SEEKING"} or not d.state_parity.astype(int).eq(1).all():
        raise RuntimeError("new lineage or state parity invalid")
    if d.branch_id.duplicated().any() or set(d.branch_id) & {x.branch_id for x in old_traces}:
        raise RuntimeError("new branch IDs are not unique/disjoint")
    traces, meta = [], {}
    old_by_context = {}
    for tr in old_traces:
        old_by_context.setdefault(tr.context_id, []).append(float(tr.force))
    force_rank = {}
    for cid, q in d.groupby("context_id"):
        vals = sorted(set(old_by_context.get(cid, []) + q.force_N.astype(float).tolist()))
        for rank, force in enumerate(vals):
            force_rank[(cid, force)] = rank
    for r in d.to_dict("records"):
        cid, bid = str(r["context_id"]), str(r["branch_id"])
        if cid not in cmap:
            raise RuntimeError(f"new context not in frozen context map: {cid}")
        telemetry = Path(str(r["telemetry_path"]))
        if not telemetry.exists():
            raise RuntimeError(f"new telemetry missing: {telemetry}")
        td = pd.read_csv(telemetry)
        needed = {"phase", "object_z_analysis_only", "cmd_x", "cmd_y", "cmd_z"}
        if len(td) < early.H + 1 or not needed <= set(td.columns):
            raise RuntimeError(f"invalid new telemetry: {telemetry}")
        state, mask = tpi.state_from(td)
        state, mask = state.copy(), mask.copy()
        state[0] = cmap[cid]["preprobe_state"]
        mask[0] = cmap[cid]["preprobe_mask"]
        force, mu = float(r["force_N"]), float(r["friction"])
        nominal = tpi.nominal_from(td, 0, force, mu, state, mask)
        tr = tpi.Trace(
            bid, cid, str(r["root_id"]), int(r["task"]), "TRAIN", force, mu,
            int(r["full_task_success"]), "continuous", telemetry, state, mask, nominal,
            td.phase.astype(str).tolist(), 1.0, "BOUNDARY_SEEKING",
        )
        traces.append(tr)
        meta[bid] = early.Meta(
            bid, cid, int(r["task"]), str(r["root_id"]), str(r["friction_band"]),
            int(r["full_task_success"]), force, int(r["repeat"]), force_rank[(cid, force)],
        )
    return traces, meta, d


def fold_pairs(old_traces):
    roots = {task: sorted({t.root_id for t in old_traces if int(t.task) == task}) for task in TASKS}
    if any(len(x) != 6 for x in roots.values()):
        raise RuntimeError("expected six roots per task")
    return {f: {(task, root) for task in TASKS for root in roots[task][f::3]} for f in FOLDS}


def train_fixed_budget(full, traces, segs, norm, cmap, meta, seed: int, examples_per_epoch: int):
    early.seed_everything(seed)
    model = full.FeasibilityOnly().to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=early.LR, weight_decay=early.WEIGHT_DECAY)
    history, steps = [], 0
    for epoch in range(1, early.EPOCHS + 1):
        ids = early.sampled_ids(traces, meta, seed, epoch, n=examples_per_epoch)
        losses = []
        model.train()
        for start in range(0, len(ids), early.BATCH):
            q = [traces[int(i)] for i in ids[start:start+early.BATCH]]
            step, cond, _, y = early.branch_tensors(q, segs, norm, cmap, "cpu")
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(step, cond), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); steps += 1; losses.append(float(loss.item()))
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses)), "optimizer_steps": steps})
    model.eval()
    return model, history, steps


def predict(model, traces, segs, norm, cmap):
    out = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(traces), 256):
            q = traces[start:start+256]
            step, cond, _, _ = early.branch_tensors(q, segs, norm, cmap, "cpu")
            p = torch.sigmoid(model(step, cond)).cpu().numpy()
            out.update({tr.branch_id: float(v) for tr, v in zip(q, p)})
    return out


def run() -> None:
    g = gate()
    if not g["allowed"]:
        print(json.dumps(g, indent=2)); raise SystemExit(75)
    scratch = OUT / "_matched_training_audit"
    scratch.mkdir(exist_ok=True)
    tpi, cf, full, cmap, old, old_meta, audits, _, _, _ = pooled.load_population(scratch)
    new, new_meta, _ = load_new(tpi, cmap, old, Path(g["new_manifest"]))
    all_traces = old + new
    all_meta = {**old_meta, **new_meta}
    held_pairs = fold_pairs(old)
    pred_rows, hist_rows, ck_rows = [], [], []
    ckdir = OUT / "matched_direct_checkpoints"; ckdir.mkdir(exist_ok=True)
    for fold in FOLDS:
        held = held_pairs[fold]
        old_train = [t for t in old if (int(t.task), t.root_id) not in held]
        eval_rows = [t for t in all_traces if (int(t.task), t.root_id) in held]
        arms = [("OLD_DATA_ONLY", old_train), ("OLD_PLUS_BOUNDARY_DATA", [t for t in all_traces if (int(t.task), t.root_id) not in held])]
        for arm, train in arms:
            train_meta = {t.branch_id: all_meta[t.branch_id] for t in train}
            segs, norm = early.build_segments_and_norm(cf, tpi, train)
            eval_segs = {t.branch_id: cf.build_seg(tpi, t, t.force, early.H) for t in eval_rows}
            all_segs = {**segs, **eval_segs}
            examples = len(old_train)
            for seed in SEEDS:
                model, hist, steps = train_fixed_budget(full, train, segs, norm, cmap, train_meta, seed, examples)
                expected_steps = early.EPOCHS * math.ceil(examples / early.BATCH)
                if steps != expected_steps:
                    raise RuntimeError("optimizer-step mismatch")
                probs = predict(model, eval_rows, all_segs, norm, cmap)
                for tr in eval_rows:
                    m = all_meta[tr.branch_id]
                    pred_rows.append({
                        "arm": arm, "fold": fold, "seed": seed, "branch_id": tr.branch_id,
                        "context_id": tr.context_id, "task": int(tr.task), "root_id": tr.root_id,
                        "friction_band": m.friction_band, "force_N": float(tr.force), "repeat": int(m.repeat),
                        "success": int(tr.outcome), "acquisition_type": "BOUNDARY_SEEKING" if tr.branch_id in new_meta else "UNIFORM_OLD",
                        "p_success": probs[tr.branch_id],
                    })
                hist_rows.extend({"arm": arm, "fold": fold, "seed": seed, "train_branches": len(train), "examples_per_epoch": examples, **x} for x in hist)
                path = ckdir / f"{arm}_fold{fold}_seed{seed}.pt"
                torch.save({
                    "arm": arm, "fold": fold, "seed": seed, "state_dict": model.state_dict(),
                    "epochs": early.EPOCHS, "batch_size": early.BATCH, "lr": early.LR,
                    "weight_decay": early.WEIGHT_DECAY, "optimizer_steps": steps,
                    "examples_per_epoch": examples, "train_branches": len(train),
                    "normalization": {k: v.tolist() for k, v in zip(["x_mean", "x_std", "y_mean", "y_std"], norm)},
                }, path)
                ck_rows.append({"arm": arm, "fold": fold, "seed": seed, "path": str(path), "sha256": sha(path), "optimizer_steps": steps})
    pd.DataFrame(pred_rows).to_csv(OUT / "BOUNDARY_DIRECT_MATCHED_OOF_PREDICTIONS.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(OUT / "BOUNDARY_DIRECT_MATCHED_TRAINING_HISTORY.csv", index=False)
    pd.DataFrame(ck_rows).to_csv(OUT / "BOUNDARY_DIRECT_MATCHED_CHECKPOINTS.csv", index=False)
    (OUT / "BOUNDARY_DIRECT_MATCHED_TRAINING_COMPLETE.json").write_text(json.dumps({
        "status": "COMPLETE", "arms": ["OLD_DATA_ONLY", "OLD_PLUS_BOUNDARY_DATA"],
        "folds": list(FOLDS), "seeds": list(SEEDS), "old_branches": len(old), "new_branches": len(new),
        "matched_optimizer_steps": sorted(set(x["optimizer_steps"] for x in ck_rows)),
        "predictions": str(OUT / "BOUNDARY_DIRECT_MATCHED_OOF_PREDICTIONS.csv"),
        "task1_caveat": audits[1], "test_used": False,
    }, indent=2, default=str) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        run(); return
    result = gate(); print(json.dumps(result, indent=2))
    if not result["allowed"]:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
