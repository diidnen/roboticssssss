#!/usr/bin/env python3
"""Small action-relevant GRU smoke test for successor method development.

This intentionally uses only the frozen P7-B TRAIN contexts with complete
matched force branches.  It is not an official DEV/TEST result: those roots
have valid successor probe sequences, but no matched downstream force labels
because the VLA continuation was censored by chunk exhaustion.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = Path("/home/exouser/Tabero")
FROZEN = REPO / "analysis/results/p7b_scientific_main_20260828_000729"
SUCCESSOR = REPO / "analysis/results/minimal_agentic_probing_successor_20260828_161533"
OUT_DEFAULT = SUCCESSOR
FORCES = [5.0, 6.0, 7.0, 8.0]
FEATURES = [
    "t_s", "force_target", "measured_squeeze", "target_normal_force", "measured_fn", "measured_ft",
    "ft_over_fn", "force_imbalance", "force_imbalance_ratio", "gripper_opening", "commanded_tangent_increment_mm",
    "accumulated_displacement_mm", "marker_motion", "marker_tangential", "marker_velocity", "marker_loading_unloading",
    "contact_left", "contact_right", "tactile_ok", "eef_dx", "eef_dy", "eef_dz",
]
TRAIN_ROOTS = {10100, 10101, 10102, 10103, 10104, 10105, 10106, 10107}
DEV_ROOTS = {10108, 10109}
INTERNAL_TEST_ROOTS = {10110}


class MinimalGRU(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.input = nn.Linear(d, 32)
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(33, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x, force):
        z = torch.tanh(self.input(x))
        _, h = self.gru(z)
        h = h[-1]
        f = ((force - 6.5) / 1.5).reshape(-1, 1)
        return self.head(torch.cat([h, f], dim=1)).reshape(-1)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load():
    contexts = pd.read_csv(FROZEN / "P7B_CONTEXT_MANIFEST.csv")
    branches = pd.read_csv(FROZEN / "P7B_BRANCH_MANIFEST.csv")
    contexts = contexts[contexts.query_qualified.astype(int) == 1].copy()
    branches = branches[branches.context_id.isin(set(contexts.context_id))].copy()
    if len(contexts) != 44 or len(branches) != 352:
        raise RuntimeError(f"unexpected frozen matched TRAIN data: contexts={len(contexts)} branches={len(branches)}")
    seq = {}
    for _, row in contexts.iterrows():
        q = pd.read_csv(row.query_telemetry_path)
        arr = q.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
        if len(arr) == 0:
            raise RuntimeError(f"empty query sequence {row.context_id}")
        seq[row.context_id] = arr
    # Collapse policy repeats so a context/force is one supervision unit.
    labels = branches.groupby(["context_id", "root_group_id", "requested_force_N"], as_index=False).full_task_success_y.mean()
    return contexts, labels, seq


def tensors(contexts, labels, seq, ids, mean, std, maxlen):
    xs, fs, ys, cids = [], [], [], []
    by_id = {x: seq[x] for x in ids}
    for cid in ids:
        a = (by_id[cid] - mean) / std
        z = np.zeros((maxlen, a.shape[1]), np.float32)
        z[: len(a)] = a
        for _, row in labels[labels.context_id == cid].iterrows():
            xs.append(z)
            fs.append(float(row.requested_force_N))
            ys.append(float(row.full_task_success_y))
            cids.append(cid)
    return torch.tensor(np.asarray(xs)), torch.tensor(np.asarray(fs), dtype=torch.float32), torch.tensor(np.asarray(ys), dtype=torch.float32), cids


def evaluate(model, ids, labels, seq, mean, std, maxlen):
    model.eval()
    rows = []
    with torch.no_grad():
        for cid in ids:
            a = (seq[cid] - mean) / std
            z = np.zeros((maxlen, a.shape[1]), np.float32)
            z[: len(a)] = a
            x = torch.tensor(z)[None].repeat(len(FORCES), 1, 1)
            f = torch.tensor(FORCES)
            p = torch.sigmoid(model(x, f)).numpy()
            actual = labels[labels.context_id == cid].set_index("requested_force_N").full_task_success_y
            for force, pred in zip(FORCES, p):
                rows.append({"context_id": cid, "root_group_id": int(labels[labels.context_id == cid].root_group_id.iloc[0]), "force_N": force, "pred_success": float(pred), "actual_success": float(actual.get(force, np.nan))})
    return rows


def nll(rows, pkey="pred_success"):
    y = np.asarray([r["actual_success"] for r in rows], float)
    p = np.clip(np.asarray([r[pkey] for r in rows], float), 1e-5, 1 - 1e-5)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    out = args.out.resolve()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    contexts, labels, seq = load()
    train_ids = contexts[contexts.root_group_id.isin(TRAIN_ROOTS)].context_id.tolist()
    dev_ids = contexts[contexts.root_group_id.isin(DEV_ROOTS)].context_id.tolist()
    test_ids = contexts[contexts.root_group_id.isin(INTERNAL_TEST_ROOTS)].context_id.tolist()
    maxlen = max(len(seq[x]) for x in seq)
    all_train = np.concatenate([seq[x] for x in train_ids])
    mean = all_train.mean(0)
    std = all_train.std(0)
    std[std < 1e-6] = 1.0
    train_x, train_f, train_y, _ = tensors(contexts, labels, seq, train_ids, mean, std, maxlen)
    model = MinimalGRU(len(FEATURES))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = None
    best_dev = float("inf")
    for epoch in range(1, 81):
        model.train()
        logits = model(train_x, train_f)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if epoch % 10 == 0:
            dev_rows = evaluate(model, dev_ids, labels, seq, mean, std, maxlen)
            score = nll(dev_rows)
            if score < best_dev:
                best_dev = score
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best is None:
        raise RuntimeError("no model checkpoint selected")
    model.load_state_dict(best)
    train_rows = evaluate(model, train_ids, labels, seq, mean, std, maxlen)
    dev_rows = evaluate(model, dev_ids, labels, seq, mean, std, maxlen)
    test_rows = evaluate(model, test_ids, labels, seq, mean, std, maxlen)

    # No-probe prior is fit only from TRAIN contexts and uses the same fixed
    # force penalty as the GRU planner.
    prior = labels[labels.root_group_id.isin(TRAIN_ROOTS)].groupby("requested_force_N").full_task_success_y.mean().to_dict()
    penalty = 0.02
    selection = []
    for cid in test_ids + dev_ids:
        rr = [r for r in (dev_rows if cid in dev_ids else test_rows) if r["context_id"] == cid]
        pp = {r["force_N"]: r["pred_success"] for r in rr}
        actual = {r["force_N"]: r["actual_success"] for r in rr}
        prior_p = {f: float(prior.get(f, 0.0)) for f in FORCES}
        gru_force = max(FORCES, key=lambda f: (pp[f] - penalty * (f - 5.0), -f))
        prior_force = max(FORCES, key=lambda f: (prior_p[f] - penalty * (f - 5.0), -f))
        selection += [{"split": "internal_DEV" if cid in dev_ids else "internal_TEST", "context_id": cid, "root_group_id": rr[0]["root_group_id"], "model": "probe_GRU", "selected_force_N": gru_force, "selected_actual_success": actual[gru_force]},
                      {"split": "internal_DEV" if cid in dev_ids else "internal_TEST", "context_id": cid, "root_group_id": rr[0]["root_group_id"], "model": "no_probe_prior", "selected_force_N": prior_force, "selected_actual_success": actual[prior_force]}]
    all_metrics = [{"split": "internal_TRAIN", "n": len(train_rows), "nll": nll(train_rows)}, {"split": "internal_DEV", "n": len(dev_rows), "nll": nll(dev_rows)}, {"split": "internal_TEST", "n": len(test_rows), "nll": nll(test_rows)}]
    write_json(out / "MINIMAL_GRU_CONFIG.json", {"architecture": "Linear(21,32)->GRU(32,32)->MLP(33,16,1)", "features": FEATURES, "target": "full_task_success_y conditioned on candidate force", "forces_N": FORCES, "train_roots": sorted(TRAIN_ROOTS), "internal_dev_roots": sorted(DEV_ROOTS), "internal_test_roots": sorted(INTERNAL_TEST_ROOTS), "official_dev_test_labels": False, "force_penalty": penalty})
    write_json(out / "MINIMAL_GRU_NORMALIZATION.json", {"fit": "internal TRAIN roots only", "mean": mean.tolist(), "std": std.tolist()})
    torch.save(model.state_dict(), out / "MINIMAL_GRU.pt")
    write_csv(out / "MINIMAL_GRU_PREDICTIONS.csv", [{"split": "internal_TRAIN", **r} for r in train_rows] + [{"split": "internal_DEV", **r} for r in dev_rows] + [{"split": "internal_TEST", **r} for r in test_rows])
    write_csv(out / "MINIMAL_FORCE_SELECTION.csv", selection)
    write_json(out / "MINIMAL_GRU_RESULT.json", {"status": "INTERNAL_ROOT_HELD_OUT_SMOKE_COMPLETE", "official_dev_test_evaluable": False, "metrics": all_metrics, "train_contexts": len(train_ids), "dev_contexts": len(dev_ids), "test_contexts": len(test_ids), "train_samples": len(train_rows), "dev_samples": len(dev_rows), "test_samples": len(test_rows), "selection_rows": len(selection), "leakage_audit": {"hidden_friction_input": False, "root_id_input": False, "branch_outcome_input": False, "train_test_root_overlap": False}, "caveat": "P7-B TRAIN roots were subdivided for smoke testing; official DEV/TEST successor roots have no matched force labels."})
    (out / "MINIMAL_GRU_REPORT.md").write_text("""# Minimal GRU smoke test\n\nThis is a development smoke test, not the successor paper result. It uses 44 valid frozen P7-B TRAIN contexts and 352 matched force branches, subdivided by root into internal TRAIN/DEV/TEST groups. The official successor DEV/TEST roots have valid probe sequences but no matched downstream force labels because the continuation collector was censored by VLA chunk exhaustion.\n\nArchitecture: `21-channel sequence -> Linear(32) -> GRU(32) -> small force-conditioned MLP head`. Target: candidate-force downstream success. Hidden friction, root id, and future branch outcomes are excluded from the input.\n\nThe resulting metrics and paired internal force selections are in `MINIMAL_GRU_RESULT.json`, `MINIMAL_GRU_PREDICTIONS.csv`, and `MINIMAL_FORCE_SELECTION.csv`. No official held-out generalization, downstream improvement, or agentic trigger claim is made.\n""", encoding="utf-8")
    print(json.dumps({"status": "INTERNAL_ROOT_HELD_OUT_SMOKE_COMPLETE", "metrics": all_metrics, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
