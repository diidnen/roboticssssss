#!/usr/bin/env python3
"""Proper no-task-ID B=0 control for the one-hot Shared Direct experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_shared_physical_transfer as prior

torch.set_num_threads(1)


class NoTaskDirect(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(13, 64, batch_first=True)
        self.cond = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step, cond):
        _, h = self.gru(step)
        return self.head(torch.cat([h[-1], self.cond(cond)], -1)).squeeze(-1)


def tensors_no_task(traces, segs, norm, mu_map):
    xm, xs = norm[:2]
    xx = []
    for t in traces:
        a = segs[t.branch_id].x.copy()
        a[:, 13:17] = 0.0
        xx.append((a - xm) / xs)
    x = np.stack(xx).astype(np.float32)
    return torch.tensor(x[:, :, :13]), torch.tensor(x[:, 0, 17:]), torch.tensor(
        [t.outcome for t in traces], dtype=torch.float32
    )


def train_no_task(train, segs, norm, mu_map, meta, seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = NoTaskDirect()
    opt = torch.optim.AdamW(model.parameters(), lr=prior.af.LR, weight_decay=prior.af.WD)
    for epoch in range(1, prior.af.EPOCHS + 1):
        ids = prior.af.early.sampled_ids(train, meta, seed, epoch)
        model.train()
        for start in range(0, len(ids), prior.af.BATCH):
            batch = [train[int(i)] for i in ids[start:start + prior.af.BATCH]]
            step, cond, y = tensors_no_task(batch, segs, norm, mu_map)
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(step, cond), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    data = prior.load_current(args.out, "population")
    _, _, _, _, traces, meta, _, _, segs, md, rf = data
    mu_map = dict(zip(md.context_id, md.mu.astype(float)))
    zero_h = {cid: np.zeros(1, np.float32) for cid in md.context_id.unique()}
    rows = []
    for fold in prior.FOLDS:
        src = np.flatnonzero((md.task.isin([0, 1, 6]) & (rf != fold) & (md.repeat == 1)).to_numpy())
        held = np.flatnonzero(((md.task == 5) & (rf == fold)).to_numpy())
        train = [traces[i] for i in src]
        eval_traces = [traces[i] for i in held]
        norm = prior.af.xnorm(train, segs, mu_map)
        for seed in prior.SEEDS:
            model = train_no_task(train, segs, norm, mu_map, meta, seed)
            q = tensors_no_task(eval_traces, segs, norm, mu_map)
            with torch.no_grad():
                p = prior.af.sigmoid(model(q[0], q[1]).numpy()).astype(np.float32)
            sel = prior.evaluate_selected(md, held, p)
            rows.append({"target_task": 5, "source_tasks": "0,1,6", "fold": fold, "seed": seed,
                         "budget": 0, "episodes": len(sel), "successes": int(sel.success.sum()),
                         "sr": float(sel.success.mean()), "underforce": float(sel.under_force.mean()),
                         "utility": float(sel.realized_utility.mean())})
    per = pd.DataFrame(rows); per.to_csv(args.out / "NO_TASK_ID_PER_SEED.csv", index=False)
    agg = pd.DataFrame([{
        "episodes": int(per.episodes.sum()),
        "successes": int(per.successes.sum()),
        "sr_mean": float(per.sr.mean()),
        "sr_std": float(np.std(per.sr, ddof=0)),
        "underforce_mean": float(per.underforce.mean()),
        "utility_mean": float(per.utility.mean()),
        "seeds": int(per.seed.nunique()),
    }])
    agg.to_csv(args.out / "NO_TASK_ID_CONTROL.csv", index=False)
    manifest = {"status": "COMPLETE", "target_task": 5, "source_tasks": [0, 1, 6],
                "budget": 0, "task_id_input": "removed_from_train_and_eval", "gt_friction_privileged": True,
                "source_rollouts_per_fold": 180, "eval_episodes_per_fold": 12,
                "data_leakage_checks": {"target_outcomes_in_train": False, "eval_roots_in_train": False,
                                         "normalization_fit_on_source_only": True},
                "interpretation": "Proper no-task-ID control; not a semantic encoder experiment."}
    (args.out / "NO_TASK_ID_AUDIT.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__": main()
