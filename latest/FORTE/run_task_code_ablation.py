#!/usr/bin/env python3
"""Audit the unseen one-hot coordinate in the B=0 Shared Direct diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_shared_physical_transfer as prior

torch.set_num_threads(1)

TARGET = 5
SOURCE = (0, 1, 6)
TASKS = (0, 1, 5, 6)
CODES = {
    "target_onehot": np.array([0, 0, 1, 0], np.float32),
    "zero_task_code": np.zeros(4, np.float32),
    "source_mean_code": np.array([1 / 3, 1 / 3, 0, 1 / 3], np.float32),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def score_code(model, traces, segs, norm, mu_map, code):
    xm, xs = norm[:2]
    xs_list = []
    cs = []
    for t in traces:
        a = segs[t.branch_id].x.copy()
        a[:, 13:17] = code
        x = (a - xm) / xs
        xs_list.append(x)
    x = np.stack(xs_list).astype(np.float32)
    step = torch.tensor(x[:, :, :17])
    cond = torch.tensor(x[:, 0, 17:])
    with torch.no_grad():
        return prior.af.sigmoid(model(step, cond).numpy()).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--source-root", type=Path,
        default=Path("/home/exouser/FORTE/activeforcing_shared_physical_transfer_20260901_094722"),
    )
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = prior.load_current(out, "population")
    _, _, _, _, traces, meta, _, _, segs, md, rf = data
    zero_h = {cid: np.zeros(1, np.float32) for cid in md.context_id.unique()}
    gt_mu = dict(zip(md.context_id, md.mu.astype(float)))
    rows = []

    for fold in prior.FOLDS:
        src_idx = np.flatnonzero((md.task.isin(SOURCE) & (rf != fold) & (md.repeat == 1)).to_numpy())
        held_idx = np.flatnonzero(((md.task == TARGET) & (rf == fold)).to_numpy())
        assert len(src_idx) == 180 and len(held_idx) == 60
        train = [traces[i] for i in src_idx]
        held = [traces[i] for i in held_idx]
        norm = prior.af.xnorm(train, segs, gt_mu)
        for seed in prior.SEEDS:
            model = prior.af.train_direct(train, segs, norm, gt_mu, zero_h, meta, seed, False)
            for name, code in CODES.items():
                p = score_code(model, held, segs, norm, gt_mu, code)
                sel = prior.evaluate_selected(md, held_idx, p)
                rows.append({
                    "target_task": TARGET,
                    "source_tasks": ",".join(map(str, SOURCE)),
                    "fold": fold,
                    "seed": seed,
                    "code": name,
                    "episodes": len(sel),
                    "successes": int(sel.success.sum()),
                    "sr": float(sel.success.mean()),
                    "underforce": float(sel.under_force.mean()),
                    "mean_force": float(sel.selected_force_N.mean()),
                    "utility": float(sel.realized_utility.mean()),
                })

    per = pd.DataFrame(rows)
    per.to_csv(out / "TASK_CODE_ABLATION_PER_SEED.csv", index=False)
    agg = (
        per.groupby(["code", "target_task", "source_tasks"], as_index=False)
        .agg(
            episodes=("episodes", "sum"),
            successes=("successes", "sum"),
            sr_mean=("sr", "mean"),
            sr_std=("sr", lambda x: float(np.std(x, ddof=0))),
            underforce_mean=("underforce", "mean"),
            utility_mean=("utility", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    agg.to_csv(out / "TASK_CODE_ABLATION.csv", index=False)

    prior_t5 = pd.read_csv(args.source_root / "NEW_TASK_FEWSHOT_TRANSFER.csv")
    prior_t5 = prior_t5[(prior_t5.target_task == TARGET) & (prior_t5.physics_estimator == "GT") & (prior_t5.budget == 0)]
    rerun = (
        per[per.code == "target_onehot"]
        .groupby("seed", as_index=False)
        .agg(successes=("successes", "sum"), episodes=("episodes", "sum"))
    )
    rerun["sr"] = rerun.successes / rerun.episodes
    normal = rerun.merge(prior_t5[["seed", "sr"]].rename(columns={"sr": "prior_sr"}), on="seed", validate="many_to_one")
    normal["delta_vs_prior"] = normal.sr - normal.prior_sr
    normal.to_csv(out / "CHECK_TARGET_CODE_AGAINST_PRIOR.csv", index=False)
    if not np.allclose(normal.sr, normal.prior_sr, atol=1e-12):
        raise RuntimeError("target one-hot ablation does not reproduce prior Task5 B0")

    split_audit = {
        "status": "COMPLETE",
        "target_task": TARGET,
        "source_tasks": list(SOURCE),
        "budget": 0,
        "source_rollouts_per_fold": 180,
        "eval_rollouts_per_fold": 60,
        "eval_episodes_per_fold": 12,
        "codes": {k: v.tolist() for k, v in CODES.items()},
        "target_onehot_unseen_during_training": True,
        "normalization_fit_on_source_only": True,
        "gt_friction_privileged": True,
        "prior_acquisition_sha256": sha(args.source_root / "TARGET_ACQUISITION_TRAJECTORIES.csv"),
        "target_onehot_reproduces_prior": True,
        "interpretation": "B0 target-code sensitivity is an encoding diagnostic, not a semantic transfer claim.",
    }
    (out / "TASK_CODE_ABLATION_AUDIT.json").write_text(json.dumps(split_audit, indent=2, sort_keys=True) + "\n")
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
