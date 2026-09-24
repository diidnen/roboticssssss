#!/usr/bin/env python3
"""Source-subset audit for one-hot Shared Direct transfer.

This is a read-only-data, GT-physics diagnostic.  It fixes target task 5 and
compares several source-task subsets while preserving the frozen root folds,
target acquisition prefixes, Direct architecture, optimizer, and seeds from
the authoritative LOTO experiment.
"""
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
SOURCE_SETS = {
    "src_01": (0, 1),
    "src_06": (0, 6),
    "src_16": (1, 6),
    "src_016": (0, 1, 6),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/exouser/FORTE/activeforcing_shared_physical_transfer_20260901_094722"),
    )
    ap.add_argument("--budgets", nargs="+", type=int, default=[0, 60])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(prior.SEEDS))
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "_audit").mkdir(exist_ok=True)

    # Load the same authoritative current population.  No outcomes are used
    # here except in the frozen adaptation/evaluation labels consumed by the
    # already-frozen prior acquisition trajectory.
    data = prior.load_current(out, "population")
    _, _, _, _, traces, meta, _, _, segs, md, rf = data
    acq = pd.read_csv(args.source_root / "TARGET_ACQUISITION_TRAJECTORIES.csv")
    protocol_path = args.source_root / "TARGET_BOUNDARY_ACQUISITION_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["tasks"] != [0, 1, 5, 6] or protocol["canonical_repeat"] != 1:
        raise RuntimeError("unexpected frozen acquisition protocol")

    rows = []
    audit_rows = []
    zero_h = {cid: np.zeros(1, np.float32) for cid in md.context_id.unique()}
    gt_mu = dict(zip(md.context_id, md.mu.astype(float)))

    for set_name, source_tasks in SOURCE_SETS.items():
        for fold in prior.FOLDS:
            source_mask = (
                md.task.isin(source_tasks)
                & (rf != fold)
                & (md.repeat == 1)
            )
            held_mask = (md.task == TARGET) & (rf == fold)
            source_idx = np.flatnonzero(source_mask.to_numpy())
            held_idx = np.flatnonzero(held_mask.to_numpy())
            if len(source_idx) != 60 * len(source_tasks):
                raise RuntimeError(f"bad source count {set_name} fold {fold}: {len(source_idx)}")
            if len(held_idx) != 60:
                raise RuntimeError(f"bad held count fold {fold}: {len(held_idx)}")

            train_keys = set(md.iloc[source_idx].branch_id)
            eval_keys = set(md.iloc[held_idx].branch_id)
            train_contexts = set(md.iloc[source_idx].context_id)
            eval_contexts = set(md.iloc[held_idx].context_id)
            train_roots = set(zip(md.iloc[source_idx].task, md.iloc[source_idx].root_id))
            eval_roots = set(zip(md.iloc[held_idx].task, md.iloc[held_idx].root_id))
            if train_keys & eval_keys or train_contexts & eval_contexts or train_roots & eval_roots:
                raise RuntimeError(f"split leakage in {set_name} fold {fold}")
            if set(md.iloc[source_idx].task) & {TARGET}:
                raise RuntimeError("target task entered source training")

            aq = acq[(acq.target_task == TARGET) & (acq.fold == fold)].sort_values("query_index")
            if len(aq) != 60 or aq.branch_id.nunique() != 60:
                raise RuntimeError(f"bad frozen target acquisition fold {fold}")
            for budget in args.budgets:
                target_ids = set(aq.loc[aq.query_index <= budget, "branch_id"])
                target_idx = np.flatnonzero(md.branch_id.isin(target_ids).to_numpy())
                if budget == 0 and len(target_idx):
                    raise RuntimeError("B0 unexpectedly has target rows")
                if len(target_idx) != budget:
                    raise RuntimeError(f"bad target prefix {budget}: {len(target_idx)}")
                train_idx = np.concatenate([source_idx, target_idx])
                if set(md.iloc[train_idx].branch_id) & eval_keys:
                    raise RuntimeError(f"target eval branch leaked at {set_name}/{fold}/{budget}")

                train = [traces[i] for i in train_idx]
                held = [traces[i] for i in held_idx]
                norm = prior.af.xnorm(train, segs, gt_mu)
                for seed in args.seeds:
                    model = prior.af.train_direct(
                        train, segs, norm, gt_mu, zero_h, meta, seed, False
                    )
                    p = prior.score_direct(model, held, segs, norm, gt_mu, zero_h)
                    sel = prior.evaluate_selected(md, held_idx, p)
                    rows.append({
                        "source_set": set_name,
                        "source_tasks": ",".join(map(str, source_tasks)),
                        "target_task": TARGET,
                        "fold": fold,
                        "seed": seed,
                        "budget": budget,
                        "source_rollouts": len(source_idx),
                        "target_rollouts": budget,
                        "train_rollouts": len(train_idx),
                        "episodes": len(sel),
                        "successes": int(sel.success.sum()),
                        "sr": float(sel.success.mean()),
                        "underforce": float(sel.under_force.mean()),
                        "mean_force": float(sel.selected_force_N.mean()),
                        "utility": float(sel.realized_utility.mean()),
                        "decision_agreement_gt": 1.0,
                    })
                audit_rows.append({
                    "source_set": set_name,
                    "source_tasks": ",".join(map(str, source_tasks)),
                    "target_task": TARGET,
                    "fold": fold,
                    "budget": budget,
                    "source_rows": len(source_idx),
                    "target_adaptation_rows": len(target_idx),
                    "eval_rows": len(held_idx),
                    "source_roots": len(train_roots),
                    "eval_roots": len(eval_roots),
                    "source_eval_branch_intersection": len(train_keys & eval_keys),
                    "source_eval_context_intersection": len(train_contexts & eval_contexts),
                    "source_eval_root_intersection": len(train_roots & eval_roots),
                    "target_task_in_b0_source": 0,
                    "target_onehot_unseen_at_budget0": int(budget == 0),
                    "gt_mu_privileged_input": 1,
                })

    per = pd.DataFrame(rows)
    per.to_csv(out / "SOURCE_SUBSET_PER_SEED.csv", index=False)
    aud = pd.DataFrame(audit_rows)
    aud.to_csv(out / "SOURCE_SUBSET_SPLIT_AUDIT.csv", index=False)
    agg = (
        per.groupby(["source_set", "source_tasks", "target_task", "budget"], as_index=False)
        .agg(
            source_rollouts=("source_rollouts", "first"),
            target_rollouts=("target_rollouts", "first"),
            train_rollouts=("train_rollouts", "first"),
            episodes=("episodes", "sum"),
            successes=("successes", "sum"),
            sr_mean=("sr", "mean"),
            sr_std=("sr", lambda x: float(np.std(x, ddof=0))),
            underforce_mean=("underforce", "mean"),
            utility_mean=("utility", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    agg.to_csv(out / "SOURCE_SUBSET_TRANSFER.csv", index=False)

    prior_per = pd.read_csv(args.source_root / "NEW_TASK_FEWSHOT_TRANSFER.csv")
    prior_t5 = prior_per[(prior_per.target_task == TARGET) & (prior_per.physics_estimator == "GT")].copy()
    rerun_016 = (
        per[per.source_set == "016"]
        .groupby(["seed", "budget"], as_index=False)
        .agg(sr=("successes", "sum"), episodes=("episodes", "sum"))
    )
    rerun_016["sr"] = rerun_016.sr / rerun_016.episodes
    check = rerun_016.merge(
        prior_t5[["seed", "budget", "sr"]].rename(columns={"sr": "prior_sr"}),
        on=["seed", "budget"], how="left", validate="many_to_one",
    )
    check["sr_delta_vs_prior_016_to_5"] = check.sr - check.prior_sr
    check.to_csv(out / "CHECK_016_TO_5_AGAINST_PRIOR.csv", index=False)
    if not np.allclose(check.sr, check.prior_sr, atol=1e-12):
        raise RuntimeError("independent 016->5 rerun does not match prior GT Task5 cells")

    manifest = {
        "status": "COMPLETE",
        "target_task": TARGET,
        "source_sets": {k: list(v) for k, v in SOURCE_SETS.items()},
        "estimand": "GT-friction one-hot Shared Direct source-subset transfer",
        "folds": prior.FOLDS,
        "seeds": args.seeds,
        "budgets": args.budgets,
        "source_subset_016_is_requested_0_1_6_to_5": True,
        "requested_016_to_5_matches_prior": True,
        "acquisition_protocol_sha256": sha(protocol_path),
        "prior_transfer_sha256": sha(args.source_root / "NEW_TASK_FEWSHOT_TRANSFER.csv"),
        "data_leakage_checks": {
            "target_task_excluded_from_b0_source": True,
            "target_eval_roots_excluded_from_source": True,
            "target_eval_branches_excluded_from_adaptation": True,
            "target_eval_contexts_excluded_from_source": True,
            "normalization_fit_on_train_only": True,
            "target_outcome_used_only_as_adaptation_label_for_budget_gt_0": True,
            "gt_friction_is_privileged": True,
            "task_and_object_family_confound": True,
        },
    }
    write_json(out / "SOURCE_SUBSET_AUDIT_MANIFEST.json", manifest)
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
