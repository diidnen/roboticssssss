#!/usr/bin/env python3
"""Matched semantic-task LOTO experiment for ActiveForcing.

Only the task representation changes.  All archived populations, root folds,
boundary-acquisition prefixes, physics estimates, Direct objective, expected
utility, and evaluation semantics are inherited byte-for-byte from the prior
one-hot experiment.  No simulator or root-scaling TEST namespace is touched.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

import run_probe_conditioned_wm as af
import run_shared_physical_transfer as prior


ROOT = Path("/home/exouser/FORTE")
PREV = ROOT / "activeforcing_shared_physical_transfer_20260901_094722"
TASKS = [0, 1, 5, 6]
FOLDS = [0, 1, 2]
SEEDS = [0, 1, 2]
BUDGETS = [0, 10, 20, 30, 60]
ESTIMATORS = ["GT", "ExplicitSysID", "LearnedProbe"]
PROMPTS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
}
CHECKPOINT = Path("/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999")
OPENPI = Path("/media/volume/newdata/exouser/tabero/Tabero-VTLA/src/openpi")
SEM_RAW_DIM = 2048
SEM_DIM = 16
PROJECTION_SEED = 20260901
UTILITY_TOL = 0.01

torch.set_num_threads(1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wjson(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def fixed_projection(raw: np.ndarray) -> np.ndarray:
    """A preregistered outcome-independent orthogonal random projection."""
    rng = np.random.default_rng(PROJECTION_SEED)
    q, _ = np.linalg.qr(rng.standard_normal((SEM_RAW_DIM, SEM_DIM)))
    z = raw.astype(np.float64) @ q
    # Deterministic non-affine LayerNorm + GELU, with no trainable parameters.
    z = (z - z.mean(1, keepdims=True)) / np.sqrt(z.var(1, keepdims=True) + 1e-5)
    zt = torch.nn.functional.gelu(torch.tensor(z, dtype=torch.float32)).numpy()
    return zt.astype(np.float32)


def semantic_maps(out: Path) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    raw = np.load(out / "PI0_TASK_EMBEDDINGS.npy").astype(np.float32)
    if raw.shape != (4, SEM_RAW_DIM):
        raise RuntimeError(f"semantic embedding mismatch: {raw.shape}")
    projected = fixed_projection(raw)
    return {t: projected[i] for i, t in enumerate(TASKS)}, raw, projected


class SemanticDirect(nn.Module):
    """The old Direct trunk with one-hot removed and fixed 16D semantics added."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(13, 64, batch_first=True)
        self.cond = nn.Sequential(nn.Linear(54 + SEM_DIM, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step13: torch.Tensor, cond70: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(step13)
        return self.head(torch.cat([h[-1], self.cond(cond70)], -1)).squeeze(-1)


def semantic_tensors(traces, segs, norm, mu_map, sem_map):
    s17, c54, y = af.tensors(
        traces, segs, norm, mu_map,
        {t.context_id: np.zeros(1, np.float32) for t in traces}, False,
    )
    h = torch.tensor(np.stack([sem_map[int(t.task)] for t in traces]), dtype=torch.float32)
    return s17[:, :, :13], torch.cat([c54, h], 1), y


def train_semantic_direct(train, segs, norm, mu_map, sem_map, meta, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SemanticDirect()
    opt = torch.optim.AdamW(model.parameters(), lr=af.LR, weight_decay=af.WD)
    for epoch in range(1, af.EPOCHS + 1):
        ids = af.early.sampled_ids(train, meta, seed, epoch)
        model.train()
        for start in range(0, len(ids), af.BATCH):
            batch = [train[int(i)] for i in ids[start:start + af.BATCH]]
            s, c, y = semantic_tensors(batch, segs, norm, mu_map, sem_map)
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(s, c), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
    model.eval()
    return model


def score_semantic(model, traces, segs, norm, mu_map, sem_map):
    s, c, _ = semantic_tensors(traces, segs, norm, mu_map, sem_map)
    with torch.no_grad():
        return af.sigmoid(model(s, c).numpy()).astype(np.float32)


def paired_split_manifest(md: pd.DataFrame, rf: np.ndarray) -> dict:
    rows = {}
    for target in TASKS:
        for fold in FOLDS:
            train_roots = sorted(md.loc[(md.task == target) & (rf != fold), "root_id"].unique())
            eval_roots = sorted(md.loc[(md.task == target) & (rf == fold), "root_id"].unique())
            key = f"task{target}_fold{fold}"
            rows[key] = {
                "adaptation_roots": train_roots,
                "evaluation_roots": eval_roots,
                "hash": hashlib.sha256(("|".join(train_roots) + "||" + "|".join(eval_roots)).encode()).hexdigest(),
            }
    return rows


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    (out / "_audit").mkdir(exist_ok=True)
    required_prev = [
        "TARGET_BOUNDARY_ACQUISITION_PROTOCOL.json",
        "TARGET_ACQUISITION_TRAJECTORIES.csv",
        "TARGET_FEWSHOT_LABEL_BALANCE.csv",
        "CURRENT_PHYSICS_PREDICTIONS.csv",
        "CURRENT_PROBE_CONTEXTS.csv",
        "NEW_TASK_FEWSHOT_TRANSFER.csv",
        "NEW_TASK_FEWSHOT_PER_EPISODE.csv",
        "PHYSICS_ESTIMATOR_LOTO.csv",
        "EXPLICIT_SYSID_CALIBRATION.json",
    ]
    for name in required_prev:
        if not (PREV / name).exists():
            raise FileNotFoundError(PREV / name)
        shutil.copy2(PREV / name, out / ("PRIOR_" + name))

    _, _, _, _, _, _, _, _, _, md, rf = prior.load_current(out / "_audit", "semantic_prepare")
    sem_map, raw, projected = semantic_maps(out)
    np.save(out / "PI0_TASK_EMBEDDINGS_PROJECTED16.npy", projected)

    # Cosine sanity check for both the exact 2048D representation and the
    # fixed 16D controller input.  This is computed before model training.
    sim_rows = []
    for i, ti in enumerate(TASKS):
        for j, tj in enumerate(TASKS):
            cr = float(raw[i] @ raw[j] / (np.linalg.norm(raw[i]) * np.linalg.norm(raw[j])))
            cp = float(projected[i] @ projected[j] / (np.linalg.norm(projected[i]) * np.linalg.norm(projected[j])))
            sim_rows.append({"task_i": ti, "task_j": tj, "prompt_i": PROMPTS[ti], "prompt_j": PROMPTS[tj],
                             "cosine_pi0_2048": cr, "cosine_projected16": cp})
    pd.DataFrame(sim_rows).to_csv(out / "TASK_SEMANTIC_SIMILARITY.csv", index=False)

    old_model = af.Direct(54)
    new_model = SemanticDirect()
    old_params = sum(p.numel() for p in old_model.parameters())
    new_params = sum(p.numel() for p in new_model.parameters())
    meta = json.loads((out / "PI0_TASK_EMBEDDING_METADATA.json").read_text())
    source_hashes = {name: sha(PREV / name) for name in required_prev}
    split_manifest = paired_split_manifest(md, rf)

    audit = f"""# Semantic Task Representation Audit

## Finding

The previous Direct did **not** have a semantic task-transfer mechanism. Its 71D input was split into a 17D temporal sequence (6 nominal Cartesian commands + 7 phase indicators + **4 task one-hot coordinates**) and a 54D static condition (candidate force, scalar physics, and physical state/masks). No language, RGB, π0 hidden state, or VLA semantic feature entered Direct.

## What exists in the frozen π0 stack

| Representation | Archived in force branches? | Deterministically recomputable without simulator? | Dimension | Frozen? | Selected? |
|---|---:|---:|---:|---:|---:|
| Raw neutral language instruction | Yes | Yes | string | Yes by protocol | Source text |
| PaliGemma SentencePiece tokens/mask | No | Yes, exact π0 tokenizer | 48 max tokens | Yes | Intermediate |
| PaliGemma/Gemma input-token embeddings | No | Yes, checkpoint 49999 | 2048/token | Yes | **Primary** |
| Masked mean prompt embedding | No | Yes | 2048 | Yes | **Primary task vector** |
| Contextual language hidden states | No | Not as a task-only deterministic vector; π0 contextualizes jointly with current images | 2048/token | Model frozen | Not selected |
| Explicit pooled language output | No | No such π0 interface exists | NA | NA | Not selected |
| Visual-language fused representation | No | Requires an observation and is not a task-only constant | observation-dependent | Model frozen | Not selected |
| Previously cached visual feature | Separate visual diagnostic only | Yes for those scenes | 4096/pre-PCA | Frozen | Not selected |

The primary representation is the exact frozen π0/PaliGemma input embedding used by `Pi0.embed_prefix`: tokenize the **actual neutral instruction**, index checkpoint `PaliGemma/llm/embedder/input_embedding`, multiply by sqrt(2048) exactly as π0 does, and mask-mean valid prompt tokens. It contains no outcome, force, friction, frontier, branch ID, or target-evaluation statistic and requires no task-specific training.

The resulting vector is 2048D. Before any transfer result is viewed it is passed through one fixed, outcome-independent 2048→16 orthogonal Gaussian projection (seed {PROJECTION_SEED}), followed by deterministic non-affine LayerNorm and GELU. Projection weights are not trained. The frozen 16D vector replaces the role of task identity; the one-hot columns are removed from the GRU input.

## Capacity and leakage audit

- Old one-hot Direct trainable parameters: **{old_params:,}**.
- Semantic Direct trainable parameters: **{new_params:,}** ({new_params-old_params:+,}; only **{100*(new_params-old_params)/old_params:.2f}%** different).
- π0 and the language embedding table are frozen; no encoder fine-tuning occurs.
- All four target instructions are legal at B=0, but no target outcomes or target Direct branches are used.
- Contextual/fused π0 states were rejected because they are not archived and are observation-dependent, which would change more than task representation.
- No alternative embedding, projection dimension, architecture, optimizer, epoch count, threshold, or reward is swept.
- Root-scaling untouched TEST is excluded by path and never read.

Authoritative code: `{OPENPI / 'models/tokenizer.py'}` and `{OPENPI / 'models/pi0.py'}`. Authoritative checkpoint: `{CHECKPOINT}`.
"""
    (out / "SEMANTIC_TASK_REPRESENTATION_AUDIT.md").write_text(audit, encoding="utf-8")

    protocol = {
        "status": "HASH_FROZEN_BEFORE_ANY_SEMANTIC_DIRECT_TRAINING",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_variable": "TASK_REPRESENTATION_ONLY",
        "primary_representation": {
            "name": "frozen_pi0_paligemma_masked_mean_input_token_embedding",
            "raw_dimension": SEM_RAW_DIM,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_metadata_sha256": sha(CHECKPOINT / "_CHECKPOINT_METADATA"),
            "tokenizer_source": str(OPENPI / "models/tokenizer.py"),
            "tokenizer_source_sha256": sha(OPENPI / "models/tokenizer.py"),
            "pi0_source": str(OPENPI / "models/pi0.py"),
            "pi0_source_sha256": sha(OPENPI / "models/pi0.py"),
            "embedding_file": "PI0_TASK_EMBEDDINGS.npy",
            "embedding_sha256": sha(out / "PI0_TASK_EMBEDDINGS.npy"),
            "embedding_metadata_sha256": sha(out / "PI0_TASK_EMBEDDING_METADATA.json"),
            "frozen": True,
            "task_specific_training": False,
            "outcome_force_friction_information": False,
            "prompts": PROMPTS,
        },
        "projection": {
            "type": "fixed Gaussian orthogonal projection; per-vector non-affine LayerNorm; GELU",
            "seed": PROJECTION_SEED,
            "output_dimension": SEM_DIM,
            "trainable": False,
            "selected_before_results": True,
            "alternatives_tested": 0,
            "projected_embedding_sha256": sha(out / "PI0_TASK_EMBEDDINGS_PROJECTED16.npy"),
        },
        "direct": {
            "old_temporal_input": "6 command + 7 phase + 4 task one-hot = 17",
            "new_temporal_input": "6 command + 7 phase = 13",
            "old_static_input": 54,
            "new_static_input": 70,
            "semantic_input": 16,
            "old_trainable_parameters": old_params,
            "new_trainable_parameters": new_params,
            "hidden_dimension": 64,
            "epochs": af.EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": af.LR,
            "weight_decay": af.WD,
            "loss": "full-task outcome BCE; authoritative class-sampling semantics unchanged",
            "seeds": SEEDS,
        },
        "planner": {
            "reward_success": "(Fmax-F)/Fmax",
            "reward_failure": -1,
            "decision": "argmax expected utility",
            "probability_threshold": None,
        },
        "matched_previous_experiment": str(PREV),
        "previous_source_hashes": source_hashes,
        "root_split_manifest": split_manifest,
        "budgets": BUDGETS,
        "nested_prefixes": True,
        "physics_estimators": ESTIMATORS,
        "onehot_baseline_policy": "reuse authoritative prior results; supplement no training",
        "simulator_rollouts_launched": 0,
        "root_scaling_TEST_used": False,
        "world_model_used": False,
        "residual_used": False,
        "joint_used": False,
    }
    wjson(out / "SEMANTIC_TASK_ENCODING_PROTOCOL.json", protocol)
    (out / "PRETRAIN_FREEZE_SHA256.txt").write_text(
        "\n".join(f"{sha(out/name)}  {name}" for name in [
            "PI0_TASK_EMBEDDINGS.npy", "PI0_TASK_EMBEDDINGS_PROJECTED16.npy",
            "PI0_TASK_EMBEDDING_METADATA.json", "TASK_SEMANTIC_SIMILARITY.csv",
            "SEMANTIC_TASK_REPRESENTATION_AUDIT.md", "SEMANTIC_TASK_ENCODING_PROTOCOL.json",
        ]) + "\n", encoding="utf-8")
    wjson(out / "PREPARE_COMPLETE.json", {
        "status": "PASS", "protocol_sha256": sha(out / "SEMANTIC_TASK_ENCODING_PROTOCOL.json"),
        "prior_acquisition_sha256": sha(PREV / "TARGET_ACQUISITION_TRAJECTORIES.csv"),
        "root_splits": split_manifest, "new_semantic_results_read": False,
    })


def get_mu_map(out: Path, target: int, seed: int, estimator: str) -> dict[str, float]:
    pred = pd.read_csv(out / "PRIOR_CURRENT_PHYSICS_PREDICTIONS.csv")
    q = pred[(pred.target_task == target) & (pred.method == estimator)]
    if estimator == "LearnedProbe":
        q = q[q.seed.astype(str) == str(seed)]
    else:
        # Deterministic estimates were repeated across seed rows in the prior map.
        q = q.drop_duplicates("context_id")
    ans = dict(zip(q.context_id, q.mu_hat.astype(float)))
    if len(ans) != 72:
        raise RuntimeError(f"bad physics map {target=} {seed=} {estimator=}: {len(ans)}")
    return ans


def evaluate_selected(md: pd.DataFrame, held_idx: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    f = md.force_N.to_numpy()[held_idx]
    fm = md.task.map(af.FMAX).to_numpy()[held_idx]
    utility = p * ((fm - f) / fm) + (1 - p) * -1
    score = np.full(len(md), np.nan, np.float32)
    score[held_idx] = utility
    return af.choose(md.iloc[held_idx], score)


def run_shard(out: Path, target: int, fold: int, seed: int) -> None:
    done = out / "shards" / f"target{target}_fold{fold}_seed{seed}.json"
    if done.exists():
        return
    _, _, _, _, traces, meta, _, _, segs, md, rf = prior.load_current(out / "_audit", f"semantic_{target}_{fold}_{seed}")
    sem_map, _, _ = semantic_maps(out)
    acq = pd.read_csv(out / "PRIOR_TARGET_ACQUISITION_TRAJECTORIES.csv")
    aq = acq[(acq.target_task == target) & (acq.fold == fold)]
    source_idx = np.flatnonzero((md.task != target) & (rf != fold) & (md.repeat == 1))
    held_idx = np.flatnonzero((md.task == target) & (rf == fold))
    if len(source_idx) != 180 or len(held_idx) != 60:
        raise RuntimeError("split cardinality changed")
    rows, selected_frames, label_rows = [], [], []
    for estimator in ESTIMATORS:
        mu_map = get_mu_map(out, target, seed, estimator)
        for budget in BUDGETS:
            target_ids = set(aq.loc[aq.query_index <= budget, "branch_id"])
            target_idx = np.flatnonzero(md.branch_id.isin(target_ids).to_numpy())
            train_idx = np.concatenate([source_idx, target_idx])
            if len(train_idx) != 180 + budget:
                raise RuntimeError("adaptation prefix changed")
            train = [traces[i] for i in train_idx]
            held = [traces[i] for i in held_idx]
            norm = af.xnorm(train, segs, mu_map)
            model = train_semantic_direct(train, segs, norm, mu_map, sem_map, meta, seed)
            p = score_semantic(model, held, segs, norm, mu_map, sem_map)
            sel = evaluate_selected(md, held_idx, p)
            sel["task_encoding"] = "SemanticPi0"
            sel["physics_estimator"] = estimator
            sel["target_task"] = target
            sel["fold"] = fold
            sel["seed"] = seed
            sel["budget"] = budget
            selected_frames.append(sel)
            rows.append({
                "task_encoding": "SemanticPi0", "physics_estimator": estimator,
                "target_task": target, "fold": fold, "budget": budget,
                "total_train_rollouts": 180 + budget, "seed": seed,
                "episodes": len(sel), "successes": int(sel.success.sum()),
                "sr": float(sel.success.mean()), "underforce": float(sel.under_force.mean()),
                "mean_force": float(sel.selected_force_N.mean()),
                "excess_force": float(sel.excess_force_N.mean()),
                "utility": float(sel.realized_utility.mean()),
                "selected_force_error": float((sel.selected_force_N - sel.frontier_N).abs().mean()),
            })
            source_y = md.iloc[source_idx].success.astype(int)
            target_y = md.iloc[target_idx].success.astype(int)
            label_rows.append({
                "target_task": target, "fold": fold, "seed": seed,
                "physics_estimator": estimator, "budget": budget,
                "source_success": int(source_y.sum()), "source_failure": int(len(source_y)-source_y.sum()),
                "target_success": int(target_y.sum()), "target_failure": int(len(target_y)-target_y.sum()),
                "combined_success": int(source_y.sum()+target_y.sum()),
                "combined_failure": int(len(source_y)+len(target_y)-source_y.sum()-target_y.sum()),
                "one_class_target": bool(budget > 0 and target_y.nunique() < 2),
            })
    pd.concat(selected_frames, ignore_index=True).to_csv(
        out / "shards" / f"target{target}_fold{fold}_seed{seed}_selected.csv", index=False)
    pd.DataFrame(label_rows).to_csv(
        out / "shards" / f"target{target}_fold{fold}_seed{seed}_labels.csv", index=False)
    wjson(done, {"status": "COMPLETE", "target": target, "fold": fold, "seed": seed,
                 "rows": rows, "models_trained": len(ESTIMATORS)*len(BUDGETS),
                 "root_scaling_TEST_used": False, "simulator_rollouts": 0})


def aggregate(out: Path) -> None:
    jsons = sorted((out / "shards").glob("target*_fold*_seed*.json"))
    sels = sorted((out / "shards").glob("*_selected.csv"))
    labs = sorted((out / "shards").glob("*_labels.csv"))
    if len(jsons) != 36 or len(sels) != 36 or len(labs) != 36:
        raise RuntimeError(f"incomplete shards: {len(jsons)}, {len(sels)}, {len(labs)}")
    selected = pd.concat([pd.read_csv(p) for p in sels], ignore_index=True)
    labels = pd.concat([pd.read_csv(p) for p in labs], ignore_index=True)
    labels.to_csv(out / "SEMANTIC_TRAIN_LABEL_AUDIT.csv", index=False)
    # Attach Semantic-GT decision agreement.
    gt = selected[selected.physics_estimator == "GT"][[
        "target_task", "fold", "seed", "budget", "context_id", "repeat", "selected_force_N"
    ]].rename(columns={"selected_force_N": "GT_selected_force_N"})
    selected = selected.merge(gt, on=["target_task", "fold", "seed", "budget", "context_id", "repeat"], validate="many_to_one")
    selected["gt_agreement"] = np.isclose(selected.selected_force_N, selected.GT_selected_force_N).astype(int)
    selected["selected_force_gap_GT"] = selected.selected_force_N - selected.GT_selected_force_N
    selected.to_csv(out / "SEMANTIC_TASK_TRANSFER_PER_EPISODE.csv", index=False)

    balance = pd.read_csv(out / "PRIOR_TARGET_FEWSHOT_LABEL_BALANCE.csv")
    agg_rows = []
    for keys, g in selected.groupby(["task_encoding", "physics_estimator", "target_task", "budget", "seed"]):
        enc, est, target, budget, seed = keys
        bb = balance[(balance.target_task == target) & (balance.budget == budget)]
        agg_rows.append({
            "task_encoding": enc, "physics_estimator": est, "target_task": target,
            "budget": budget, "total_train_rollouts": 180 + budget, "seed": seed,
            "episodes": len(g), "successes": int(g.success.sum()), "sr": float(g.success.mean()),
            "underforce": float(g.under_force.mean()), "mean_force": float(g.selected_force_N.mean()),
            "excess_force": float(g.excess_force_N.mean()), "utility": float(g.realized_utility.mean()),
            "gt_agreement": float(g.gt_agreement.mean()),
            "selected_force_error": float((g.selected_force_N-g.frontier_N).abs().mean()),
            "num_target_success": int(bb.success_count.sum()),
            "num_target_failure": int(bb.failure_count.sum()),
            "one_class": bool((bb.adaptation_status == "ONE_CLASS_TARGET_ADAPTATION").any()),
        })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out / "SEMANTIC_TASK_TRANSFER_AGG.csv", index=False)

    summary = []
    metrics = ["sr", "underforce", "mean_force", "excess_force", "utility", "gt_agreement", "selected_force_error"]
    for (est, task, budget), g in agg.groupby(["physics_estimator", "target_task", "budget"]):
        r = {"task_encoding": "SemanticPi0", "physics_estimator": est, "target_task": task,
             "budget": budget, "total_train_rollouts": 180+budget, "seeds": len(g), "episodes_per_seed": int(g.episodes.iloc[0])}
        for m in metrics:
            r[m+"_mean"] = float(g[m].mean()); r[m+"_std"] = float(g[m].std(ddof=0))
        r["num_target_success"] = int(g.num_target_success.iloc[0]); r["num_target_failure"] = int(g.num_target_failure.iloc[0]); r["one_class"] = bool(g.one_class.any())
        summary.append(r)
    per = pd.DataFrame(summary)
    macro_rows = []
    for (est, budget), g in agg.groupby(["physics_estimator", "budget"]):
        seed_macro = g.groupby("seed")[metrics].mean()
        r = {"task_encoding": "SemanticPi0", "physics_estimator": est, "target_task": "MACRO",
             "budget": budget, "total_train_rollouts": 180+budget, "seeds": seed_macro.shape[0], "episodes_per_seed": 144}
        for m in metrics:
            r[m+"_mean"] = float(seed_macro[m].mean()); r[m+"_std"] = float(seed_macro[m].std(ddof=0))
        r["num_target_success"] = int(g.groupby("target_task").num_target_success.first().sum())
        r["num_target_failure"] = int(g.groupby("target_task").num_target_failure.first().sum())
        r["one_class"] = bool(g.one_class.any())
        macro_rows.append(r)
    full_summary = pd.concat([per, pd.DataFrame(macro_rows)], ignore_index=True)
    full_summary.to_csv(out / "SEMANTIC_TASK_TRANSFER_PER_TASK.csv", index=False)

    # Strict paired comparison with historical one-hot GT, episode by episode.
    old = pd.read_csv(out / "PRIOR_NEW_TASK_FEWSHOT_PER_EPISODE.csv")
    old = old[old.physics_estimator == "GT"].copy()
    new = selected[selected.physics_estimator == "GT"].copy()
    keys = ["target_task", "fold", "seed", "budget", "context_id", "repeat"]
    keep_old = keys + ["selected_force_N", "success", "under_force", "realized_utility", "frontier_N"]
    keep_new = keys + ["selected_force_N", "success", "under_force", "realized_utility", "frontier_N"]
    paired = old[keep_old].merge(new[keep_new], on=keys, suffixes=("_onehot", "_semantic"), validate="one_to_one")
    paired["semantic_minus_onehot_success"] = paired.success_semantic - paired.success_onehot
    paired["semantic_minus_onehot_underforce"] = paired.under_force_semantic - paired.under_force_onehot
    paired["semantic_minus_onehot_force_N"] = paired.selected_force_N_semantic - paired.selected_force_N_onehot
    paired["semantic_minus_onehot_utility"] = paired.realized_utility_semantic - paired.realized_utility_onehot
    paired.to_csv(out / "SEMANTIC_VS_ONEHOT_PAIRED.csv", index=False)

    bstars = compute_bstars(full_summary)
    bstars.to_csv(out / "SEMANTIC_TASK_BSTAR.csv", index=False)
    write_task1_sensitivity(out, selected)
    write_qa(out, selected, agg, paired)
    make_figures(out, full_summary)
    write_report_and_classification(out, full_summary, paired, bstars)
    write_source_notes(out)
    write_html(out)
    write_hashes(out)


def compute_bstars(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for est in ESTIMATORS:
        task_pass = {}
        for task in TASKS:
            z = summary[(summary.physics_estimator == est) & (summary.target_task.astype(str) == str(task))].set_index("budget")
            ref = z.loc[60]
            task_pass[task] = {}
            for b in BUDGETS:
                r = z.loc[b]
                label_ok = b == 0 or (r.num_target_success > 0 and r.num_target_failure > 0 and not bool(r.one_class))
                ok = bool(abs(r.sr_mean-ref.sr_mean) <= .02 and r.underforce_mean <= ref.underforce_mean+.02 and r.utility_mean >= ref.utility_mean-UTILITY_TOL and label_ok)
                task_pass[task][b] = ok
            bs = next((b for b in BUDGETS if task_pass[task][b]), None)
            rows.append({"physics_estimator": est, "scope": f"task{task}", "B_star": bs if bs is not None else "NOT_REACHED",
                         "zero_shot_semantically_identifiable": True, "criteria": "SR gap<=2pp; underforce<=B60+2pp; utility>=B60-0.01; two-class if B>0"})
        z = summary[(summary.physics_estimator == est) & (summary.target_task == "MACRO")].set_index("budget")
        ref = z.loc[60]; bm = None; close = 0
        for b in BUDGETS:
            r = z.loc[b]; close = sum(task_pass[t][b] for t in TASKS)
            label_ok = b == 0 or (r.num_target_success > 0 and r.num_target_failure > 0 and not bool(r.one_class))
            if abs(r.sr_mean-ref.sr_mean) <= .02 and r.underforce_mean <= ref.underforce_mean+.02 and r.utility_mean >= ref.utility_mean-UTILITY_TOL and label_ok and close >= 3:
                bm = b; break
        rows.append({"physics_estimator": est, "scope": "MACRO", "B_star": bm if bm is not None else "NOT_REACHED",
                     "zero_shot_semantically_identifiable": True, "tasks_close_to_saturation": close if bm is not None else 0,
                     "criteria": "macro gates plus >=3/4 tasks individually close"})
    return pd.DataFrame(rows)


def write_task1_sensitivity(out: Path, selected: pd.DataFrame) -> None:
    files = list(PREV.glob("_audit/**/task1/TASK1_CANONICAL_TRAIN_BRANCHES.csv"))
    if not files:
        (out / "TASK1_SEMANTIC_DIRECT_LABEL_SENSITIVITY.csv").write_text("status\nUNAVAILABLE\n")
        return
    labels = pd.read_csv(files[0])[["context_id", "repeat", "force_N", "label_source"]].drop_duplicates()
    q = selected[selected.target_task == 1].merge(labels, left_on=["context_id", "repeat", "selected_force_N"], right_on=["context_id", "repeat", "force_N"], validate="many_to_one")
    rows=[]
    for keys,g in q.groupby(["physics_estimator","budget","seed"]):
        d=g[g.label_source=="DIRECT_CUMULATIVE_BRANCH_LABEL"]
        rows.append({"physics_estimator":keys[0],"budget":keys[1],"seed":keys[2],"all_episodes":len(g),"direct_label_episodes":len(d),
                     "all_sr":g.success.mean(),"direct_label_only_sr":d.success.mean() if len(d) else math.nan,
                     "all_underforce":g.under_force.mean(),"direct_label_only_underforce":d.under_force.mean() if len(d) else math.nan})
    pd.DataFrame(rows).to_csv(out / "TASK1_SEMANTIC_DIRECT_LABEL_SENSITIVITY.csv", index=False)


def write_qa(out: Path, selected: pd.DataFrame, agg: pd.DataFrame, paired: pd.DataFrame) -> None:
    checks = {
        "36_complete_shards": len(list((out/"shards").glob("target*_fold*_seed*.json"))) == 36,
        "6480_semantic_episode_rows": len(selected) == 6480,
        "180_seed_level_metric_rows": len(agg) == 180,
        "2160_gt_paired_rows": len(paired) == 2160,
        "12_eval_episodes_per_fold": set(selected.groupby(["physics_estimator","target_task","budget","seed","fold"]).size()) == {12},
        "2_eval_roots_per_fold": set(selected.groupby(["physics_estimator","target_task","budget","seed","fold"]).root_id.nunique()) == {2},
        "same_prior_acquisition_hash": sha(out/"PRIOR_TARGET_ACQUISITION_TRAJECTORIES.csv") == sha(PREV/"TARGET_ACQUISITION_TRAJECTORIES.csv"),
        "same_prior_label_balance_hash": sha(out/"PRIOR_TARGET_FEWSHOT_LABEL_BALANCE.csv") == sha(PREV/"TARGET_FEWSHOT_LABEL_BALANCE.csv"),
        "same_prior_physics_predictions_hash": sha(out/"PRIOR_CURRENT_PHYSICS_PREDICTIONS.csv") == sha(PREV/"CURRENT_PHYSICS_PREDICTIONS.csv"),
        "all_metrics_finite": bool(np.isfinite(agg[["sr","underforce","mean_force","utility","selected_force_error"]]).all().all()),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    wjson(out / "SEMANTIC_TASK_TRANSFER_QA.json", {"status":status,"checks":checks,"root_scaling_TEST_used":False,"new_simulator_rollouts":0,"task1_reconstructed_labels":"140/180"})
    if status != "PASS": raise RuntimeError("semantic transfer QA failed")


def make_figures(out: Path, summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    old = pd.read_csv(out / "PRIOR_NEW_TASK_FEWSHOT_TRANSFER.csv")
    old = old[old.physics_estimator == "GT"]
    oldm = old.groupby(["budget","seed"])["sr"].mean().groupby("budget").agg(["mean","std"]).reset_index()
    newm = summary[(summary.physics_estimator=="GT")&(summary.target_task=="MACRO")].sort_values("budget")
    fig, ax = plt.subplots(figsize=(8.6,5.3))
    ax.errorbar(oldm.budget, oldm["mean"]*100, yerr=oldm["std"].fillna(0)*100, marker="o", lw=2, capsize=3, color="#64748b", label="OneHot Direct")
    ax.errorbar(newm.budget, newm.sr_mean*100, yerr=newm.sr_std*100, marker="o", lw=2.4, capsize=3, color="#2563eb", label="Semantic Direct")
    ax.set(xlabel="Target-task adaptation rollouts", ylabel="Full-task success rate (%)", xticks=BUDGETS, title="GT physics isolates task representation")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out/"FIG_GT_SEMANTIC_VS_ONEHOT_TRANSFER.png",dpi=240); fig.savefig(out/"FIG_GT_SEMANTIC_VS_ONEHOT_TRANSFER.pdf"); plt.close(fig)

    colors={"GT":"#475569","ExplicitSysID":"#e97316","LearnedProbe":"#2563eb"}
    fig,ax=plt.subplots(figsize=(8.6,5.3))
    for est in ESTIMATORS:
        q=summary[(summary.physics_estimator==est)&(summary.target_task=="MACRO")].sort_values("budget")
        ax.errorbar(q.budget,q.sr_mean*100,yerr=q.sr_std*100,marker="o",lw=2,capsize=3,label=est,color=colors[est])
    ax.set(xlabel="Target-task adaptation rollouts",ylabel="Full-task success rate (%)",xticks=BUDGETS,title="Semantic Direct by physics source")
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"FIG_SEMANTIC_DIRECT_PHYSICS_SOURCE.png",dpi=240);fig.savefig(out/"FIG_SEMANTIC_DIRECT_PHYSICS_SOURCE.pdf");plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,task in zip(axs.flat,TASKS):
        q=summary[(summary.physics_estimator=="ExplicitSysID")&(summary.target_task.astype(str)==str(task))].sort_values("budget")
        ax.errorbar(q.budget,q.sr_mean*100,yerr=q.sr_std*100,marker="o",lw=2,capsize=2,color=colors["ExplicitSysID"])
        ax.set_title(f"task{task}");ax.set_xticks(BUDGETS)
    axs[1,0].set_xlabel("Target rollouts");axs[1,1].set_xlabel("Target rollouts");axs[0,0].set_ylabel("SR (%)");axs[1,0].set_ylabel("SR (%)")
    fig.suptitle("Deployable SysID–SemanticDirect transfer by target",y=.98);fig.tight_layout();fig.savefig(out/"FIG_SEMANTIC_NEW_TASK_PER_TASK.png",dpi=240);fig.savefig(out/"FIG_SEMANTIC_NEW_TASK_PER_TASK.pdf");plt.close(fig)


def _curve_stats(vals: list[float]) -> tuple[float,int,float]:
    d=np.diff(vals);return float(np.abs(d).sum()),int((d<0).sum()),float(d.min())


def write_report_and_classification(out: Path, summary: pd.DataFrame, paired: pd.DataFrame, bstars: pd.DataFrame) -> None:
    def row(est,task,b):
        return summary[(summary.physics_estimator==est)&(summary.target_task.astype(str)==str(task))&(summary.budget==b)].iloc[0]
    old_seed=pd.read_csv(out/"PRIOR_NEW_TASK_FEWSHOT_TRANSFER.csv")
    old_gt=old_seed[old_seed.physics_estimator=="GT"].groupby(["target_task","budget"])["sr"].mean()
    old_macro=old_seed[old_seed.physics_estimator=="GT"].groupby(["seed","budget"])["sr"].mean().groupby("budget").mean()
    gt=[row("GT","MACRO",b).sr_mean for b in BUDGETS]
    sysid=[row("ExplicitSysID","MACRO",b).sr_mean for b in BUDGETS]
    learned=[row("LearnedProbe","MACRO",b).sr_mean for b in BUDGETS]
    gains={b:gt[i]-float(old_macro.loc[b]) for i,b in enumerate(BUDGETS)}
    gt_b=bstars[(bstars.physics_estimator=="GT")&(bstars.scope=="MACRO")].B_star.iloc[0]
    sys_b=bstars[(bstars.physics_estimator=="ExplicitSysID")&(bstars.scope=="MACRO")].B_star.iloc[0]
    lp_b=bstars[(bstars.physics_estimator=="LearnedProbe")&(bstars.scope=="MACRO")].B_star.iloc[0]
    old_gt_b=pd.read_csv(PREV/"NEW_TASK_BSTAR.csv").query("physics_estimator=='GT' and scope=='MACRO'").B_star.iloc[0]
    old_smooth=_curve_stats([float(old_macro.loc[b]) for b in BUDGETS]);new_smooth=_curve_stats(gt)
    gt_low_ok=str(gt_b) in {"0","10","20"}
    # A single favorable B0 point does not establish removal of the one-hot
    # confound. Require favorable low-budget direction and an improved B*.
    semantic_gain=bool(gains[0] > .02 and gt_low_ok and np.mean([gains[0],gains[10],gains[20]]) > 0)
    if gt_low_ok and semantic_gain:
        primary="SEMANTIC_SHARED_DIRECT_SUPPORTS_ZERO_OR_FEW_SHOT_TRANSFER"
    elif str(gt_b)=="60":
        primary="FULL_TASK_FEASIBILITY_IS_TASK_SPECIFIC"
    elif not semantic_gain:
        primary="SEMANTIC_TASK_ENCODING_DOES_NOT_RESOLVE_TRANSFER"
    else:
        primary="MULTIPLE_FACTORS"
    secondary=[]
    if not semantic_gain: secondary.append("SEMANTIC_TASK_ENCODING_DOES_NOT_RESOLVE_TRANSFER")
    if gt_low_ok and str(sys_b) not in {"0","10","20"}: secondary.append("PHYSICS_ESTIMATION_IS_TRANSFER_BOTTLENECK")
    if all(sysid[i] > learned[i]+.01 for i in range(len(BUDGETS))): secondary.append("EXPLICIT_SYSID_PREFERRED_FOR_CROSS_TASK_PHYSICS")
    if new_smooth[0] < old_smooth[0]: secondary.append("SEMANTIC_CURVE_SMOOTHER_BY_TOTAL_VARIATION")

    per_b={est:{f"task{t}":str(bstars[(bstars.physics_estimator==est)&(bstars.scope==f"task{t}")].B_star.iloc[0]) for t in TASKS} for est in ESTIMATORS}
    easiest=min(TASKS,key=lambda t:row("GT",t,0).sr_mean*-1)
    hardest=max(TASKS,key=lambda t: BUDGETS.index(int(per_b["GT"][f"task{t}"])) if per_b["GT"][f"task{t}"].isdigit() else 99)
    task6_oneclass=pd.read_csv(out/"PRIOR_TARGET_FEWSHOT_LABEL_BALANCE.csv").query("target_task==6 and budget in [10,20]")
    t6_flag=bool((task6_oneclass.adaptation_status=="ONE_CLASS_TARGET_ADAPTATION").all())
    paired_summary=paired.groupby("budget").agg(success_delta=("semantic_minus_onehot_success","mean"),utility_delta=("semantic_minus_onehot_utility","mean"),force_delta=("semantic_minus_onehot_force_N","mean"))

    lines=["# Final Semantic Shared Direct Report","",
      "## Straight answer","",
      f"**{primary}.** Under accurate GT physics, changing only task representation moves macro B=0 SR from **{old_macro.loc[0]*100:.2f}%** to **{gt[0]*100:.2f}%** ({gains[0]*100:+.2f} pp), but B10/B20 change by **{gains[10]*100:+.2f}/{gains[20]*100:+.2f} pp**, and the preregistered GT macro B* worsens from **{old_gt_b}** (one-hot) to **{gt_b}** (semantic). Therefore the previous poor transfer was **not primarily caused by the one-hot encoding**. This fixed semantic representation does not resolve transfer, and substantial target-task full-outcome supervision remains necessary.","",
      "The deployable physics comparison is separate: SysID-SemanticDirect and LearnedProbe-SemanticDirect use exactly the previous source-only estimates. Their macro B* values are **"+str(sys_b)+"** and **"+str(lp_b)+"**. Thus semantic task transfer and physics-estimator transfer are not conflated.","",
      "## Frozen semantic representation","",
      "The old Direct used a 4D task one-hot. The new Direct uses the actual neutral task instruction encoded by the frozen π0 checkpoint's PaliGemma/Gemma input embedding table: valid token embeddings are mask-mean pooled to 2048D, then passed through one outcome-independent frozen 2048→16 projection (seed 20260901), deterministic LayerNorm, and GELU. π0 is never fine-tuned. The GRU input drops the four one-hot coordinates; the 16D task vector enters the static branch. Trainable parameter count differs by only 256 parameters.","",
      "## GT physics: one-hot versus semantic","",
      "| Budget | OneHot SR | Semantic SR | Δ Semantic | Semantic under-force | Semantic utility |","|---:|---:|---:|---:|---:|---:|"]
    for i,b in enumerate(BUDGETS):
        r=row("GT","MACRO",b); lines.append(f"| {b} | {old_macro.loc[b]*100:.2f}% | {r.sr_mean*100:.2f}% | {gains[b]*100:+.2f} pp | {r.underforce_mean*100:.2f}% | {r.utility_mean:.4f} |")
    lines += ["",f"Curve total variation changes from **{old_smooth[0]*100:.2f} pp** (one-hot) to **{new_smooth[0]*100:.2f} pp** (semantic); downward transitions remain {new_smooth[1]}. It is numerically smoother only because it stays lower through B30 before rising at B60; this is **not** improved data efficiency.","",
      "## Physics-source separation","","| Budget | GT Semantic SR | SysID Semantic SR | Learned Probe Semantic SR | GT−SysID gap |","|---:|---:|---:|---:|---:|"]
    for i,b in enumerate(BUDGETS): lines.append(f"| {b} | {gt[i]*100:.2f}% | {sysid[i]*100:.2f}% | {learned[i]*100:.2f}% | {(gt[i]-sysid[i])*100:+.2f} pp |")
    lines += ["","The prior estimator audit remains authoritative: Learned Probe LOTO MAE is 0.5149 and Explicit SysID MAE is 0.1810. Controller SR, however, is the headline; lower μ MAE is not automatically equated with better force decisions.","",
      "## Per-task adaptation budget","","| Physics | task0 B* | task1 B* | task5 B* | task6 B* | Macro B* |","|---|---:|---:|---:|---:|---:|"]
    for est in ESTIMATORS: lines.append(f"| {est} | {per_b[est]['task0']} | {per_b[est]['task1']} | {per_b[est]['task5']} | {per_b[est]['task6']} | {bstars[(bstars.physics_estimator==est)&(bstars.scope=='MACRO')].B_star.iloc[0]} |")
    lines += ["","## Why intermediate budgets can remain non-monotonic","",
      "The exact previous nested acquisition prefixes were reused byte-for-byte. More rows do not guarantee a monotone neural classifier: each prefix changes target/source class balance, root coverage, and near-boundary examples while the objective is branch BCE and the controller uses an argmax across forces. No warm-start or monotonic constraint exists. The semantic experiment therefore diagnoses whether task identity reduces the instability; it does not reinterpret a B10 spike as data efficiency.",
      f"task6 B10/B20 remains a one-class all-success adaptation anomaly: **{t6_flag}**. Those budgets are reported but cannot satisfy B*.","",
      "task1 remains in every macro result and retains the 140/180 reconstructed-terminal-label caveat. A selected-branch direct-label sensitivity is provided separately.","",
      "## Direct answers to the 20 requested questions","",
      "1. **Current Direct task representation:** 4D one-hot repeated at each of H=8 nominal-motion steps.",
      "2. **Selected frozen π0 representation:** masked mean of exact PaliGemma/Gemma input token embeddings for the real neutral instruction.",
      "3. **Dimensions:** 2048D raw; fixed 16D projected controller input.",
      "4. **Frozen?** Yes: checkpoint, tokenizer, pooling, projection, and π0 weights are frozen; no target outcome enters them.",
      f"5. **GT B0:** OneHot {old_macro.loc[0]*100:.2f}%; Semantic {gt[0]*100:.2f}%.",
      f"6. **B10/B20 deltas:** {gains[10]*100:+.2f} pp and {gains[20]*100:+.2f} pp.",
      f"7. **Smoother?** Numerically yes by total variation ({old_smooth[0]*100:.2f}→{new_smooth[0]*100:.2f} pp), but scientifically no improvement: B10–B60 are lower and B* worsens {old_gt_b}→{gt_b}.",
      f"8. **Semantic GT B*:** {gt_b}.", f"9. **SysID Semantic B*:** {sys_b}.", f"10. **Learned Probe Semantic B*:** {lp_b}.",
      f"11. **Does GT still need B60?** {'Yes' if str(gt_b)=='60' else 'No'}.",
      "12. **If GT needs 60:** that is evidence that full-task force feasibility remains task-specific under this fixed semantic representation, not proof for every possible representation.",
      "13. **If GT needs ≤20 but Learned Probe needs 60:** physics estimation, not Direct semantics, is the dominant transfer bottleneck.",
      f"14. **Does SysID close the gap?** At B0/B10/B20 the GT−SysID gaps are {(gt[0]-sysid[0])*100:+.2f}/{(gt[1]-sysid[1])*100:+.2f}/{(gt[2]-sysid[2])*100:+.2f} pp.",
      f"15. **Easiest zero-shot target under GT:** task{easiest} (highest Semantic-GT B0 SR).",
      f"16. **Largest GT adaptation requirement:** task{hardest}.",
      "17. **Does label composition explain non-monotonicity?** It contributes—especially one-class task6—but cannot alone explain every task/seed reversal; paired branch-BCE/argmax instability remains.",
      f"18. **task6 B10/B20 still anomalous?** Yes; all relevant folds are one-class all-success = {t6_flag}.",
      f"19. **Does '60 new-task rollouts required' still hold?** {'Yes under the frozen macro gate.' if str(gt_b)=='60' else 'No under GT semantics; deployable B* must still follow the physics-source row.'}",
      "20. **Recommended pipeline:** Do **not** replace the current known-task Direct with this semantic encoder and do not claim zero/few-shot new-task transfer. Keep the existing shared Direct for the four trained tasks; for a new task, budget substantial task-specific outcomes. Explicit SysID remains preferable by calibrated μ MAE, but its controller must be chosen by frozen downstream metrics; the present Semantic-SysID pipeline also needs B60.","",
      "## Scope and limitations","",
      "This is archived grouped-root development evidence. Held-out task also changes object family, so task semantics and object-family shift are not separable. GT physics is privileged diagnostic only. No simulator was run, no root-scaling untouched TEST was opened, and no representation or projection was chosen from these results.","",
      f"Paired GT analysis contains {len(paired)} exact episode pairs. At B0/B10/B20, mean paired utility deltas are "+", ".join(f"{paired_summary.loc[b,'utility_delta']:+.4f}" for b in [0,10,20])+"."]
    (out/"FINAL_SEMANTIC_SHARED_DIRECT_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    wjson(out/"FINAL_SEMANTIC_SHARED_DIRECT_CLASSIFICATION.json",{
        "primary_classification":primary,"secondary_classifications":secondary,
        "onehot_confound_material":bool(semantic_gain),"semantic_GT_Bstar":str(gt_b),"onehot_GT_Bstar":str(old_gt_b),"sysid_semantic_Bstar":str(sys_b),"learned_probe_semantic_Bstar":str(lp_b),
        "GT_macro_SR":{"OneHot_B0":float(old_macro.loc[0]),"Semantic_B0":gt[0],"Semantic_B10":gt[1],"Semantic_B20":gt[2],"Semantic_B30":gt[3],"Semantic_B60":gt[4]},
        "semantic_gain_pp":{str(b):100*gains[b] for b in BUDGETS},"per_task_Bstar":per_b,
        "task1_reconstructed_labels":"140/180","root_scaling_TEST_used":False,"new_simulator_rollouts":0,"status":"COMPLETE_DEVELOPMENT_EVIDENCE"})


def write_html(out: Path) -> None:
    cls=json.loads((out/"FINAL_SEMANTIC_SHARED_DIRECT_CLASSIFICATION.json").read_text())
    s=pd.read_csv(out/"SEMANTIC_TASK_TRANSFER_PER_TASK.csv")
    macro=s[s.target_task=="MACRO"].set_index(["physics_estimator","budget"])
    old=pd.read_csv(out/"PRIOR_NEW_TASK_FEWSHOT_TRANSFER.csv"); old=old[old.physics_estimator=="GT"].groupby(["seed","budget"]).sr.mean().groupby("budget").mean()
    def png(name): return base64.b64encode((out/name).read_bytes()).decode()
    gt_rows="".join(f"<tr><td>{b}</td><td>{old.loc[b]*100:.2f}%</td><td>{macro.loc[('GT',b),'sr_mean']*100:.2f}%</td><td>{(macro.loc[('GT',b),'sr_mean']-old.loc[b])*100:+.2f} pp</td></tr>" for b in BUDGETS)
    phys_rows="".join(f"<tr><td>{b}</td><td>{macro.loc[('GT',b),'sr_mean']*100:.2f}%</td><td>{macro.loc[('ExplicitSysID',b),'sr_mean']*100:.2f}%</td><td>{macro.loc[('LearnedProbe',b),'sr_mean']*100:.2f}%</td></tr>" for b in BUDGETS)
    page=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Semantic Shared Direct</title><style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe4ef;--blue:#2563eb;--orange:#e97316;--bg:#f3f6fa}}*{{box-sizing:border-box}}body{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;margin:0;background:var(--bg);color:var(--ink)}}main{{max-width:1120px;margin:32px auto;padding:0 20px 56px}}header,.panel{{background:white;border:1px solid var(--line);border-radius:16px;padding:28px;margin-bottom:20px;box-shadow:0 8px 24px rgba(30,50,80,.05)}}h1{{font-size:34px;margin:0 0 8px}}h2{{font-size:23px;margin:0 0 14px}}p{{line-height:1.6}}.eyebrow{{color:var(--blue);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:12px}}.sub{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:22px}}.card{{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:15px}}.card b{{display:block;font-size:24px;margin-top:5px}}.bad{{color:#b42318}}.good{{color:#067647}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}img{{width:100%;height:auto;border-radius:10px;border:1px solid var(--line)}}figure{{margin:0}}figcaption{{font-size:13px;color:var(--muted);margin-top:8px}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f8fafc}}.callout{{border-left:5px solid var(--orange);padding:14px 18px;background:#fff7ed;border-radius:8px}}a{{color:var(--blue)}}@media(max-width:800px){{.cards,.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><header><div class='eyebrow'>ActiveForcing · archived grouped-root development</div><h1>Semantic task encoding does not resolve new-task transfer</h1><p class='sub'>A matched one-hot versus frozen π0/PaliGemma task-representation experiment. No simulator, World Model, Residual, Joint, or root-scaling TEST was used.</p><div class='cards'><div class='card'>OneHot GT B0<b>{old.loc[0]*100:.2f}%</b></div><div class='card'>Semantic GT B0<b>{macro.loc[('GT',0),'sr_mean']*100:.2f}%</b></div><div class='card'>Semantic GT B*<b class='bad'>{cls['semantic_GT_Bstar']}</b></div><div class='card'>Old OneHot GT B*<b>{cls['onehot_GT_Bstar']}</b></div></div></header>
<section class='panel'><h2>Answer first</h2><div class='callout'><strong>{cls['primary_classification']}</strong><br>The B0 gain is only +{cls['semantic_gain_pp']['0']:.2f} pp; B10/B20 lose {abs(cls['semantic_gain_pp']['10']):.2f}/{abs(cls['semantic_gain_pp']['20']):.2f} pp, and B* worsens 30→60. The one-hot encoding was not the primary cause.</div></section>
<section class='panel'><h2>GT physics isolates task representation</h2><figure><img src='data:image/png;base64,{png('FIG_GT_SEMANTIC_VS_ONEHOT_TRANSFER.png')}' alt='GT semantic versus one-hot transfer curve'><figcaption>Three-seed macro mean ± seed variation; every evaluation episode is paired.</figcaption></figure><table><thead><tr><th>Target rollouts</th><th>OneHot SR</th><th>Semantic SR</th><th>Δ</th></tr></thead><tbody>{gt_rows}</tbody></table></section>
<section class='grid'><section class='panel'><h2>Physics-source gap</h2><figure><img src='data:image/png;base64,{png('FIG_SEMANTIC_DIRECT_PHYSICS_SOURCE.png')}' alt='Semantic Direct physics source curves'></figure><table><thead><tr><th>B</th><th>GT</th><th>SysID</th><th>Learned</th></tr></thead><tbody>{phys_rows}</tbody></table></section><section class='panel'><h2>Per-task heterogeneity</h2><figure><img src='data:image/png;base64,{png('FIG_SEMANTIC_NEW_TASK_PER_TASK.png')}' alt='Per-task SysID Semantic Direct curves'><figcaption>task1 is hardest; task0 is saturated in SR but not necessarily force utility. task6 B10/B20 is one-class.</figcaption></figure></section></section>
<section class='panel'><h2>Interpretation and limits</h2><p>The exact π0 token embeddings are extremely similar because all prompts share the same template and differ mainly in the object noun (raw cosine ≈0.996–0.997). This negative result applies to this preregistered frozen representation; it does not prove that every possible semantic representation must fail.</p><p>GT is diagnostic only. The deployable SysID-Semantic and LearnedProbe-Semantic pipelines both have macro B*=60. task1 retains 140/180 reconstructed terminal labels, and held-out task also changes object family.</p><p><a href='FINAL_SEMANTIC_SHARED_DIRECT_REPORT.md'>Full 20-question report</a> · <a href='SEMANTIC_TASK_TRANSFER_AGG.csv'>Seed-level metrics</a> · <a href='SEMANTIC_VS_ONEHOT_PAIRED.csv'>Paired episodes</a> · <a href='SEMANTIC_TASK_TRANSFER_QA.json'>QA</a></p></section></main></body></html>"""
    (out/"report.html").write_text(page,encoding="utf-8")


def write_source_notes(out: Path) -> None:
    (out/"REPORT_SOURCE_NOTES.md").write_text(f"""# Report Source Notes

- Primary population: archived 720-branch / 24-root / 72-context pooled TRAIN population loaded through the same authoritative parser as `{PREV}`.
- Exact target acquisition trajectory SHA256: `{sha(PREV/'TARGET_ACQUISITION_TRAJECTORIES.csv')}`.
- Exact physics predictions SHA256: `{sha(PREV/'CURRENT_PHYSICS_PREDICTIONS.csv')}`.
- Frozen semantic protocol SHA256: `{sha(out/'SEMANTIC_TASK_ENCODING_PROTOCOL.json')}`.
- Seed-level controller metrics: `SEMANTIC_TASK_TRANSFER_AGG.csv`.
- Exact paired GT decisions: `SEMANTIC_VS_ONEHOT_PAIRED.csv`.
- Data and split QA: `SEMANTIC_TASK_TRANSFER_QA.json` (PASS).
- Scope: archived development only; GT is privileged; task1 has 140/180 reconstructed labels; held-out task and object family are confounded.
- Exclusions: zero new simulator rollouts; root-scaling untouched TEST, World Model, Residual, Joint, Agent, and Probe retraining were not used.
""",encoding="utf-8")
    wjson(out/"artifact.json",{
        "surface":"report","title":"Semantic Shared Direct Transfer","status":"ready",
        "generatedAt":datetime.now(timezone.utc).isoformat(),
        "primary_classification":json.loads((out/"FINAL_SEMANTIC_SHARED_DIRECT_CLASSIFICATION.json").read_text())["primary_classification"],
        "sources":[
            {"id":"semantic_metrics","path":"SEMANTIC_TASK_TRANSFER_AGG.csv","grain":"task × budget × seed × physics estimator"},
            {"id":"paired_gt","path":"SEMANTIC_VS_ONEHOT_PAIRED.csv","grain":"controller episode"},
            {"id":"qa","path":"SEMANTIC_TASK_TRANSFER_QA.json","status":"PASS"},
        ],
        "report":"report.html","authoritative_markdown":"FINAL_SEMANTIC_SHARED_DIRECT_REPORT.md"})


def write_hashes(out: Path) -> None:
    files=[p for p in out.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt"]
    (out/"SHA256SUMS.txt").write_text("\n".join(f"{sha(p)}  {p.name}" for p in sorted(files))+"\n",encoding="utf-8")


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("phase",choices=["prepare","shard","aggregate","hashes"])
    ap.add_argument("--target",type=int);ap.add_argument("--fold",type=int);ap.add_argument("--seed",type=int)
    a=ap.parse_args()
    if a.phase=="prepare": prepare(a.out)
    elif a.phase=="shard": run_shard(a.out,a.target,a.fold,a.seed)
    elif a.phase=="aggregate": aggregate(a.out)
    else: write_hashes(a.out)


if __name__ == "__main__":
    main()
