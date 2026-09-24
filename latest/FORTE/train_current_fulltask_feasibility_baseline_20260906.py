#!/usr/bin/env python3
"""Current-runtime, pre-action, posterior-marginalized feasibility baseline.

The script has an immutable freeze phase and a train phase.  TEST row files are
not opened until every seed has been selected by validation posterior NLL and a
selection lock has been written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch import nn


ROOT = Path("/home/exouser/FORTE")
STAGE = ROOT / "analysis/results/current_matched_stage1_unknown_quarantine_v1_20260906"
COLLECTION = ROOT / "analysis/results/current_matched_resumption_v1_20260906"
QA = STAGE / "qa/20260906T190204_688775Z/FULL_STAGE_QA.json"
STAGE_MANIFEST = ROOT / "analysis/results/current_matched_stage1_648_v1_20260906/STAGE_MANIFEST.json"
OUT = ROOT / "analysis/results/current_fulltask_feasibility_baseline_v1_20260906"
PROTOCOL = OUT / "TRAINING_PROTOCOL.json"
SELECTION_LOCK = OUT / "CHECKPOINT_SELECTION_LOCK.json"
SEEDS = (0, 1, 2)
EPOCHS = 80
BATCH = 64
LR = 8e-4
WEIGHT_DECAY = 1e-4
FORCE_GRID = np.round(np.arange(3.0, 5.0001, 0.05), 8)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class FeasibilityOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.command_gru = nn.GRU(17, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step, cond):
        _, hidden = self.command_gru(step)
        joined = torch.cat([hidden[-1], self.condition(cond)], dim=-1)
        return self.head(joined).squeeze(-1)


def parameter_count():
    return sum(p.numel() for p in FeasibilityOnly().parameters())


def context_maps():
    stage = load(STAGE_MANIFEST)
    contexts = {c["id"]: c for c in stage["contexts"]}
    return stage, contexts


def row_paths(split):
    stage, _ = context_maps()
    allowed = {cid for cid, value in stage["split_by_context"].items() if value == split}
    paths = []
    for path in sorted((COLLECTION / "rows").glob("*.json")):
        cid = path.stem.rsplit("__F", 1)[0]
        if cid in allowed:
            paths.append(path)
    return paths


def load_rows(split):
    _, contexts = context_maps()
    rows = []
    for path in row_paths(split):
        row = load(path)
        base = np.load(Path(row["job"]) / "PREACTION_SEQUENCE.npy", allow_pickle=False).astype(np.float32)
        x = base.copy()
        x[:, 17] = float(row["force"]) / 8.0
        x[:, 18] = float(contexts[row["context_id"]]["mu"])
        rows.append(dict(row=row, x=x, y=float(row["full_task_success_y"])))
    return rows


def fit_normalization(rows):
    values = np.stack([r["x"] for r in rows])
    mean = values.mean(axis=(0, 1)).astype(np.float32)
    std = values.std(axis=(0, 1)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def tensors(rows, mean, std, device):
    x = (np.stack([r["x"] for r in rows]) - mean[None, None]) / std[None, None]
    step = torch.as_tensor(x[:, :, :17], dtype=torch.float32, device=device)
    cond = torch.as_tensor(x[:, 0, 17:], dtype=torch.float32, device=device)
    y = torch.as_tensor([r["y"] for r in rows], dtype=torch.float32, device=device)
    return step, cond, y


def metric(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    order = np.argsort(-p)
    ys = y[order]
    positives = int(y.sum()); negatives = len(y) - positives
    auprc = float((np.cumsum(ys) / np.arange(1, len(y) + 1))[ys == 1].mean()) if positives else None
    order_up = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and p[order_up[j]] == p[order_up[i]]: j += 1
        ranks[order_up[i:j]] = (i + 1 + j) / 2.0
        i = j
    auroc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)) if positives and negatives else None
    ece = 0.0
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        mask = (p >= lo) & ((p < hi) if i < 9 else (p <= hi))
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return dict(n=len(y), positive_count=positives,
                nll=float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))),
                brier=float(np.mean((p - y) ** 2)), auroc=auroc, auprc=auprc, ece_10bin=ece)


def normalized_batch(base, forces, mus, mean, std):
    n = len(forces)
    x = np.broadcast_to(base, (n,) + base.shape).copy()
    x[:, :, 17] = np.asarray(forces, np.float32)[:, None] / 8.0
    x[:, :, 18] = np.asarray(mus, np.float32)[:, None]
    return (x - mean[None, None]) / std[None, None]


def model_probabilities(model, x, device, batch=4096):
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            value = torch.as_tensor(x[start:start + batch], dtype=torch.float32, device=device)
            output.append(torch.sigmoid(model(value[:, :, :17], value[:, 0, 17:])).cpu().numpy())
    return np.concatenate(output)


def posterior_predictions(models, rows, mean, std, device):
    by_context = defaultdict(list)
    for record in rows:
        by_context[record["row"]["context_id"]].append(record)
    predictions = {}
    for cid, group in by_context.items():
        reference = Path(group[0]["row"]["reference"])
        posterior = load(reference / "PREACTION_POSTERIOR.json")
        nodes = np.asarray(posterior["integration_nodes"], np.float32)
        weights = np.asarray(posterior["integration_weights"], float)
        base = np.load(reference / "PREACTION_SEQUENCE.npy", allow_pickle=False).astype(np.float32)
        forces = np.asarray([r["row"]["force"] for r in group], np.float32)
        all_forces = np.repeat(forces, len(nodes)); all_mus = np.tile(nodes, len(forces))
        x = normalized_batch(base, all_forces, all_mus, mean, std)
        values = np.mean([model_probabilities(model, x, device) for model in models], axis=0)
        values = values.reshape(len(forces), len(nodes)) @ weights
        for record, value in zip(group, values):
            predictions[record["row"]["context_id"], float(record["row"]["force"])] = float(value)
    return np.asarray([predictions[r["row"]["context_id"], float(r["row"]["force"])] for r in rows])


def validation_nll(model, rows, mean, std, device):
    p = posterior_predictions([model], rows, mean, std, device)
    return metric([r["y"] for r in rows], p)["nll"]


def planner(models, rows, mean, std, device):
    by_context = {}
    for record in rows:
        by_context.setdefault(record["row"]["context_id"], record)
    outputs = []
    for cid, record in sorted(by_context.items()):
        row = record["row"]; reference = Path(row["reference"])
        posterior = load(reference / "PREACTION_POSTERIOR.json")
        nodes = np.asarray(posterior["integration_nodes"], np.float32)
        weights = np.asarray(posterior["integration_weights"], float)
        base = np.load(reference / "PREACTION_SEQUENCE.npy", allow_pickle=False).astype(np.float32)
        forces = np.repeat(FORCE_GRID, len(nodes)); mus = np.tile(nodes, len(FORCE_GRID))
        x = normalized_batch(base, forces, mus, mean, std)
        probability = np.mean([model_probabilities(model, x, device) for model in models], axis=0)
        curve = (probability.reshape(len(FORCE_GRID), len(nodes)) @ weights).astype(float)
        utility = curve * (5.0 - FORCE_GRID) / 5.0 + (1.0 - curve) * -1.0
        selected = int(np.argmax(utility))
        point_x = normalized_batch(base, FORCE_GRID, np.full(len(FORCE_GRID), posterior["posterior_moments"]["mean"]), mean, std)
        point_curve = np.mean([model_probabilities(model, point_x, device) for model in models], axis=0)
        point_utility = point_curve * (5.0 - FORCE_GRID) / 5.0 + (1.0 - point_curve) * -1.0
        outputs.append(dict(context_id=cid, task=row["task"], root=row["root"], band=row["band"], split=row["split"],
                            true_mu=context_maps()[1][cid]["mu"], posterior_mean=posterior["posterior_moments"]["mean"],
                            posterior_std=posterior["posterior_moments"]["std"], selected_force=float(FORCE_GRID[selected]),
                            point_mu_selected_force=float(FORCE_GRID[int(np.argmax(point_utility))]),
                            p3=float(curve[0]), p5=float(curve[-1]), force_sensitivity=float(curve[-1] - curve[0]),
                            monotonicity_violation_rate=float(np.mean(np.diff(curve) < -1e-8)),
                            curve=curve.tolist(), point_mu_curve=point_curve.astype(float).tolist()))
    return outputs


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    def ranks(a):
        order = np.argsort(a, kind="mergesort"); result = np.empty(len(a), float)
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a[order[j]] == a[order[i]]: j += 1
            result[order[i:j]] = (i + j - 1) / 2
            i = j
        return result
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else None


def planner_summary(outputs):
    selected = np.asarray([r["selected_force"] for r in outputs])
    groups = defaultdict(list)
    for row in outputs: groups[(row["task"], row["root"])].append(row)
    changed = [len({r["selected_force"] for r in group}) > 1 for group in groups.values()]
    return dict(contexts=len(outputs), mean_selected_force=float(selected.mean()), std_selected_force=float(selected.std()),
                min_selected_force=float(selected.min()), max_selected_force=float(selected.max()),
                unique_selected_force_count=len(set(selected)), lower_bound_selection_rate=float(np.mean(selected == 3.0)),
                interior_selection_rate=float(np.mean((selected > 3.0) & (selected < 5.0))),
                upper_bound_selection_rate=float(np.mean(selected == 5.0)),
                decision_change_rate=float(np.mean(changed)),
                low_mu_mean_force=float(np.mean([r["selected_force"] for r in outputs if r["band"] == "LOW"])),
                mid_mu_mean_force=float(np.mean([r["selected_force"] for r in outputs if r["band"] == "MID"])),
                high_mu_mean_force=float(np.mean([r["selected_force"] for r in outputs if r["band"] == "HIGH"])),
                friction_force_spearman=spearman([r["true_mu"] for r in outputs], selected),
                force_sensitivity_mean=float(np.mean([r["force_sensitivity"] for r in outputs])),
                force_sensitivity_median=float(np.median([r["force_sensitivity"] for r in outputs])),
                monotonicity_violation_rate=float(np.mean([r["monotonicity_violation_rate"] for r in outputs])),
                point_mu_mean_selected_force=float(np.mean([r["point_mu_selected_force"] for r in outputs])))


def freeze():
    if OUT.exists():
        raise RuntimeError("output already exists; refusing overwrite")
    qa = load(QA); stage = load(STAGE_MANIFEST)
    if not qa["structural_pass"] or not qa["fulltask_probability_training_admitted"]:
        raise RuntimeError("full-stage QA did not admit training")
    train, val = row_paths("TRAIN"), row_paths("VAL")
    OUT.mkdir(parents=True)
    shutil.copy2(__file__, OUT / Path(__file__).name)
    protocol = dict(version="CURRENT_FULLTASK_PREACTION_POSTERIOR_BASELINE_V1", created_utc=now(),
                    dataset_grain="current post-probe pre-candidate decision state x continuous force -> full-task success",
                    label="full_task_success_y", unknown_rows_excluded=qa["censored_unknown_rows"], old720_rows_used=0,
                    rows_known_before_selection={"TRAIN": len(train), "VAL": len(val), "TEST": "LABELS_NOT_OPENED"},
                    split_unit="root", roots_by_split=qa["roots_by_split"], root_leakage=False,
                    architecture="GRU(17,64)+MLP(54,64)+head(128,64,1)", parameter_count=parameter_count(),
                    sequence_length=8, sequence_dim=17, condition_dim=54, candidate_force_normalization="F / 8",
                    training_mu="simulator friction used only as conditional supervision; never a deployment observable",
                    runtime_mu="formal continuous positive-mixture posterior nodes/weights; sigma used",
                    normalization="TRAIN-only, per-channel across sample and time axes", sampling="uniform shuffled minibatches",
                    seeds=list(SEEDS), epochs=EPOCHS, batch_size=BATCH, optimizer="AdamW", lr=LR,
                    weight_decay=WEIGHT_DECAY, gradient_clip=1.0,
                    checkpoint_selection="lowest validation posterior-marginalized raw NLL per seed",
                    test_access="only after CHECKPOINT_SELECTION_LOCK.json", calibration="NONE_BASELINE_RAW",
                    force_support=[3.0, 5.0], planner_grid_step=0.05,
                    utility="p*(Fmax-F)/Fmax + (1-p)*(-1); lower-force tie break",
                    qa_path=str(QA), qa_sha256=sha(QA), stage_manifest=str(STAGE_MANIFEST), stage_manifest_sha256=sha(STAGE_MANIFEST),
                    runtime_sha256=qa["runtime_sha256"], trainer_sha256=sha(OUT / Path(__file__).name),
                    train_val_row_hashes={str(p): sha(p) for p in train + val})
    write(PROTOCOL, protocol)
    write(OUT / "PROTOCOL_LOCK.json", {"protocol_sha256": sha(PROTOCOL)})
    print(json.dumps({"out": str(OUT), "protocol_sha256": sha(PROTOCOL), "train": len(train), "val": len(val)}, indent=2))


def check_protocol():
    protocol = load(PROTOCOL)
    if sha(PROTOCOL) != load(OUT / "PROTOCOL_LOCK.json")["protocol_sha256"]:
        raise RuntimeError("protocol changed")
    if sha(OUT / Path(__file__).name) != protocol["trainer_sha256"]:
        raise RuntimeError("frozen trainer changed")
    if sha(QA) != protocol["qa_sha256"] or sha(STAGE_MANIFEST) != protocol["stage_manifest_sha256"]:
        raise RuntimeError("dataset QA/stage changed")
    for path, digest in protocol["train_val_row_hashes"].items():
        if sha(path) != digest: raise RuntimeError("TRAIN/VAL row changed: " + path)
    return protocol


def train():
    protocol = check_protocol()
    if SELECTION_LOCK.exists(): raise RuntimeError("training already selected; refusing overwrite")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    train_rows, val_rows = load_rows("TRAIN"), load_rows("VAL")
    mean, std = fit_normalization(train_rows)
    tr_step, tr_cond, tr_y = tensors(train_rows, mean, std, device)
    checkpoints = []
    histories = []
    models = []
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = FeasibilityOnly().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        rng = np.random.default_rng(seed)
        best = None; history = []
        for epoch in range(1, EPOCHS + 1):
            model.train(); permutation = rng.permutation(len(train_rows)); losses = []
            for start in range(0, len(permutation), BATCH):
                index = torch.as_tensor(permutation[start:start + BATCH], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(tr_step[index], tr_cond[index]), tr_y[index])
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_nll = validation_nll(model, val_rows, mean, std, device)
            history.append(dict(epoch=epoch, train_minibatch_bce=float(np.mean(losses)), val_posterior_nll=val_nll))
            if best is None or val_nll < best["val_nll"]:
                best = dict(epoch=epoch, val_nll=val_nll,
                            state={k: value.detach().cpu().clone() for k, value in model.state_dict().items()})
        model.load_state_dict(best["state"]); model.eval()
        checkpoint = OUT / f"CURRENT_FULLTASK_FEAS_seed{seed}.pt"
        torch.save(dict(state_dict=best["state"], seed=seed, selected_epoch=best["epoch"],
                        validation_posterior_nll=best["val_nll"], normalization_mean=mean,
                        normalization_std=std, architecture=protocol["architecture"], target="full_task_success_y",
                        protocol_sha256=sha(PROTOCOL)), checkpoint)
        checkpoints.append(dict(seed=seed, path=str(checkpoint), sha256=sha(checkpoint),
                                selected_epoch=best["epoch"], validation_posterior_nll=best["val_nll"]))
        histories.extend(dict(seed=seed, **item) for item in history)
        models.append(model)
        print(f"seed={seed} epoch={best['epoch']} val_nll={best['val_nll']:.6f}", flush=True)
    write(SELECTION_LOCK, dict(created_utc=now(), rule=protocol["checkpoint_selection"], checkpoints=checkpoints,
                               test_labels_accessed_before_lock=False, protocol_sha256=sha(PROTOCOL)))

    # TEST labels and row hashes are first opened after the immutable lock above.
    test_rows = load_rows("TEST")
    split_rows = {"TRAIN": train_rows, "VAL": val_rows, "TEST": test_rows}
    metrics = {}
    prediction_rows = []
    for split, rows in split_rows.items():
        probabilities = posterior_predictions(models, rows, mean, std, device)
        metrics[split] = metric([r["y"] for r in rows], probabilities)
        for record, probability in zip(rows, probabilities):
            row = record["row"]
            prediction_rows.append(dict(split=split, context_id=row["context_id"], task=row["task"], root=row["root"],
                                        band=row["band"], force=row["force"], y=int(record["y"]), p_success=float(probability)))
    planner_outputs = planner(models, val_rows + test_rows, mean, std, device)
    summary = planner_summary(planner_outputs)

    import csv
    for path, rows, exclude in ((OUT / "TRAINING_HISTORY.csv", histories, set()),
                                (OUT / "PREDICTIONS.csv", prediction_rows, set()),
                                (OUT / "HELDOUT_PLANNER.csv", planner_outputs, {"curve", "point_mu_curve"})):
        cooked = [{k: v for k, v in row.items() if k not in exclude} for row in rows]
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cooked[0])); writer.writeheader(); writer.writerows(cooked)
    write(OUT / "HELDOUT_CURVES.json", {r["context_id"]: {"force_grid": FORCE_GRID.tolist(),
          "posterior_curve": r["curve"], "point_mu_curve": r["point_mu_curve"]} for r in planner_outputs})
    write(OUT / "TRAINING_RESULTS.json", dict(created_utc=now(), protocol_sha256=sha(PROTOCOL),
          selection_lock_sha256=sha(SELECTION_LOCK), device=torch.cuda.get_device_name(0), checkpoints=checkpoints,
          metrics=metrics, heldout_planner=summary, calibration="NONE_BASELINE_RAW",
          test_rows=len(test_rows), test_row_hashes={str(p): sha(p) for p in row_paths("TEST")},
          normalization={"mean": mean.tolist(), "std": std.tolist(), "fit_split": "TRAIN"},
          artifacts={"history": str(OUT / "TRAINING_HISTORY.csv"), "predictions": str(OUT / "PREDICTIONS.csv"),
                     "planner": str(OUT / "HELDOUT_PLANNER.csv"), "curves": str(OUT / "HELDOUT_CURVES.json")}))
    print(json.dumps(dict(metrics=metrics, heldout_planner=summary, checkpoints=checkpoints), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.freeze: freeze()
    elif args.train: train()
