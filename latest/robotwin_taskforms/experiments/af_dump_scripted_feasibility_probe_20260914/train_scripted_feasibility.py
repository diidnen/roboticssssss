"""Scripted-dump feasibility probe.

Same dump phase-free network (GRU-10 + 54-D condition) as native AF.
Features are NOT π0 chunks: force and μ occupy the official slots; remaining
channels are zeros (no VLA motion). Labels are scripted play_once success.

This is a label-landscape probe, not online AF efficacy.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments")
EXPS = [
    BASE / "af_dump_scripted_lowforce_1_3_20260914",
    BASE / "af_dump_scripted_lowforce_3_8_20260914",
    BASE / "af_dump_scripted_lowforce_5_10_20260914",
    BASE / "af_dump_scripted_highforce_20260914",
]
FORCE_NORM = 20.0
SEEDS = (0, 1, 2)
EPOCHS = 80
BATCH = 64


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.command_gru = nn.GRU(10, 64, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(54, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, step, cond):
        _, h = self.command_gru(step)
        return self.head(torch.cat([h[-1], self.condition(cond)], -1)).squeeze(-1)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def load_labels() -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for exp in EXPS:
        recs = exp / "main_records"
        if not recs.exists():
            continue
        for path in sorted(recs.glob("*.json")):
            record = json.loads(path.read_text())
            ctx = record["context"]
            for outcome in record["outcomes"]:
                if outcome.get("unstable_layout"):
                    continue
                row = {
                    "context_id": ctx["id"],
                    "seed": int(ctx["seed"]),
                    "mu": float(ctx["friction"]),
                    "force": float(outcome["force_N"]),
                    "y": int(outcome["success"]),
                    "band": exp.name,
                    "contact_ratio": outcome.get("contact_ratio"),
                }
                grouped[(row["context_id"], row["force"])].append(row)
    labels = []
    disagreements = []
    for key, rows in sorted(grouped.items()):
        ys = {row["y"] for row in rows}
        if len(ys) > 1:
            disagreements.append(
                {
                    "context_id": key[0],
                    "force": key[1],
                    "votes": [{"band": r["band"], "y": r["y"]} for r in rows],
                }
            )
            continue
        keep = rows[0]
        labels.append(keep)
    return labels, disagreements


def make_x(mu: float, force: float) -> np.ndarray:
    x = np.zeros((8, 64), dtype=np.float32)
    x[:, 10] = float(force) / FORCE_NORM
    x[:, 11] = float(mu)
    return x


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    acc = float(np.mean(pred == y))
    pos = int(y.sum())
    neg = int(len(y) - pos)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pclip = np.clip(p, 1e-7, 1 - 1e-7)
    nll = float(np.mean(-(y * np.log(pclip) + (1 - y) * np.log(1 - pclip))))
    brier = float(np.mean((p - y) ** 2))
    majority = float(max(pos, neg) / len(y))
    fail_recall = float(tn / neg) if neg else None
    return {
        "n": int(len(y)),
        "positives": pos,
        "negatives": neg,
        "accuracy": acc,
        "majority_baseline": majority,
        "accuracy_minus_majority": acc - majority,
        "brier": brier,
        "nll": nll,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fail_recall": fail_recall,
        "success_recall": float(tp / pos) if pos else None,
    }


def predict(model, xs, mean, std, device):
    x = (xs - mean[None, None]) / std[None, None]
    step = torch.as_tensor(x[:, :, :10], device=device)
    cond = torch.as_tensor(x[:, 0, 10:], device=device)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(step, cond)).cpu().numpy()


def train_one(train_rows, val_rows, device, seed: int):
    xs = np.stack([make_x(r["mu"], r["force"]) for r in train_rows])
    ys = np.asarray([r["y"] for r in train_rows], np.float32)
    mean = xs.mean(axis=(0, 1)).astype(np.float32)
    std = xs.std(axis=(0, 1)).astype(np.float32)
    std[std < 1e-6] = 1.0
    x = (xs - mean[None, None]) / std[None, None]
    tr_step = torch.as_tensor(x[:, :, :10], device=device)
    tr_cond = torch.as_tensor(x[:, 0, 10:], device=device)
    tr_y = torch.as_tensor(ys, device=device)
    val_x = np.stack([make_x(r["mu"], r["force"]) for r in val_rows]) if val_rows else xs[:1]
    val_y = np.asarray([r["y"] for r in val_rows], np.float32) if val_rows else ys[:1]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Network().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = rng.permutation(len(train_rows))
        losses = []
        for start in range(0, len(perm), BATCH):
            index = torch.as_tensor(perm[start : start + BATCH], device=device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(tr_step[index], tr_cond[index]), tr_y[index]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        p = np.clip(predict(model, val_x, mean, std, device), 1e-7, 1 - 1e-7)
        vnll = float(np.mean(-(val_y * np.log(p) + (1 - val_y) * np.log(1 - p))))
        if best is None or vnll < best["val_nll"]:
            best = {
                "epoch": epoch,
                "val_nll": vnll,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
    model.load_state_dict(best["state"])
    model.eval()
    return model, mean, std, best


def evaluate_ensemble(models_pack, rows, device):
    ps = []
    xs = np.stack([make_x(r["mu"], r["force"]) for r in rows])
    y = np.asarray([r["y"] for r in rows], np.float32)
    for model, mean, std, _ in models_pack:
        ps.append(predict(model, xs, mean, std, device))
    p = np.mean(ps, axis=0)
    out = metrics(y, p)
    per = []
    for row, prob in zip(rows, p):
        per.append(
            {
                "context_id": row["context_id"],
                "force": row["force"],
                "y": row["y"],
                "p": float(prob),
                "pred": int(prob >= 0.5),
            }
        )
    return out, per


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Need CUDA")
    torch.use_deterministic_algorithms(False)
    labels, disagreements = load_labels()
    contexts = sorted({row["context_id"] for row in labels})
    by_ctx = defaultdict(list)
    for row in labels:
        by_ctx[row["context_id"]].append(row)

    # Held-out split: train 200002, val 200003, test remaining (200004 + 200010).
    def split_of(cid: str) -> str:
        if "seed200002" in cid:
            return "TRAIN"
        if "seed200003" in cid:
            return "VAL"
        return "TEST"

    split_rows = defaultdict(list)
    for row in labels:
        split_rows[split_of(row["context_id"])].append(row)

    models_pack = []
    for seed in SEEDS:
        model, mean, std, best = train_one(split_rows["TRAIN"], split_rows["VAL"], device, seed)
        models_pack.append((model, mean, std, best))
        print("SEED", seed, "val_nll", best["val_nll"], "epoch", best["epoch"], flush=True)

    results = {"splits": {}, "loco": []}
    for name in ["TRAIN", "VAL", "TEST"]:
        stats, per = evaluate_ensemble(models_pack, split_rows[name], device)
        results["splits"][name] = {"metrics": stats, "n_contexts": len({r["context_id"] for r in split_rows[name]})}
        write(HERE / f"PRED_{name}.json", per)
        print(name, stats, flush=True)

    # Leave-one-context-out with F,μ only.
    for hold in contexts:
        train = [r for r in labels if r["context_id"] != hold]
        test = by_ctx[hold]
        val = train[:: max(1, len(train) // 16)]
        pack = []
        for seed in SEEDS:
            model, mean, std, best = train_one(train, val, device, seed)
            pack.append((model, mean, std, best))
        stats, _ = evaluate_ensemble(pack, test, device)
        results["loco"].append({"held_out": hold, "metrics": stats})
        print("LOCO", hold, stats["accuracy"], "fail_recall", stats["fail_recall"], flush=True)

    loco_acc = [x["metrics"]["accuracy"] for x in results["loco"]]
    summary = {
        "created_utc": now(),
        "claim_boundary": "scripted play_once labels; F+μ features; not π0 / not online AF",
        "architecture": "native dump phase-free GRU-10 + 54-D condition, 3-seed ensemble",
        "force_normalization_N": FORCE_NORM,
        "n_labels": len(labels),
        "n_contexts": len(contexts),
        "n_disagreements_excluded": len(disagreements),
        "disagreements": disagreements,
        "split": {k: results["splits"][k]["metrics"] for k in results["splits"]},
        "loco_mean_accuracy": float(np.mean(loco_acc)),
        "loco": results["loco"],
        "device": str(device),
    }
    write(HERE / "FINAL_RESULTS.json", summary)
    print(json.dumps({k: summary[k] for k in ["n_labels", "split", "loco_mean_accuracy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
