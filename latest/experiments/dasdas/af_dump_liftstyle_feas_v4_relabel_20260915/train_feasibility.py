"""Train dump feasibility on v4 lift-style remainder labels, F in [0.25, 5]."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
SOURCE = HERE
FORCE_NORM = 8.0
FORCE_MAX = 5.0
FORCE_MIN = 0.25
MEMBER_SEEDS = (0, 1, 2)
EPOCHS = 80
BATCH = 64
SPLIT_BY_SEED = {
    200002: "TRAIN",
    200003: "TRAIN",
    200010: "VAL",
    200014: "TEST",
}
EXPECTED_FORCES = [round(0.25 * i, 10) for i in range(1, 21)]
EXPECTED_ROWS = 32 * 20


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


def auroc(y, p):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if y.min() == y.max():
        return None
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    tpr = np.concatenate([[0.0], tp / tp[-1]])
    fpr = np.concatenate([[0.0], fp / fp[-1]])
    area = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(area(tpr, fpr))


def force_tag(force: float) -> str:
    return f"{float(force):g}N"


def load_rows():
    protocol = json.loads((SOURCE / "plan/PROTOCOL.json").read_text())
    contexts = json.loads((SOURCE / "plan/MAIN_CONTEXTS.json").read_text())
    rows = []
    for ctx in contexts:
        rec = json.loads((SOURCE / "main_records" / f"{ctx['id']}.json").read_text())
        raw = SOURCE / "main_raw" / ctx["id"] / "job"
        split = SPLIT_BY_SEED[int(ctx["seed"])]
        for outcome in rec["outcomes"]:
            if outcome.get("unstable_layout"):
                continue
            force = float(outcome["force_N"])
            if force < FORCE_MIN - 1e-9 or force > FORCE_MAX + 1e-9:
                continue
            branch = next(p for p in raw.glob("branch_*") if p.name.endswith("_" + force_tag(force)))
            feat = json.loads((branch / "PREACTION_FEATURE.json").read_text())
            x = np.asarray(feat["sequence"], np.float32).copy()
            if x.shape != (8, 64):
                raise RuntimeError(f"bad feature {branch} {x.shape}")
            x[:, 10] = force / FORCE_NORM
            x[:, 11] = float(ctx["friction"])
            rows.append(
                {
                    "context_id": ctx["id"],
                    "split": split,
                    "force": force,
                    "mu": float(ctx["friction"]),
                    "x": x,
                    "y": int(outcome["success"]),
                }
            )
    forces = sorted({round(r["force"], 10) for r in rows})
    if forces != EXPECTED_FORCES:
        raise RuntimeError(f"unexpected force grid after clip: {forces}")
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    val_y = [r["y"] for r in rows if r["split"] == "VAL"]
    if not val_y or min(val_y) == max(val_y):
        raise RuntimeError(f"VAL still has a single label class: {val_y[:8]} n={len(val_y)}")
    return rows, protocol


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    acc = float(np.mean(pred == y))
    majority = float(max(pos, neg) / max(len(y), 1))
    pclip = np.clip(p, 1e-7, 1 - 1e-7)
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "n": int(len(y)),
        "positives": pos,
        "negatives": neg,
        "accuracy": acc,
        "majority_baseline": majority,
        "accuracy_minus_majority": acc - majority,
        "auroc": auroc(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(np.mean(-(y * np.log(pclip) + (1 - y) * np.log(1 - pclip)))),
        "fail_recall": float(tn / neg) if neg else None,
        "success_recall": float(int(((pred == 1) & (y == 1)).sum()) / pos) if pos else None,
    }


def predict(model, xs, mean, std, device):
    x = (xs - mean[None, None]) / std[None, None]
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(
                torch.as_tensor(x[:, :, :10], device=device),
                torch.as_tensor(x[:, 0, 10:], device=device),
            )
        ).cpu().numpy()


def train_one(train_rows, val_rows, device, seed):
    xs = np.stack([r["x"] for r in train_rows])
    ys = np.asarray([r["y"] for r in train_rows], np.float32)
    mean = xs.mean(axis=(0, 1)).astype(np.float32)
    std = xs.std(axis=(0, 1)).astype(np.float32)
    std[std < 1e-6] = 1.0
    x = (xs - mean[None, None]) / std[None, None]
    tr_step = torch.as_tensor(x[:, :, :10], device=device)
    tr_cond = torch.as_tensor(x[:, 0, 10:], device=device)
    tr_y = torch.as_tensor(ys, device=device)
    pos = float(ys.sum())
    neg = float(len(ys) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    val_x = np.stack([r["x"] for r in val_rows])
    val_y = np.asarray([r["y"] for r in val_rows], np.float32)
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
        for start in range(0, len(perm), BATCH):
            index = torch.as_tensor(perm[start : start + BATCH], device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(tr_step[index], tr_cond[index])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, tr_y[index], pos_weight=pos_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        p = np.clip(predict(model, val_x, mean, std, device), 1e-7, 1 - 1e-7)
        vnll = float(np.mean(-(val_y * np.log(p) + (1 - val_y) * np.log(1 - p))))
        if best is None or vnll < best["val_nll"]:
            best = {
                "epoch": epoch,
                "val_nll": vnll,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
    model.load_state_dict(best["state"])
    return model, mean, std, {"epoch": best["epoch"], "val_nll": best["val_nll"]}


def selection_table(rows, pack, device):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["context_id"]].append(row)
    table = []
    for context_id, items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: r["force"])
        xs = np.stack([r["x"] for r in items])
        ps = np.mean([predict(model, xs, mean, std, device) for model, mean, std, _ in pack], axis=0)
        ys = [int(r["y"]) for r in items]
        forces = [float(r["force"]) for r in items]
        best_i = int(np.argmax(ps))
        true_ok = [f for f, y in zip(forces, ys) if y == 1]
        table.append(
            {
                "context_id": context_id,
                "split": items[0]["split"],
                "mu": items[0]["mu"],
                "selected_force_N": forces[best_i],
                "selected_p": float(ps[best_i]),
                "selected_true_success": ys[best_i],
                "p_success_by_force": {str(f): float(p) for f, p in zip(forces, ps)},
                "y_by_force": {str(f): y for f, y in zip(forces, ys)},
                "true_success_forces_N": true_ok,
            }
        )
    return table


def main():
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    rows, protocol = load_rows()
    by = defaultdict(list)
    for row in rows:
        by[row["split"]].append(row)
    if any(name not in by or not by[name] for name in ("TRAIN", "VAL", "TEST")):
        raise RuntimeError({k: len(v) for k, v in by.items()})
    write(
        HERE / "PROTOCOL.json",
        {
            "version": "AF_DUMP_LIFTSTYLE_FEAS_V4_RELABEL",
            "source": str(SOURCE),
            "source_version": protocol["version"],
            "force_support_N": [FORCE_MIN, FORCE_MAX],
            "force_step_N": 0.25,
            "force_norm": FORCE_NORM,
            "n_rows": EXPECTED_ROWS,
            "filter": "v4 snapshot-fork rows already on [0.25,5]",
            "no_new_rollouts": True,
            "claim_boundary": (
                "scripted remainder labels on [0.25,5]; "
                "not pi0; not online AF; not official 18/19"
            ),
            "split_by_seed": SPLIT_BY_SEED,
            "created_utc": now(),
        },
    )
    pack = []
    seed_rows = []
    models_dir = HERE / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for seed in MEMBER_SEEDS:
        model, mean, std, best = train_one(by["TRAIN"], by["VAL"], device, seed)
        pack.append((model, mean, std, best))
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "best": best,
                "seed": seed,
                "force_support_N": [FORCE_MIN, FORCE_MAX],
            },
            models_dir / f"member_seed{seed}.pt",
        )
        seed_rows.append({"seed": seed, **best})
        print("SEED", seed, best, flush=True)
    selections = selection_table(rows, pack, device)
    selected = Counter((row["split"], row["selected_force_N"]) for row in selections)
    summary = {
        "created_utc": now(),
        "force_norm": FORCE_NORM,
        "force_support_N": [FORCE_MIN, FORCE_MAX],
        "architecture": "dump_phasefree_gru10_cond54",
        "protocol_version": "AF_DUMP_LIFTSTYLE_FEAS_V4_RELABEL",
        "source": str(SOURCE),
        "claim_boundary": "scripted remainder labels on [0.25,5]; not pi0; not online AF; not official 18/19",
        "n_rows": {name: len(by[name]) for name in ["TRAIN", "VAL", "TEST"]},
        "label_rate": {
            name: float(np.mean([r["y"] for r in by[name]])) for name in ["TRAIN", "VAL", "TEST"]
        },
        "members": seed_rows,
        "splits": {},
        "selections": selections,
        "selected_force_counts": {
            f"{split}:{force:g}": count for (split, force), count in sorted(selected.items())
        },
    }
    for name in ["TRAIN", "VAL", "TEST"]:
        xs = np.stack([r["x"] for r in by[name]])
        y = np.asarray([r["y"] for r in by[name]], np.float32)
        p = np.mean([predict(model, xs, mean, std, device) for model, mean, std, _ in pack], axis=0)
        summary["splits"][name] = metrics(y, p)
        print(name, summary["splits"][name], flush=True)
    print("SELECTED", summary["selected_force_counts"], flush=True)
    write(HERE / "TRAIN_RESULTS.json", summary)


if __name__ == "__main__":
    main()
