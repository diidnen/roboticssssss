#!/usr/bin/env python3
"""Diagnostic Direct-utility proposal followed by trajectory-only verification.

The frozen Fixed-Scene Joint-Original physics heads produce H=8 physical
trajectories.  A separate linear evaluator is trained only on summaries of
those predicted trajectories.  It never sees force, task, root/context id,
Direct scores, Joint latent features, or DEV labels during fitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


FORTE = Path("/home/exouser/FORTE")
FIXED = FORTE / "fixed_scene_reframing_20260831_174144"
PREDICTIONS = FORTE / "joint_decision_alignment_20260831_200441" / "JOINT_DECISIONALIGNED_ALL_PREDICTIONS.csv"
SEEDS = [0, 1, 2]
H = 8
EPOCHS = 80
BATCH = 64
LR = 8e-4
WEIGHT_DECAY = 1e-4
VERIFY_THRESHOLD = 0.8
FMAX = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        if rows:
            w.writerows(rows)


def protocol(out: Path) -> dict:
    sources = [
        PREDICTIONS,
        FORTE / "gnp_style_continuous.py",
        FORTE / "train_fixed_scene_exact.py",
        Path("/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py"),
        Path("/home/exouser/Tabero/analysis/counterfactual_force_world_model.py"),
    ] + [FIXED / f"FIXED_SCENE_JOINT_seed{s}.pt" for s in SEEDS]
    return {
        "name": "DIRECT_UTILITY_TRAJECTORY_ONLY_VERIFIER_DIAGNOSTIC",
        "status": "FROZEN_BEFORE_EVALUATOR_TRAINING",
        "scope": "retrospective fixed-scene held-out-friction DEV diagnostic only; no TEST",
        "proposal": {
            "model": "Direct 100%-TRAIN three-seed ensemble",
            "rule": "argmax p_direct*(Fmax-F)+(1-p_direct)*(-Fmax)",
            "C_fail": "Fmax",
            "repeat_reduction": "mean probability per context/force",
        },
        "world_model": {
            "source": "frozen Fixed-Scene Joint-Original physics trajectory head",
            "seeds": SEEDS,
            "horizon": H,
            "output": "denormalized H8x13 predicted physical states",
            "retrained": False,
        },
        "outcome_evaluator": {
            "input": "trajectory only: concat(final, mean, std, max) over H8x13 = 52 dimensions",
            "forbidden_inputs": ["force", "task", "root/context identity", "friction", "Direct probability", "Joint latent", "real future trajectory"],
            "architecture": "Linear(52,1)",
            "loss": "BCEWithLogits",
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seeds": SEEDS,
            "normalization": "per-seed TRAIN predicted-trajectory features only",
            "checkpoint": "final epoch; no DEV selection",
        },
        "verification": {
            "threshold": VERIFY_THRESHOLD,
            "threshold_source": "user-fixed before this diagnostic",
            "strict": "Direct proposal accepted iff verifier p>0.8",
            "execution_fallback": "if rejected, scan higher candidate forces ascending; if none passes, use maximum candidate and mark unvalidated fallback",
        },
        "primary_questions": [
            "Does the trajectory-only verifier reject failed Direct utility proposals without rejecting successful ones?",
            "Does the cascade improve success/under-force/realized utility over Direct utility alone?",
        ],
        "no_tuning": ["architecture", "threshold", "epochs", "optimizer", "feature family", "C_fail", "DEV calibration"],
        "historical_warning": "Prior real-trajectory evaluator reached AUROC 1.0, while imagined-trajectory AUROC was 0.5; this run trains on imagined TRAIN trajectories to remove that train/deploy interface mismatch.",
        "source_hashes": {str(p): sha256(p) for p in sources},
    }


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    p = out / "WORLD_MODEL_VERIFIER_PROTOCOL.json"
    if p.exists():
        print(json.dumps({"status": "ALREADY_FROZEN", "protocol": str(p), "sha256": sha256(p)}, indent=2))
        return
    write_json(p, protocol(out))
    (out / "PRIOR_TRAJECTORY_EVALUATOR_AUDIT.md").write_text(
        "# Prior trajectory-to-outcome evidence\n\n"
        "The prior trajectory physical imagination run trained an outcome evaluator on real trajectories. "
        "It reached TEST AUROC 1.0 on real trajectories, but AUROC 0.5 on imagined trajectories and produced "
        "71 imagined false-safe branches. A later forensic found real-trajectory boundary correctness 1.0 and "
        "saved model-trajectory correctness 0.0 on 11 DEV triplets. Therefore this diagnostic does not repeat "
        "the real-trajectory evaluator: it fits the evaluator directly on TRAIN imagined trajectories and tests "
        "only on held-out-context DEV imagined trajectories.\n\n"
        "Sources:\n"
        "- /home/exouser/Tabero/analysis/results/trajectory_physical_imagination_20260829_065220/FINAL_REPORT.md\n"
        "- /home/exouser/Tabero/analysis/results/force_sensitivity_forensic_20260829_090000/FINAL_REPORT.md\n"
        "- /home/exouser/Tabero/analysis/results/evaluator_interface_calibration_20260829_112603/FINAL_REPORT.md\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "FROZEN", "protocol": str(p), "sha256": sha256(p)}, indent=2))


def trajectory_feature(state: np.ndarray) -> np.ndarray:
    if state.shape != (H, 13):
        raise RuntimeError(f"unexpected predicted trajectory shape {state.shape}")
    return np.concatenate([state[-1], state.mean(0), state.std(0), state.max(0)]).astype(np.float32)


class OutcomeEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(52, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def fit_evaluator(x: np.ndarray, y: np.ndarray, seed: int, device) -> tuple[OutcomeEvaluator, np.ndarray, np.ndarray, list[dict]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    xm = x.mean(0).astype(np.float32)
    xs = x.std(0).astype(np.float32)
    xs[xs < 1e-6] = 1.0
    xn = ((x - xm) / xs).astype(np.float32)
    xt = torch.tensor(xn, device=device)
    yt = torch.tensor(y.astype(np.float32), device=device)
    model = OutcomeEvaluator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history: list[dict] = []
    for epoch in range(1, EPOCHS + 1):
        rng = np.random.default_rng(seed * 1000003 + epoch * 1009 + 313)
        order = rng.permutation(len(x))
        losses = []
        model.train()
        for st in range(0, len(order), BATCH):
            ids = torch.tensor(order[st:st+BATCH], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xt[ids]), yt[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        history.append({"seed": seed, "epoch": epoch, "train_bce": float(np.mean(losses))})
    model.eval()
    return model, xm, xs, history


def auc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "branches": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "AUROC": auc(y, p),
        "MAE": float(np.mean(np.abs(p - y))),
        "Brier": float(np.mean((p - y) ** 2)),
        "NLL": float(np.mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)))),
        "p_mean_success": float(p[y == 1].mean()),
        "p_mean_failure": float(p[y == 0].mean()),
        "failure_p_gt_0p8": float((p[y == 0] > VERIFY_THRESHOLD).mean()),
    }


def selection_summary(rows: list[dict], name: str) -> dict:
    q = pd.DataFrame(rows)
    return {
        "policy": name,
        "contexts": int(len(q)),
        "success_rate": float(q.actual_success_rate.mean()),
        "under_force_rate": float(q.under_force.mean()),
        "mean_force_N": float(q.selected_force_N.mean()),
        "realized_utility": float(q.realized_utility.mean()),
        "validated_rate": float(q.validated.mean()),
        "changed_from_direct_rate": float(q.changed_from_direct.mean()),
    }


def evaluate_decisions(dev_predictions: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    direct = pd.read_csv(PREDICTIONS)
    direct = direct[(direct.fraction == "100%") & (direct.method == "Direct")]
    dcell = direct.groupby(["context_id", "scene_id", "task", "force_N"], as_index=False).agg(
        p_direct=("p_success", "mean"), actual_success=("actual_success", "mean")
    )
    wcell = dev_predictions.groupby(["context_id", "force_N"], as_index=False).agg(p_world=("p_world", "mean"))
    cells = dcell.merge(wcell, on=["context_id", "force_N"], validate="one_to_one")
    if len(cells) != 120:
        raise RuntimeError(f"expected 120 DEV force cells, got {len(cells)}")
    all_rows: list[dict] = []
    gate_rows: list[dict] = []
    for cid, base in cells.groupby("context_id"):
        q = base.sort_values("force_N").copy()
        fmax = FMAX[int(q.task.iloc[0])]
        q["u_direct"] = q.p_direct * (fmax - q.force_N) + (1 - q.p_direct) * (-fmax)
        q["u_real"] = q.actual_success * (fmax - q.force_N) + (1 - q.actual_success) * (-fmax)
        q["u_world"] = q.p_world * (fmax - q.force_N) + (1 - q.p_world) * (-fmax)
        boundary = float(q.loc[q.actual_success > 0, "force_N"].min()) if (q.actual_success > 0).any() else float("nan")
        proposal = q.sort_values(["u_direct", "force_N"], ascending=[False, True]).iloc[0]
        world_choice = q.sort_values(["u_world", "force_N"], ascending=[False, True]).iloc[0]
        accepted = bool(float(proposal.p_world) > VERIFY_THRESHOLD)
        higher = q[(q.force_N >= float(proposal.force_N) - 1e-9) & (q.p_world > VERIFY_THRESHOLD)].sort_values("force_N")
        cascaded = higher.iloc[0] if len(higher) else q.iloc[-1]
        for policy, row, validated in [
            ("DIRECT_UTILITY", proposal, True),
            ("WORLD_UTILITY", world_choice, True),
            ("DIRECT_THEN_WORLD_GT_0P8", cascaded, bool(len(higher))),
        ]:
            all_rows.append({
                "policy": policy,
                "context_id": cid,
                "scene_id": row.scene_id,
                "task": int(row.task),
                "selected_force_N": float(row.force_N),
                "actual_success_rate": float(row.actual_success),
                "p_direct": float(row.p_direct),
                "p_world": float(row.p_world),
                "realized_utility": float(row.u_real),
                "under_force": int(math.isfinite(boundary) and float(row.force_N) < boundary),
                "validated": int(validated),
                "changed_from_direct": int(abs(float(row.force_N) - float(proposal.force_N)) > 1e-9),
                "direct_proposal_force_N": float(proposal.force_N),
            })
        gate_rows.append({
            "context_id": cid,
            "scene_id": proposal.scene_id,
            "task": int(proposal.task),
            "direct_proposal_force_N": float(proposal.force_N),
            "direct_proposal_actual_success_rate": float(proposal.actual_success),
            "direct_proposal_p": float(proposal.p_direct),
            "world_verifier_p": float(proposal.p_world),
            "accepted_p_gt_0p8": int(accepted),
            "correctly_rejected_imperfect": int((not accepted) and float(proposal.actual_success) < 1.0),
            "incorrectly_rejected_perfect": int((not accepted) and float(proposal.actual_success) == 1.0),
            "incorrectly_accepted_imperfect": int(accepted and float(proposal.actual_success) < 1.0),
        })
    summaries = [selection_summary([r for r in all_rows if r["policy"] == p], p) for p in ["DIRECT_UTILITY", "WORLD_UTILITY", "DIRECT_THEN_WORLD_GT_0P8"]]
    return all_rows, summaries, gate_rows


def run(out: Path) -> None:
    protocol_path = out / "WORLD_MODEL_VERIFIER_PROTOCOL.json"
    if not protocol_path.exists():
        raise RuntimeError("run --prepare first")
    frozen = json.loads(protocol_path.read_text())
    if frozen["status"] != "FROZEN_BEFORE_EVALUATOR_TRAINING":
        raise RuntimeError("protocol not frozen")

    gnp = load("gnp_world_verify", FORTE / "gnp_style_continuous.py")
    tpi = load("tpi_world_verify", gnp.TPI_CODE)
    cf = load("cf_world_verify", gnp.CF_CODE)
    train_mod = load("fixed_world_verify", FORTE / "train_fixed_scene_exact.py")
    train, dev, _, _ = train_mod.build_population(FIXED, gnp, tpi, cf)
    if len(train) != 480 or len(dev) != 240:
        raise RuntimeError("fixed-scene population mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_rows: list[dict] = []
    training_rows: list[dict] = []
    metric_rows: list[dict] = []
    checkpoints = []
    for seed in SEEDS:
        world_path = FIXED / f"FIXED_SCENE_JOINT_seed{seed}.pt"
        world_ck = torch.load(world_path, map_location=device, weights_only=False)
        norm_obj = world_ck["normalization"]
        norm = tuple(np.asarray(norm_obj[k], np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"])
        world = tpi.ShortHorizonPhysicsGRU(17, 54, H).to(device)
        world.load_state_dict(world_ck["physics_state_dict"])
        world.eval()
        x_train = np.stack([trajectory_feature(cf.pred_state(world, tr, tr.force, norm, tpi, device)) for tr in train])
        x_dev = np.stack([trajectory_feature(cf.pred_state(world, tr, tr.force, norm, tpi, device)) for tr in dev])
        y_train = np.asarray([tr.outcome for tr in train], np.float32)
        y_dev = np.asarray([tr.outcome for tr in dev], np.float32)
        evaluator, xm, xs, hist = fit_evaluator(x_train, y_train, seed, device)
        training_rows.extend(hist)
        with torch.no_grad():
            logits = evaluator(torch.tensor((x_dev - xm) / xs, dtype=torch.float32, device=device))
            p_dev = torch.sigmoid(logits).cpu().numpy()
        m = probability_metrics(y_dev, p_dev)
        metric_rows.append({"seed": seed, "split": "DEV", **m})
        ck_path = out / f"TRAJECTORY_ONLY_OUTCOME_EVALUATOR_seed{seed}.pt"
        torch.save({
            "seed": seed,
            "state_dict": evaluator.state_dict(),
            "x_mean": xm.tolist(),
            "x_std": xs.tolist(),
            "feature": "predicted H8x13 state concat(final,mean,std,max)",
            "world_checkpoint": str(world_path),
            "threshold": VERIFY_THRESHOLD,
            "epochs": EPOCHS,
        }, ck_path)
        checkpoints.append({"seed": seed, "path": str(ck_path), "sha256": sha256(ck_path), "world_path": str(world_path), "world_sha256": sha256(world_path)})
        for tr, p in zip(dev, p_dev):
            pred_rows.append({
                "seed": seed,
                "branch_id": tr.branch_id,
                "context_id": tr.context_id,
                "task": int(tr.task),
                "force_N": float(tr.force),
                "repeat": 1 if "_R1_" in tr.branch_id else 2,
                "actual_success": int(tr.outcome),
                "p_world_seed": float(p),
            })
    per_seed = pd.DataFrame(pred_rows)
    ensemble = per_seed.groupby(["branch_id", "context_id", "task", "force_N", "repeat", "actual_success"], as_index=False).agg(
        p_world=("p_world_seed", "mean"), p_world_seed_std=("p_world_seed", "std")
    )
    metric_rows.append({"seed": "ensemble", "split": "DEV", **probability_metrics(ensemble.actual_success.to_numpy(float), ensemble.p_world.to_numpy(float))})
    decision_rows, decision_summaries, gate_rows = evaluate_decisions(ensemble)

    write_csv(out / "WORLD_MODEL_VERIFIER_TRAINING.csv", training_rows)
    write_csv(out / "WORLD_MODEL_VERIFIER_PER_SEED_PREDICTIONS.csv", pred_rows)
    ensemble.to_csv(out / "WORLD_MODEL_VERIFIER_DEV_PREDICTIONS.csv", index=False)
    write_csv(out / "WORLD_MODEL_VERIFIER_PROBABILITY_METRICS.csv", metric_rows)
    write_csv(out / "DIRECT_WORLD_MODEL_VERIFIER_SELECTIONS.csv", decision_rows)
    write_csv(out / "DIRECT_WORLD_MODEL_VERIFIER_SUMMARY.csv", decision_summaries)
    write_csv(out / "DIRECT_WORLD_MODEL_GATE_AUDIT.csv", gate_rows)
    write_json(out / "WORLD_MODEL_VERIFIER_CHECKPOINTS.json", {"device": str(device), "checkpoints": checkpoints})

    ms = {x["policy"]: x for x in decision_summaries}
    gate = pd.DataFrame(gate_rows)
    pm = metric_rows[-1]
    report = "# Direct Utility + Trajectory-Only World-Model Verifier\n\n"
    report += f"The evaluator was trained directly on frozen TRAIN imagined trajectories, removing the old real-to-imagined evaluator distribution mismatch. It sees only H8 predicted trajectory summaries. Device: `{device}`. No TEST was read.\n\n"
    report += "## Probability quality\n\n"
    report += f"DEV branch AUROC={pm['AUROC']:.3f}, Brier={pm['Brier']:.3f}, NLL={pm['NLL']:.3f}; failed-branch p>0.8 rate={pm['failure_p_gt_0p8']:.3f}.\n\n"
    report += "## Decision result\n\n| Policy | SR | Under-force | Mean force | Realized utility | Validated |\n|---|---:|---:|---:|---:|---:|\n"
    for name in ["DIRECT_UTILITY", "WORLD_UTILITY", "DIRECT_THEN_WORLD_GT_0P8"]:
        x = ms[name]
        report += f"| {name} | {x['success_rate']:.3f} | {x['under_force_rate']:.3f} | {x['mean_force_N']:.3f} | {x['realized_utility']:.3f} | {x['validated_rate']:.3f} |\n"
    report += "\n## Gate audit\n\n"
    report += f"Direct proposals accepted={int(gate.accepted_p_gt_0p8.sum())}/{len(gate)}; correctly rejected imperfect={int(gate.correctly_rejected_imperfect.sum())}; incorrectly rejected perfect={int(gate.incorrectly_rejected_perfect.sum())}; incorrectly accepted imperfect={int(gate.incorrectly_accepted_imperfect.sum())}.\n\n"
    report += "This is a retrospective fixed-scene DEV mechanism diagnostic. It does not select a production backend and cannot be reported as fresh task0 TEST evidence.\n"
    (out / "FINAL_WORLD_MODEL_VERIFIER_REPORT.md").write_text(report, encoding="utf-8")

    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "out": str(out), "device": str(device), "probability": pm, "decisions": decision_summaries}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.prepare == args.run:
        raise SystemExit("choose exactly one of --prepare/--run")
    prepare(args.out) if args.prepare else run(args.out)


if __name__ == "__main__":
    main()
