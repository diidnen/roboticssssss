#!/usr/bin/env python3
"""Train/qualify current phase-free full-task MASS feasibility."""
from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/exouser/FORTE")
sys.path[:0] = [str(HERE), str(ROOT)]
import mass_belief
from mass_feasibility import Network, INTERFACE

KEEP_INPUT = list(range(6)) + list(range(13, 17)) + list(range(17, 71))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(clean(value), stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def csv_new(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def verify_freeze():
    plan_path = HERE / "MASS_TRAINING_CONTEXT_PLAN.json"
    manifest_path = HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"
    protocol_path = HERE / "MASS_TRAINING_PROTOCOL.json"
    plan, manifest, protocol = load(plan_path), load(manifest_path), load(protocol_path)
    if sha(plan_path) != manifest["context_plan_sha256"] or sha(manifest_path) != protocol["runtime_manifest_sha256"]:
        raise RuntimeError("MASS training freeze changed")
    if not load(HERE / "MASS_BELIEF_QUALIFICATION.json").get("qualified"):
        raise RuntimeError("MASS belief did not authorize feasibility")
    for path, digest in manifest["source_hashes"].items():
        if sha(path) != digest: raise RuntimeError("frozen source changed: " + path)
    return plan, manifest, protocol


def all_rows(split):
    plan, manifest, _ = verify_freeze()
    output = []
    for context in plan["contexts"]:
        if context["split"] != split: continue
        reference = HERE / "references" / context["id"]
        for force in manifest["candidate_forces_N"]:
            job = HERE / "branches" / context["id"] / f"F{force:g}"
            completion, result = load(job / "WORKER_COMPLETION.json"), load(job / "BRANCH_RESULT.json")
            if not completion.get("logical_success") or not result.get("passed") or not result["outcome"].get("label_valid"):
                continue
            base71 = np.load(job / "PREACTION_SEQUENCE.npy", allow_pickle=False).astype(np.float32)
            if base71.shape != (8, 71): raise ValueError("wrong controlled preaction shape")
            x = base71[:, KEEP_INPUT].copy()
            if x.shape != (8, 64): raise AssertionError
            x[:, 10] = float(force) / 8.0; x[:, 11] = float(context["mass_kg"])
            output.append({"context": context, "force": float(force), "x": x,
                           "y": float(result["full_task_success_y"]), "job": job, "reference": reference})
    if not output: raise RuntimeError("no valid MASS feasibility rows in " + split)
    return output


def metric(y, probability):
    y = np.asarray(y, float); p = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7)
    positives, negatives = int(y.sum()), len(y) - int(y.sum())
    order = np.argsort(-p); ranked_y = y[order]
    auprc = float((np.cumsum(ranked_y) / np.arange(1, len(y) + 1))[ranked_y == 1].mean()) if positives else None
    order_up = np.argsort(p, kind="mergesort"); ranks = np.empty(len(p), float); i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and p[order_up[j]] == p[order_up[i]]: j += 1
        ranks[order_up[i:j]] = (i + 1 + j) / 2; i = j
    auroc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)) if positives and negatives else None
    ece = 0.0
    for index in range(10):
        lo, hi = index / 10, (index + 1) / 10
        mask = (p >= lo) & ((p < hi) if index < 9 else (p <= hi))
        if mask.any(): ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return {"n": len(y), "positive_count": positives,
            "NLL": float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))),
            "Brier": float(np.mean((p - y) ** 2)), "AUROC": auroc, "AUPRC": auprc, "ECE_10bin": ece}


def tensors(rows, mean, std, device):
    x = (np.stack([row["x"] for row in rows]) - mean[None, None]) / std[None, None]
    return (torch.as_tensor(x[:, :, :10], dtype=torch.float32, device=device),
            torch.as_tensor(x[:, 0, 10:], dtype=torch.float32, device=device),
            torch.as_tensor([row["y"] for row in rows], dtype=torch.float32, device=device))


def model_probabilities(model, x, device):
    value = torch.as_tensor(x, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad(): return torch.sigmoid(model(value[:, :, :10], value[:, 0, 10:])).cpu().numpy()


def posterior_for(reference, belief):
    import csv as csv_module
    with (reference / "RAW_PROBE.csv").open() as stream: raw = list(csv_module.DictReader(stream))
    readback = load(reference / "CONTACT_PATCH_READBACK.json")
    pred = belief.rows(raw, readback, query_runtime_manifest_sha256=sha(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"))
    return pred["integration_nodes"], pred["integration_weights"]


def posterior_predictions(models, rows, mean, std, device, belief):
    groups = defaultdict(list)
    for row in rows: groups[row["context"]["id"]].append(row)
    results = {}
    for cid, group in groups.items():
        nodes, weights = posterior_for(group[0]["reference"], belief)
        base = group[0]["x"].copy(); forces = np.asarray([row["force"] for row in group])
        all_forces, masses = np.repeat(forces, len(nodes)), np.tile(nodes, len(forces))
        x = np.broadcast_to(base, (len(all_forces),) + base.shape).copy()
        x[:, :, 10] = all_forces[:, None] / 8.0; x[:, :, 11] = masses[:, None]
        x = (x - mean[None, None]) / std[None, None]
        values = np.mean([model_probabilities(model, x, device) for model in models], axis=0)
        values = values.reshape(len(forces), len(nodes)) @ weights
        for row, value in zip(group, values): results[(cid, row["force"])] = float(value)
    return np.asarray([results[(row["context"]["id"], row["force"])] for row in rows])


def prevalence_nll(rows):
    y = np.asarray([r["y"] for r in rows]); p = np.clip(y.mean(), 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def main():
    plan, manifest, protocol = verify_freeze()
    lock = HERE / "MASS_FEASIBILITY_CHECKPOINT_SELECTION_LOCK.json"
    if lock.exists(): raise RuntimeError("MASS feasibility already trained; refusing rerun")
    expected = {split: sum(c["split"] == split for c in plan["contexts"]) * len(manifest["candidate_forces_N"])
                for split in ("TRAIN", "VAL", "HELDOUT")}
    train, val = all_rows("TRAIN"), all_rows("VAL")
    if len(train) != expected["TRAIN"] or len(val) != expected["VAL"]: raise RuntimeError("all TRAIN/VAL force branches required")
    values = np.stack([row["x"] for row in train]); mean = values.mean((0, 1)); std = values.std((0, 1)); std[std < 1e-6] = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_step, tr_cond, tr_y = tensors(train, mean, std, device)
    belief = mass_belief.ContinuousMassBelief(HERE / "MASS_BELIEF_MANIFEST.json",
        manifest_sha256=sha(HERE / "MASS_BELIEF_MANIFEST.json"))
    recipe = protocol["feasibility_recipe"]
    models, checkpoints, histories = [], [], []
    for seed in recipe["seeds"]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        model = Network().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
        rng = np.random.default_rng(seed); best = None
        for epoch in range(1, recipe["epochs"] + 1):
            model.train(); permutation = rng.permutation(len(train)); losses = []
            for start in range(0, len(train), 64):
                index = torch.as_tensor(permutation[start:start + 64], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(tr_step[index], tr_cond[index]), tr_y[index])
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss))
            val_p = posterior_predictions([model], val, mean, std, device, belief)
            val_nll = metric([row["y"] for row in val], val_p)["NLL"]
            histories.append({"seed": seed, "epoch": epoch, "train_BCE": float(np.mean(losses)), "val_posterior_NLL": val_nll})
            if best is None or val_nll < best["NLL"]:
                best = {"NLL": val_nll, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        model.load_state_dict(best["state"]); model.eval()
        checkpoint = HERE / f"MASS_FEASIBILITY_seed{seed}.pt"
        torch.save({"state_dict": best["state"], "seed": seed, "selected_epoch": best["epoch"],
                    "validation_posterior_NLL": best["NLL"], "normalization_mean": mean,
                    "normalization_std": std, "architecture": "GRU(10,64)+MLP(54,64)+head",
                    "target": "full_task_success_y", "phase_representation": "NONE",
                    "training_protocol_sha256": sha(HERE / "MASS_TRAINING_PROTOCOL.json")}, checkpoint)
        checkpoints.append({"seed": seed, "path": str(checkpoint), "sha256": sha(checkpoint),
                            "selected_epoch": best["epoch"], "validation_posterior_NLL": best["NLL"]})
        models.append(model); print(json.dumps(checkpoints[-1]), flush=True)
    csv_new(HERE / "MASS_FEASIBILITY_TRAINING_HISTORY.csv", histories)
    write(lock, {"rule": recipe["checkpoint_rule"], "checkpoints": checkpoints,
                 "heldout_labels_accessed_before_lock": False, "trainer_sha256": sha(__file__)})

    heldout = all_rows("HELDOUT")
    if len(heldout) != expected["HELDOUT"]: raise RuntimeError("all HELDOUT force branches required")
    results, predictions = {}, []
    for split, rows in (("TRAIN", train), ("VAL", val), ("HELDOUT", heldout)):
        p = posterior_predictions(models, rows, mean, std, device, belief)
        results[split] = metric([row["y"] for row in rows], p)
        results[split]["prevalence_NLL"] = prevalence_nll(rows)
        for row, probability in zip(rows, p):
            predictions.append({"split": split, "context_id": row["context"]["id"], "task": row["context"]["task"],
                "root": row["context"]["root"], "true_mass_kg": row["context"]["mass_kg"],
                "force_N": row["force"], "full_task_success_y": int(row["y"]), "p_success": float(probability),
                "branch_path": str(row["job"])})
    csv_new(HERE / "MASS_FEASIBILITY_PREDICTIONS.csv", predictions)
    groups = defaultdict(list)
    for row in predictions:
        if row["split"] == "HELDOUT": groups[(row["task"], row["root"], row["true_mass_kg"])].append(row)
    force_sensitivity = [max(g, key=lambda r: r["force_N"])["p_success"] - min(g, key=lambda r: r["force_N"])["p_success"] for g in groups.values()]
    gate = recipe["qualification"]; held = results["HELDOUT"]
    better = all(results[s]["NLL"] < results[s]["prevalence_NLL"] for s in ("VAL", "HELDOUT"))
    auroc_ok = held["AUROC"] is None or held["AUROC"] >= gate["HELDOUT_AUROC_min_if_defined"]
    qualified = bool(better and auroc_ok and np.mean(force_sensitivity) >= gate["mean_predicted_p5_minus_p3_min"])
    qualification = {"qualified": qualified, "status": "PASS" if qualified else "FAIL", "predeclared_gate": gate,
        "metrics": results, "VAL_and_HELDOUT_NLL_better_than_prevalence": better,
        "mean_predicted_p5_minus_p3": float(np.mean(force_sensitivity)), "label_valid_fraction": 1.0,
        "development_online_qualification_authorized": qualified, "outcome_used_for_model_selection": False}
    write(HERE / "MASS_FEASIBILITY_METRICS.json", results)
    write(HERE / "MASS_FEASIBILITY_QUALIFICATION.json", qualification)
    sources = [Path(__file__), HERE / "mass_feasibility.py"]
    feasibility_manifest = {"version": "CURRENT_PHASE_FREE_MASS_FULLTASK_FEASIBILITY_V1", "qualified": qualified,
        "posterior_interface": INTERFACE, "label_target": "full_task_success_y", "phase_representation": "NONE",
        "input_shape": [8, 64], "sequence_dim": 10, "condition_dim": 54,
        "candidate_force_normalization": "F/8", "latent_mass_column": 11,
        "force_support": [3.0, 5.0], "planner_grid_step": 0.05,
        "utility": "p*(5-F)/5 + (1-p)*(-1); lower-force tie break",
        "checkpoints": checkpoints, "source_hashes": {str(path): sha(path) for path in sources},
        "training_protocol_sha256": sha(HERE / "MASS_TRAINING_PROTOCOL.json"),
        "qualification_path": str(HERE / "MASS_FEASIBILITY_QUALIFICATION.json"),
        "qualification_sha256": sha(HERE / "MASS_FEASIBILITY_QUALIFICATION.json"),
        "controlled_motion_auxiliary_training": True, "online_vla_transfer_requires_development_qualification": True}
    write(HERE / "MASS_FEASIBILITY_MANIFEST.json", feasibility_manifest)
    print(json.dumps(clean(qualification), indent=2))


if __name__ == "__main__":
    main()
