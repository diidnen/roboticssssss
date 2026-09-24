#!/usr/bin/env python3
"""Train the unchanged feasibility model on frozen TRAIN/VAL subset rows only."""
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
from time_bounded_mass_common import load_frozen_subset, validate_branch

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
    subset, subset_sha = load_frozen_subset()
    pause = load(HERE / "ORIGINAL_648_QUEUE_PAUSE_AUDIT.json")
    reuse = load(HERE / "TIME_BOUNDED_MASS_REUSE_SUMMARY.json")
    if not all(pause["checks"].values()) or reuse["hash_or_parity_failures"] != 0:
        raise RuntimeError("pause/reuse gate failed")
    train_val_audit = load(HERE / "TIME_BOUNDED_MASS_TRAIN_VAL_DATA_AUDIT.json")
    heldout_seal = load(HERE / "TIME_BOUNDED_MASS_HELDOUT_SEAL.json")
    if train_val_audit.get("status") != "PASS" or train_val_audit.get("valid_branches") != 180:
        raise RuntimeError("TRAIN/VAL data audit gate failed")
    if heldout_seal.get("branch_count") != 36 or heldout_seal.get("scientific_labels_parsed_or_exported") is not False:
        raise RuntimeError("HELDOUT seal gate failed")
    return plan, manifest, protocol, subset, subset_sha


def all_rows(split):
    plan, manifest, _, subset, _ = verify_freeze()
    contexts = {row["id"]: row for row in plan["contexts"]}
    output = []
    for spec in subset["jobs"]:
        if spec["split"] != split: continue
        context = contexts[spec["context_id"]]; force = float(spec["force_N"])
        reference = HERE / "references" / context["id"]
        job = HERE / spec["relative_branch_path"]
        check = validate_branch(job, spec, manifest, reference)
        if not check["valid"]:
            raise RuntimeError(f"invalid required {split} row {job}: {check['failed_checks']}")
        result = load(job / "BRANCH_RESULT.json")
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
    plan, manifest, protocol, subset, subset_sha = verify_freeze()
    lock = HERE / "TIME_BOUNDED_MASS_FEASIBILITY_CHECKPOINT_SELECTION_LOCK.json"
    if lock.exists(): raise RuntimeError("time-bounded feasibility already trained; refusing rerun")
    expected = subset["split_branch_targets"]
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
        checkpoint = HERE / f"TIME_BOUNDED_MASS_FEASIBILITY_seed{seed}.pt"
        torch.save({"state_dict": best["state"], "seed": seed, "selected_epoch": best["epoch"],
                    "validation_posterior_NLL": best["NLL"], "normalization_mean": mean,
                    "normalization_std": std, "architecture": "GRU(10,64)+MLP(54,64)+head",
                    "target": "full_task_success_y", "phase_representation": "NONE",
                    "time_bounded_subset_sha256": subset_sha,
                    "original_training_protocol_sha256": sha(HERE / "MASS_TRAINING_PROTOCOL.json")}, checkpoint)
        checkpoints.append({"seed": seed, "path": str(checkpoint), "sha256": sha(checkpoint),
                            "selected_epoch": best["epoch"], "validation_posterior_NLL": best["NLL"]})
        models.append(model); print(json.dumps(checkpoints[-1]), flush=True)
    csv_new(HERE / "TIME_BOUNDED_MASS_FEASIBILITY_TRAINING_HISTORY.csv", histories)

    results, predictions = {}, []
    for split, rows in (("TRAIN", train), ("VAL", val)):
        p = posterior_predictions(models, rows, mean, std, device, belief)
        results[split] = metric([row["y"] for row in rows], p)
        results[split]["prevalence_NLL"] = prevalence_nll(rows)
        for row, probability in zip(rows, p):
            predictions.append({"split": split, "context_id": row["context"]["id"], "task": row["context"]["task"],
                "root": row["context"]["root"], "true_mass_kg": row["context"]["mass_kg"],
                "force_N": row["force"], "full_task_success_y": int(row["y"]), "p_success": float(probability),
                "branch_path": str(row["job"])})
    csv_new(HERE / "TIME_BOUNDED_MASS_TRAIN_VAL_PREDICTIONS.csv", predictions)
    groups = defaultdict(list)
    for row in predictions:
        if row["split"] == "VAL": groups[(row["task"], row["root"], row["true_mass_kg"])].append(row)
    force_sensitivity = [max(g, key=lambda r: r["force_N"])["p_success"] - min(g, key=lambda r: r["force_N"])["p_success"] for g in groups.values()]
    gate = recipe["qualification"]; val_metric = results["VAL"]
    better = val_metric["NLL"] < val_metric["prevalence_NLL"]
    auroc_ok = val_metric["AUROC"] is None or val_metric["AUROC"] >= gate["HELDOUT_AUROC_min_if_defined"]
    qualified = bool(better and auroc_ok and np.mean(force_sensitivity) >= gate["mean_predicted_p5_minus_p3_min"])
    selection = {"validation_qualified": qualified, "status": "PASS" if qualified else "FAIL",
        "predeclared_gate_adaptation": "Original heldout metric thresholds applied to VAL before HELDOUT access.",
        "original_gate": gate, "metrics": results, "VAL_NLL_better_than_prevalence": better,
        "mean_predicted_p5_minus_p3": float(np.mean(force_sensitivity)), "label_valid_fraction": 1.0,
        "heldout_labels_accessed": False, "outcome_used_for_model_selection": False}
    write(lock, {"rule": recipe["checkpoint_rule"], "checkpoints": checkpoints,
                 "heldout_labels_accessed_before_lock": False, "trainer_sha256": sha(__file__),
                 "time_bounded_subset_sha256": subset_sha, "validation_selection": selection,
                 "heldout_evaluator_sha256": sha(HERE / "evaluate_time_bounded_mass_heldout_v1.py"),
                 "postheldout_analyzer_sha256": sha(HERE / "analyze_time_bounded_mass_v1.py"),
                 "final_validator_sha256": sha(HERE / "validate_time_bounded_mass_outputs_v1.py"),
                 "analysis_protocol_sha256": sha(HERE / "TIME_BOUNDED_MASS_ANALYSIS_PROTOCOL.json"),
                 "heldout_seal_sha256": sha(HERE / "TIME_BOUNDED_MASS_HELDOUT_SEAL.json"),
                 "dependency_sha256": {
                     str(HERE / "time_bounded_mass_common.py"): sha(HERE / "time_bounded_mass_common.py"),
                     str(HERE / "mass_feasibility.py"): sha(HERE / "mass_feasibility.py"),
                     str(HERE / "mass_belief.py"): sha(HERE / "mass_belief.py"),
                     str(HERE / "MASS_BELIEF_MANIFEST.json"): sha(HERE / "MASS_BELIEF_MANIFEST.json"),
                     str(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"): sha(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"),
                     str(HERE / "TIME_BOUNDED_MASS_TRAIN_VAL_DATA_AUDIT.json"): sha(HERE / "TIME_BOUNDED_MASS_TRAIN_VAL_DATA_AUDIT.json"),
                 }})
    write(HERE / "TIME_BOUNDED_MASS_MODEL_SELECTION_FREEZE.json", selection)
    sources = [Path(__file__), HERE / "mass_feasibility.py"]
    feasibility_manifest = {"version": "TIME_BOUNDED_COARSE_FORCE_MASS_FULLTASK_FEASIBILITY_V1", "validation_qualified": qualified,
        "posterior_interface": INTERFACE, "label_target": "full_task_success_y", "phase_representation": "NONE",
        "input_shape": [8, 64], "sequence_dim": 10, "condition_dim": 54,
        "candidate_force_normalization": "F/8", "latent_mass_column": 11,
        "training_force_anchors_N": [3.0, 4.0, 5.0], "fine_force_calibration_claim": "NOT_TESTED",
        "utility": "p*(5-F)/5 + (1-p)*(-1); lower-force tie break",
        "checkpoints": checkpoints, "source_hashes": {str(path): sha(path) for path in sources},
        "time_bounded_subset_sha256": subset_sha,
        "selection_lock_path": str(lock), "selection_lock_sha256": sha(lock),
        "controlled_motion_auxiliary_training": True, "online_vla_transfer_requires_development_qualification": True}
    write(HERE / "TIME_BOUNDED_MASS_FEASIBILITY_MANIFEST_PREHELDOUT.json", feasibility_manifest)
    print(json.dumps(clean(selection), indent=2))


if __name__ == "__main__":
    main()
