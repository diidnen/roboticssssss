#!/usr/bin/env python3
"""Train and qualify the frozen current-contract continuous MASS belief."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/exouser/FORTE")
POSITIVE = ROOT / "analysis/results/current58_positive_posterior_contract_repair_20260905"
sys.path[:0] = [str(HERE), str(ROOT), str(POSITIVE)]
from current_contract_belief_features import ProbeEvidence, SCHEMA_ID
from current_contract_physical_belief import FrictionMember, batch, predict
from positive_posterior import PositivePosterior
import mass_belief


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(clean(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def csv_new(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def verify_freeze():
    plan_path = HERE / "MASS_TRAINING_CONTEXT_PLAN.json"
    manifest_path = HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"
    protocol_path = HERE / "MASS_TRAINING_PROTOCOL.json"
    plan, manifest, protocol = load(plan_path), load(manifest_path), load(protocol_path)
    if sha(plan_path) != manifest["context_plan_sha256"] or sha(manifest_path) != protocol["runtime_manifest_sha256"]:
        raise RuntimeError("MASS training freeze changed")
    for path, digest in manifest["source_hashes"].items():
        if sha(path) != digest: raise RuntimeError("frozen source changed: " + path)
    return plan, manifest, protocol


def examples(split):
    plan, _, _ = verify_freeze()
    output = []
    for context in plan["contexts"]:
        if context["split"] != split: continue
        job = HERE / "references" / context["id"]
        completion = load(job / "WORKER_COMPLETION.json")
        result = load(job / "RESULT.json")
        if not completion.get("logical_success") or not result.get("probe_qualified"):
            continue
        evidence = ProbeEvidence().load(job)
        np.testing.assert_array_equal(evidence, np.load(job / "RAW_FEATURES.npy", allow_pickle=False))
        output.append({**context, "raw": evidence, "path": job})
    if not output: raise RuntimeError("no admitted probes in " + split)
    return output


def predict_ensemble(models, rows):
    member_means, member_sigmas = [], []
    for model in models:
        means, sigmas = predict(model, [row["x"] for row in rows])
        member_means.append(means); member_sigmas.append(sigmas)
    return np.asarray(member_means).T, np.asarray(member_sigmas).T


def pairwise_ranking(rows, predicted):
    groups = {}
    for row, value in zip(rows, predicted):
        groups.setdefault((row["task"], row["root"]), []).append((row["mass_kg"], value))
    correct = total = ties = 0
    for group in groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[i][0] == group[j][0]: continue
                total += 1
                delta = (group[i][1] - group[j][1]) * (group[i][0] - group[j][0])
                correct += delta > 0
                ties += delta == 0
    return {"correct": int(correct), "ties": int(ties), "total": total,
            "accuracy": float(correct / total) if total else None}


def metrics(rows, means, sigmas):
    y = np.asarray([row["mass_kg"] for row in rows], float)
    posteriors = [PositivePosterior(tuple(m), tuple(np.log(s)), (1 / 3,) * 3) for m, s in zip(means, sigmas)]
    predicted = np.asarray([p.moments()["mean"] for p in posteriors])
    levels = {0.68: (0.16, 0.84), 0.90: (0.05, 0.95), 0.95: (0.025, 0.975)}
    coverage = {}
    intervals = {}
    for level, (lo, hi) in levels.items():
        bounds = np.asarray([[p.ppf(lo), p.ppf(hi)] for p in posteriors])
        coverage[str(level)] = float(np.mean((y >= bounds[:, 0]) & (y <= bounds[:, 1])))
        intervals[str(level)] = bounds.tolist()
    rho = float(spearmanr(y, predicted).statistic) if np.std(predicted) else None
    base = {
        "count": len(rows), "MAE": float(np.mean(np.abs(predicted - y))),
        "RMSE": float(np.sqrt(np.mean((predicted - y) ** 2))),
        "bias": float(np.mean(predicted - y)), "Spearman": rho,
        "coverage": coverage, "intervals": intervals,
        "pairwise_ranking": pairwise_ranking(rows, predicted),
        "distinct_posterior_means": len(set(np.round(predicted, 10))),
    }
    groups = {}
    for name, key in (("task", "task"), ("root", "root")):
        groups[name] = {}
        for value in sorted({row[key] for row in rows}):
            idx = [i for i, row in enumerate(rows) if row[key] == value]
            yy, pp = y[idx], predicted[idx]
            groups[name][str(value)] = {
                "count": len(idx), "MAE": float(np.mean(np.abs(pp - yy))),
                "RMSE": float(np.sqrt(np.mean((pp - yy) ** 2))),
                "bias": float(np.mean(pp - yy)),
                "Spearman": float(spearmanr(yy, pp).statistic) if np.std(pp) and np.std(yy) else None,
                "ranking_accuracy": pairwise_ranking([rows[i] for i in idx], pp)["accuracy"],
                "distinct_posterior_means": len(set(np.round(pp, 10))),
            }
    base["by"] = groups
    return base, predicted, posteriors


def main():
    plan, manifest, protocol = verify_freeze()
    lock = HERE / "MASS_BELIEF_CHECKPOINT_SELECTION_LOCK.json"
    if lock.exists(): raise RuntimeError("MASS belief already trained; refusing lucky-seed rerun")
    expected = {split: sum(row["split"] == split for row in plan["contexts"]) for split in ("TRAIN", "VAL", "HELDOUT")}
    train, val = examples("TRAIN"), examples("VAL")
    if len(train) != expected["TRAIN"] or len(val) != expected["VAL"]:
        raise RuntimeError("all planned TRAIN/VAL probes must be valid before fitting")
    joined = np.concatenate([row["raw"] for row in train])
    mean = joined.mean(0); std = np.maximum(joined.std(0), 1e-6)
    normalization = {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "TRAIN",
                     "fit_contexts": len(train), "fit_roots": sorted({r["root"] for r in train}),
                     "fit_unpadded_steps": len(joined), "std_floor": 1e-6}
    for row in train + val: row["x"] = ((row["raw"] - mean) / std).astype(np.float32)
    root_groups = {root: [row for row in train if row["root"] == root] for root in sorted({r["root"] for r in train})}
    models, checkpoints, history = [], [], []
    recipe = protocol["belief_recipe"]
    for seed in recipe["seeds"]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = FrictionMember(58, 16, 16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        rng = np.random.default_rng(seed); best = None
        for epoch in range(1, recipe["epochs"] + 1):
            sampled = rng.choice(list(root_groups), size=len(root_groups), replace=True)
            fit = [row for root in sampled for row in root_groups[int(root)]]
            x, lengths = batch([row["x"] for row in fit])
            y = torch.tensor([row["mass_kg"] for row in fit], dtype=torch.float32)
            model.train(); optimizer.zero_grad(set_to_none=True)
            mu, log_sigma = model(x, lengths)
            loss = (0.5 * (((y - mu) / log_sigma.exp()) ** 2 + 2 * log_sigma)).mean() + 0.05 * torch.abs(mu - y).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            vm, vs = predict(model, [row["x"] for row in val]); vy = np.asarray([row["mass_kg"] for row in val])
            val_nll = float(np.mean(0.5 * (((vy - vm) / vs) ** 2 + 2 * np.log(vs) + np.log(2 * np.pi))))
            history.append({"seed": seed, "epoch": epoch, "train_loss": float(loss), "val_gaussian_nll": val_nll})
            if best is None or val_nll < best["nll"]:
                best = {"nll": val_nll, "epoch": epoch,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        model.load_state_dict(best["state"]); model.eval()
        checkpoint = HERE / f"MASS_BELIEF_seed{seed}.pt"
        torch.save({"state_dict": best["state"], "input_dim": 58, "projection_dim": 16, "hidden_dim": 16,
                    "seed": seed, "selected_epoch": best["epoch"], "selected_val_nll": best["nll"],
                    "feature_schema_id": SCHEMA_ID, "normalization": normalization,
                    "training_protocol_sha256": sha(HERE / "MASS_TRAINING_PROTOCOL.json")}, checkpoint)
        entry = {"seed": seed, "path": str(checkpoint), "sha256": sha(checkpoint),
                 "selected_epoch": best["epoch"], "selected_val_nll": best["nll"]}
        checkpoints.append(entry); models.append(model)
        print(json.dumps({"locked_seed": seed, "epoch": best["epoch"], "val_nll": best["nll"]}), flush=True)
    csv_new(HERE / "MASS_BELIEF_TRAINING_HISTORY.csv", history)
    write(lock, {"rule": recipe["checkpoint_rule"], "checkpoints": checkpoints,
                 "heldout_labels_accessed_before_lock": False,
                 "trainer_sha256": sha(__file__)})

    heldout = examples("HELDOUT")
    if len(heldout) != expected["HELDOUT"]: raise RuntimeError("all HELDOUT probes required")
    for row in heldout: row["x"] = ((row["raw"] - mean) / std).astype(np.float32)
    all_metrics, prediction_rows = {}, []
    for split, rows in (("TRAIN", train), ("VAL", val), ("HELDOUT", heldout)):
        mm, ss = predict_ensemble(models, rows)
        result, predicted, posteriors = metrics(rows, mm, ss)
        prior_mean = float(np.mean([r["mass_kg"] for r in train]))
        result["train_prior_MAE"] = float(np.mean(np.abs(np.asarray([r["mass_kg"] for r in rows]) - prior_mean)))
        all_metrics[split] = result
        for row, estimate, p, m, s in zip(rows, predicted, posteriors, mm, ss):
            moments = p.moments()
            prediction_rows.append({"split": split, "context_id": row["id"], "task": row["task"],
                "root": row["root"], "true_mass_kg": row["mass_kg"], "posterior_mean_kg": estimate,
                "posterior_std_kg": moments["std"], "member_means_kg": json.dumps(m.tolist()),
                "member_sigmas_kg": json.dumps(s.tolist()), "probe_path": str(row["path"])})
    csv_new(HERE / "MASS_BELIEF_PREDICTIONS.csv", prediction_rows)
    gate = recipe["qualification"]; h = all_metrics["HELDOUT"]
    better = all(all_metrics[s]["MAE"] < all_metrics[s]["train_prior_MAE"] for s in ("VAL", "HELDOUT"))
    admitted = len(train + val + heldout) / len(plan["contexts"])
    qualified = bool(better and h["MAE"] <= gate["HELDOUT_MAE_kg_max"] and
                     h["Spearman"] is not None and h["Spearman"] >= gate["HELDOUT_SPEARMAN_min"] and
                     h["pairwise_ranking"]["accuracy"] >= gate["HELDOUT_pairwise_ranking_accuracy_min"] and
                     h["coverage"]["0.9"] >= gate["HELDOUT_90_coverage_min"] and
                     admitted >= gate["probe_admission_fraction_min"])
    qualification = {"qualified": qualified, "status": "PASS" if qualified else "FAIL",
        "predeclared_gate": gate, "probability_mean_better_than_train_prior": better,
        "probe_admission_fraction": admitted, "heldout_metrics": h,
        "feasibility_collection_authorized": qualified, "outcome_used_for_gate": False}
    write(HERE / "MASS_BELIEF_METRICS.json", all_metrics)
    write(HERE / "MASS_BELIEF_QUALIFICATION.json", qualification)
    sources = [Path(__file__), HERE / "mass_belief.py", ROOT / "current_contract_belief_features.py",
               ROOT / "current_contract_physical_belief.py", POSITIVE / "positive_posterior.py"]
    belief_manifest = {
        "interface": mass_belief.INTERFACE, "feature_dim": 58, "feature_schema_id": SCHEMA_ID,
        "qualified_for_current_runtime": qualified, "qualified_tasks": [0, 1, 5, 6] if qualified else [],
        "mass_unit": "kg", "support": {"lower_open": 0.0, "upper": None},
        "training_support_type": "DISCRETE_ANCHORS", "training_mass_anchors_kg": plan["mass_anchors_kg"],
        "checkpoints": checkpoints, "normalization": normalization,
        "compatible_query_runtime_manifest_sha256": sha(HERE / "MASS_TRAINING_RUNTIME_MANIFEST.json"),
        "source_hashes": {str(path): sha(path) for path in sources},
        "sigma_heads_integrated_by_planner": True, "ground_truth_mass_deployment_input": False,
        "qualification_path": str(HERE / "MASS_BELIEF_QUALIFICATION.json"),
        "qualification_sha256": sha(HERE / "MASS_BELIEF_QUALIFICATION.json"),
    }
    write(HERE / "MASS_BELIEF_MANIFEST.json", belief_manifest)
    runtime = mass_belief.ContinuousMassBelief(HERE / "MASS_BELIEF_MANIFEST.json",
                                               manifest_sha256=sha(HERE / "MASS_BELIEF_MANIFEST.json"),
                                               diagnostic_only=not qualified)
    parity = 0.0
    for row in train + val + heldout:
        expected_row = next(p for p in prediction_rows if p["context_id"] == row["id"] and p["split"] == row["split"])
        got = runtime.array(row["raw"])["posterior_moments"]["mean"]
        parity = max(parity, abs(got - expected_row["posterior_mean_kg"]))
    write(HERE / "MASS_BELIEF_RUNTIME_PARITY.json", {"passed": parity < 1e-9, "max_mean_abs_diff": parity})
    print(json.dumps(clean({"qualification": qualification, "metrics": all_metrics}), indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
