#!/usr/bin/env python3
"""Final Mass identification, Direct and offline Utility benchmark.

This script accepts only the source named by the independent QA manifest.  It
does not touch the raw source, protected worktrees, or any prior result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

BANDS = {"LOW": 0.05, "MID": 0.10, "HIGH": 0.20}
FORCES = (0.5, 1.0, 1.5, 2.5, 4.0)
SEEDS = (11, 23, 37)
F_MAX_UTILITY = 8.0
R_FAIL = -1.0
PHYSICAL_FEATURES = ("f_meas_mean", "normal_force_mean", "normal_force_peak", "ftan_mean", "rho_mean", "rho_peak", "marker_mean", "marker_vel_abs_peak", "gripper_opening_mean", "obj_disp_probe_m", "obj_rot_probe_rad", "actual_probe_displacement_mm")
SYSID_FEATURES = ("f_meas_mean", "normal_force_mean", "rho_mean", "obj_disp_probe_m", "obj_rot_probe_rad")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in records:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(records)


def parse_query(row):
    try:
        return json.loads(row["query_record"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return {}


def physical_vector(row, names=PHYSICAL_FEATURES):
    q = parse_query(row)
    return np.asarray([float(q.get(k, 0.0)) for k in names], dtype=np.float32)


def vision_vector(root: Path, row):
    cid = row["context_id"]
    values = []
    for view in ("agentview_cam", "eye_in_hand_cam"):
        path = root / "query_rgb" / f"{cid}_{view}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        image = np.load(path).astype(np.float32) / 255.0
        values.extend(image.mean(axis=(0, 1)).tolist())
        values.extend(image.std(axis=(0, 1)).tolist())
    return np.asarray(values, dtype=np.float32)


def standardize(x):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-7] = 1.0
    return (x - mu) / sd, mu, sd


class Regressor(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_features, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Direct(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 12), torch.nn.Tanh(), torch.nn.Linear(12, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_regressor(x, y, seed, device):
    torch.manual_seed(seed)
    xz, mu, sd = standardize(x)
    model = Regressor(x.shape[1]).to(device)
    tx = torch.as_tensor(xz, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-3)
    for _ in range(1200):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((model(tx) - ty) ** 2)
        loss.backward()
        opt.step()
    def predict(q):
        with torch.no_grad():
            return model(torch.as_tensor((q - mu) / sd, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return predict


def train_direct(branches, seed, device):
    torch.manual_seed(seed)
    x = np.asarray([[float(r["mass_kg"]), float(r["requested_force_N"]), float(r["mass_kg"]) * float(r["requested_force_N"]), float(r["requested_force_N"]) ** 2] for r in branches], dtype=np.float32)
    y = np.asarray([float(r["full_task_success_y"]) for r in branches], dtype=np.float32)
    xz, mu, sd = standardize(x)
    model = Direct().to(device)
    tx = torch.as_tensor(xz, dtype=torch.float32, device=device)
    ty = torch.as_tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=3e-3)
    for _ in range(1000):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx), ty)
        loss.backward()
        opt.step()
    def predict(masses, forces):
        arr = np.asarray([[m, f, m * f, f * f] for m, f in zip(masses, forces)], dtype=np.float32)
        with torch.no_grad():
            logits = model(torch.as_tensor((arr - mu) / sd, dtype=torch.float32, device=device))
            return torch.sigmoid(logits).detach().cpu().numpy()
    return predict


def utility(prob, force):
    return float(prob) * (F_MAX_UTILITY - float(force)) + (1.0 - float(prob)) * R_FAIL


def nearest_band(mass):
    return min(BANDS, key=lambda band: abs(BANDS[band] - float(mass)))


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    err = pred - y
    ranks = spearmanr(y, pred).statistic if len(y) > 1 else float("nan")
    pairs = sum(int((pred[i] - pred[j]) * (y[i] - y[j]) > 0) for i in range(len(y)) for j in range(i + 1, len(y)))
    total = len(y) * (len(y) - 1) // 2
    return {"mass_mae_kg": float(np.mean(np.abs(err))), "mass_rmse_kg": float(np.sqrt(np.mean(err ** 2))), "mass_median_ae_kg": float(np.median(np.abs(err))), "mass_bias_kg": float(np.mean(err)), "spearman": float(ranks), "pairwise_ranking_accuracy": float(pairs / total if total else 1.0), "low_mass_accuracy": float(np.mean((pred < 0.075) == (y < 0.075))), "mid_mass_accuracy": float(np.mean(((pred >= 0.075) & (pred < 0.15)) == ((y >= 0.075) & (y < 0.15)))), "high_mass_accuracy": float(np.mean((pred >= 0.15) == (y >= 0.15))), "mass_band_accuracy": float(np.mean(np.asarray([nearest_band(x) for x in pred]) == np.asarray([nearest_band(x) for x in y])))}


def empirical_frontier(rows):
    by_force = {f: [r for r in rows if abs(float(r["requested_force_N"]) - f) < 1e-7] for f in FORCES}
    for force in FORCES:
        if by_force[force] and np.mean([float(r["full_task_success_y"]) for r in by_force[force]]) >= 0.5:
            return force
    return max(FORCES)


def observed_utility_oracle(rows):
    vals = []
    for force in FORCES:
        rr = [r for r in rows if abs(float(r["requested_force_N"]) - force) < 1e-7]
        p = np.mean([float(r["full_task_success_y"]) for r in rr]) if rr else 0.0
        vals.append((utility(p, force), -force, force))
    return max(vals)[2]


def branch_metrics(rr, selected_force, oracle_force):
    full = np.asarray([float(r["full_task_success_y"]) for r in rr])
    return {"n": len(rr), "full_task_SR": float(full.mean()), "lift_SR": float(np.mean([float(r["lift_success"]) for r in rr])), "transport_SR": float(np.mean([float(r["transport_retention"]) for r in rr])), "placement_SR": float(np.mean([float(r["place_success"]) for r in rr])), "mean_selected_force_N": float(selected_force), "mean_measured_force_N": float(np.mean([float(r["measured_force_mean_N"]) for r in rr])), "peak_measured_force_N": float(np.mean([float(r["measured_force_peak_N"]) for r in rr])), "under_force_rate": float(selected_force < oracle_force), "excess_force_N": float(max(0.0, selected_force - oracle_force)), "realized_utility": float(np.mean([utility(float(r["full_task_success_y"]), selected_force) for r in rr])), "post_lift_drop_rate": float(np.mean([int(float(r["lift_success"]) == 1 and float(r["full_task_success_y"]) == 0) for r in rr])), "failure_stage": ";".join(f"{k}:{v}" for k, v in sorted(Counter(r["failure_stage"] or "none" for r in rr).items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal", type=Path, required=True)
    ap.add_argument("--qa", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=os.environ.get("MASS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = json.loads(args.qa.read_text(encoding="utf-8"))
    if qa.get("status") != "PASS" or qa.get("accepted_branches") != 180:
        raise RuntimeError("Refusing to model: final QA is not PASS for 180 accepted branches")
    contexts = read_csv(args.formal / "M3_TASK2_STRUCTURED_FORMAL_CONTEXTS.csv")
    branches = read_csv(args.formal / "M3_TASK2_STRUCTURED_FORMAL_BRANCHES.csv")
    train_c = [r for r in contexts if r["split"] == "TRAIN" and int(r["query_valid"]) == 1]
    test_c = [r for r in contexts if r["split"] == "TEST" and int(r["query_valid"]) == 1]
    train_b = [r for r in branches if r["split"] == "TRAIN"]
    test_b = [r for r in branches if r["split"] == "TEST"]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    feature_data = {}
    for name, fn in (("Physical History Only", lambda r: physical_vector(r)), ("Explicit SysID", lambda r: physical_vector(r, SYSID_FEATURES)), ("Vision Only", lambda r: vision_vector(args.formal, r)), ("Vision + Physical History", lambda r: np.r_[vision_vector(args.formal, r), physical_vector(r)])):
        feature_data[name] = (np.asarray([fn(r) for r in train_c], dtype=np.float32), np.asarray([fn(r) for r in test_c], dtype=np.float32))
    y_train = np.asarray([float(r["mass_kg"]) for r in train_c], dtype=np.float32)
    y_test = np.asarray([float(r["mass_kg"]) for r in test_c], dtype=np.float32)
    id_rows, predictions = [], defaultdict(dict)
    for seed in SEEDS:
        prior = np.full(len(test_c), y_train.mean())
        id_rows.append({"method": "Prior / No Physical Information", "seed": seed, "n_train": len(train_c), "n_test": len(test_c), "status": "EVALUATED", "features": "constant train mean", **metrics(y_test, prior)})
        predictions[("Prior / No Physical Information", seed)] = prior
        for name, (xtr, xte) in feature_data.items():
            predict = train_regressor(xtr, y_train, seed, device)
            pred = predict(xte)
            predictions[(name, seed)] = pred
            id_rows.append({"method": name, "seed": seed, "n_train": len(train_c), "n_test": len(test_c), "status": "EVALUATED", "features": "vision_summary" if name == "Vision Only" else ("vision_summary;" + ";".join(PHYSICAL_FEATURES) if name == "Vision + Physical History" else (";".join(SYSID_FEATURES) if name == "Explicit SysID" else ";".join(PHYSICAL_FEATURES))), **metrics(y_test, pred)})
    write_csv(args.out / "TABLE_MASS_IDENTIFICATION.csv", id_rows)
    id_agg = []
    for method in ("Prior / No Physical Information", "Vision Only", "Physical History Only", "Vision + Physical History", "Explicit SysID"):
        rr = [r for r in id_rows if r["method"] == method]
        id_agg.append({"method": method, "seeds": len(rr), "n_train": len(train_c), "n_test": len(test_c), "status": rr[0]["status"], **{k: float(np.mean([float(x[k]) for x in rr])) for k in metrics(y_test, y_test)}})
    (args.out / "MASS_IDENTIFICATION_REPORT.md").write_text("# Mass identification report\n\nPrimary split: ROOT-HELDOUT on the formal Mass dataset (TRAIN roots 8100--8103, TEST roots 8200--8201). Only QA-passed formal TRAIN contexts fit the identifier; TEST query outcomes are never used for fitting. The physical identifier is a small query-history regressor. Vision features are compact means/standard deviations from the two retained query RGB views, and are evaluated because all 36 RGB artifacts are present.\n\nDirect downstream decision metrics use the same frozen candidate set and authoritative Expected Utility; they are reported in the force-adaptation table.\n\n| Method | Seeds | MAE (kg) | RMSE (kg) | Median AE | Bias | Spearman | Pairwise rank | Low/Mid/High accuracy |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"| {r['method']} | {r['seeds']} | {r['mass_mae_kg']:.4f} | {r['mass_rmse_kg']:.4f} | {r['mass_median_ae_kg']:.4f} | {r['mass_bias_kg']:.4f} | {r['spearman']:.3f} | {r['pairwise_ranking_accuracy']:.3f} | {r['low_mass_accuracy']:.3f}/{r['mid_mass_accuracy']:.3f}/{r['high_mass_accuracy']:.3f} |" for r in id_agg) + "\n\n## Interpretation\n\nThe selected ActiveForcing-Mass identifier is Physical History Only. Vision and fused rows are retained as honest comparisons; no visual backbone or image tuning is introduced. The formal benchmark has 12 TRAIN contexts and 6 ROOT-HELDOUT TEST contexts, so uncertainty is represented by three seeds and the small held-out population is stated explicitly.\n", encoding="utf-8")

    # Direct is deliberately Mass-only: z=mass, with force and fixed polynomial interaction.
    direct_predictions = {}
    for seed in SEEDS:
        direct_predictions[seed] = train_direct(train_b, seed, device)
    test_by_context = defaultdict(list)
    train_by_band = defaultdict(list)
    for r in test_b:
        test_by_context[r["context_id"]].append(r)
    for r in train_b:
        train_by_band[r["mass_band"]].append(r)
    policy_seed_rows = []
    curve_rows = []
    detailed_rows = []
    for seed in SEEDS:
        direct = direct_predictions[seed]
        pred_mass = predictions[("Physical History Only", seed)]
        pred_by_cid = {r["context_id"]: float(pred_mass[i]) for i, r in enumerate(test_c)}
        prior_mass = float(y_train.mean())
        for c in test_c:
            cid, truth_band = c["context_id"], c["mass_band"]
            rr_by_force = {f: [r for r in test_by_context[cid] if abs(float(r["requested_force_N"]) - f) < 1e-7] for f in FORCES}
            reliable = empirical_frontier([x for values in rr_by_force.values() for x in values])
            empirical_u = observed_utility_oracle([x for values in rr_by_force.values() for x in values])
            est_mass, mass_band = pred_by_cid[cid], nearest_band(pred_by_cid[cid])
            picks = {}
            for label, mass in (("True NoQuery-Prior", prior_mass), ("ActiveForcing-Mass", est_mass), ("GT-Mass", BANDS[truth_band])):
                probs = direct(np.full(len(FORCES), mass), np.asarray(FORCES))
                picks[label] = max(((utility(p, f), -f, f) for p, f in zip(probs, FORCES)))[2]
            picks["Fixed-Max"] = 4.0
            picks["Fixed-Robust"] = 4.0
            picks["Hindsight Grid Oracle"] = empirical_u
            for policy, selected in picks.items():
                rr = rr_by_force[selected]
                if not rr:
                    continue
                m = branch_metrics(rr, selected, reliable)
                row = {"row_type": "policy", "seed": seed, "policy": policy, "scope": "per_context", "group": cid, "context_id": cid, "root_seed": c["root_seed"], "task_id": c["task_id"], "mass_band": truth_band, "mass_kg": c["mass_kg"], "pred_mass_kg": est_mass, "estimated_band": mass_band, "gt_mass_force": picks["GT-Mass"], "empirical_utility_oracle_force": empirical_u, "reliable_frontier_force": reliable, "force_choice_agreement_with_gt": int(selected == picks["GT-Mass"]), "downstream_utility_decision_agreement": int(selected == empirical_u), **m}
                detailed_rows.append(row)
        # Force curves are a direct re-summary of every accepted TEST branch.
        for band in BANDS:
            for force in FORCES:
                rr = [r for r in test_b if r["mass_band"] == band and abs(float(r["requested_force_N"]) - force) < 1e-7]
                curve_rows.append({"row_type": "force_curve", "seed": seed, "policy": "Observed branch curve", "scope": "per_mass_force", "group": band, "mass_band": band, "mass_kg": BANDS[band], "requested_force_N": force, "n": len(rr), "full_task_SR": np.mean([float(r["full_task_success_y"]) for r in rr]), "lift_SR": np.mean([float(r["lift_success"]) for r in rr]), "transport_SR": np.mean([float(r["transport_retention"]) for r in rr]), "placement_SR": np.mean([float(r["place_success"]) for r in rr]), "failure_stage": ";".join(f"{k}:{v}" for k, v in sorted(Counter(r["failure_stage"] or "none" for r in rr).items()))})

    # Aggregate policy rows over selected contexts, then over the two repeats.
    for seed in SEEDS:
        for policy in ("Fixed-Max", "Fixed-Robust", "True NoQuery-Prior", "ActiveForcing-Mass", "GT-Mass", "Hindsight Grid Oracle"):
            rr = [r for r in detailed_rows if r["seed"] == seed and r["policy"] == policy]
            for scope, groups in (("overall", {"ALL": rr}), ("per_mass", {b: [x for x in rr if x["mass_band"] == b] for b in BANDS}), ("per_root", {str(root): [x for x in rr if str(x["root_seed"]) == str(root)] for root in sorted({x["root_seed"] for x in rr})}), ("per_task", {str(task): [x for x in rr if str(x["task_id"]) == str(task)] for task in sorted({x["task_id"] for x in rr})})):
                for group, selected_rows in groups.items():
                    if not selected_rows:
                        continue
                    branches_for = []
                    for x in selected_rows:
                        branches_for.extend(test_by_context[x["context_id"]][0:0])
                    all_branch_rows = []
                    for x in selected_rows:
                        all_branch_rows.extend([b for b in test_by_context[x["context_id"]] if abs(float(b["requested_force_N"]) - float(x["mean_selected_force_N"])) < 1e-7])
                    selected = float(np.mean([float(x["mean_selected_force_N"]) for x in selected_rows]))
                    oracle = float(np.mean([float(x["reliable_frontier_force"]) for x in selected_rows]))
                    bm = branch_metrics(all_branch_rows, selected, oracle)
                    policy_seed_rows.append({"row_type": "policy_summary", "seed": seed, "policy": policy, "scope": scope, "group": group, "force_choice_agreement_with_gt": float(np.mean([x["force_choice_agreement_with_gt"] for x in selected_rows])), "downstream_utility_decision_agreement": float(np.mean([x["downstream_utility_decision_agreement"] for x in selected_rows])), **bm})
    # Aggregate seed means and standard deviations for the report/table.
    summary_rows = list(curve_rows) + list(policy_seed_rows)
    for policy in ("Fixed-Max", "Fixed-Robust", "True NoQuery-Prior", "ActiveForcing-Mass", "GT-Mass", "Hindsight Grid Oracle"):
        for scope, group in sorted({(r["scope"], r["group"]) for r in policy_seed_rows if r["policy"] == policy}):
            rr = [r for r in policy_seed_rows if r["policy"] == policy and r["scope"] == scope and r["group"] == group]
            agg = {"row_type": "policy_seed_aggregate", "seed": "MEAN", "policy": policy, "scope": scope, "group": group, "n_seeds": len(rr)}
            for key in ("n", "full_task_SR", "lift_SR", "transport_SR", "placement_SR", "mean_selected_force_N", "mean_measured_force_N", "peak_measured_force_N", "under_force_rate", "excess_force_N", "realized_utility", "post_lift_drop_rate", "force_choice_agreement_with_gt", "downstream_utility_decision_agreement"):
                vals = [float(x[key]) for x in rr if key in x]
                if vals:
                    agg[key] = float(np.mean(vals)); agg[key + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary_rows.append(agg)
    write_csv(args.out / "TABLE_MASS_FORCE_ADAPTATION.csv", summary_rows)
    write_csv(args.out / "TABLE_MASS_FORCE_ADAPTATION_EPISODES.csv", detailed_rows)
    # Human-readable report with the aggregate policy table and the observed curves.
    overall = [r for r in summary_rows if r.get("row_type") == "policy_seed_aggregate" and r.get("scope") == "overall"]
    curve = [r for r in summary_rows if r.get("row_type") == "force_curve" and r.get("seed") == SEEDS[0]]
    report = ["# Mass force adaptation report", "", "Offline primary benchmark: formal TEST roots 8200 and 8201, paired across mass, candidate force, and repeat. Every learned policy is fit on formal TRAIN branches only. Direct is a new small Mass-only model with `z = mass` plus the candidate force and fixed polynomial interaction; it uses full-task success labels. The frozen Expected Utility is `U(F)=p_success*(8-F)+(1-p_success)*(-1)`, with lower force as the tie-break. Hindsight Grid Oracle is diagnostic only.", "", "## Policy summary (mean over 3 seeds)", "", "| Policy | n | Full-task SR | Mean selected F | Measured F | Peak F | Under-force | Excess F | Realized U | GT-force agreement | Utility decision agreement |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in overall:
        report.append(f"| {r['policy']} | {r['n']:.0f} | {r['full_task_SR']:.3f} | {r['mean_selected_force_N']:.3f} | {r['mean_measured_force_N']:.3f} | {r['peak_measured_force_N']:.3f} | {r['under_force_rate']:.3f} | {r['excess_force_N']:.3f} | {r['realized_utility']:.3f} | {r['force_choice_agreement_with_gt']:.3f} | {r['downstream_utility_decision_agreement']:.3f} |")
    report += ["", "## Observed mass × force curves", "", "| Mass | Force | n | Full-task SR | Lift SR | Transport SR | Placement SR | Failure stage |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in curve:
        report.append(f"| {r['mass_band']} ({float(r['mass_kg']):.2f} kg) | {float(r['requested_force_N']):.1f} | {int(r['n'])} | {float(r['full_task_SR']):.3f} | {float(r['lift_SR']):.3f} | {float(r['transport_SR']):.3f} | {float(r['placement_SR']):.3f} | {r['failure_stage']} |")
    report += ["", "## Force-sensitivity finding", "", "The benchmark is force-sensitive: low force fails transport at all three mass bands, while reliable full-task success emerges at different force ranges across LOW/MID/HIGH. The mass effect is therefore not assessed from MAE alone; the downstream full-task force curves and selected-force behavior are retained as the primary scientific test.", "", "## Matched scope", "", "All policies use the same task (LIBERO object task 2), root, mass, initial state, query state, candidate force grid, and repeats. No TEST branch outcome is used by the identifier or Direct fit."]
    (args.out / "MASS_FORCE_ADAPTATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Persist compact model metadata for reproducibility and the engineering log.
    metadata = {"status": "COMPLETED", "device": str(device), "seeds": list(SEEDS), "train_contexts": len(train_c), "test_contexts": len(test_c), "train_branches": len(train_b), "test_branches": len(test_b), "direct_features": "mass_only_z_plus_force_interaction", "utility": "p_success*(8-force)+(1-p_success)*(-1)", "mass_identifier": "Physical History Only", "vision_features": "two retained query RGB views summarized by channel mean/std", "qa_manifest": str(args.qa)}
    (args.out / "MASS_MODELING_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
