#!/usr/bin/env python3
"""Read-only diagnostic of feasibility data, model fit, and force-range censoring."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path: Path):
    return json.loads(path.read_text())


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def median(values):
    values = list(values)
    return statistics.median(values) if values else None


def safe_logloss(labels, probs):
    eps = 1e-12
    return mean(-(y * math.log(min(1 - eps, max(eps, p))) + (1 - y) * math.log(min(1 - eps, max(eps, 1 - p)))) for y, p in zip(labels, probs))


def auc(labels, probs):
    pos = [p for y, p in zip(labels, probs) if y == 1]
    neg = [p for y, p in zip(labels, probs) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def calibration_bins(labels, probs, n=5):
    bins = []
    for i in range(n):
        lo, hi = i / n, (i + 1) / n
        idx = [j for j, p in enumerate(probs) if lo <= p <= hi if (i == n - 1 or p < hi)]
        if idx:
            bins.append({
                "range": [lo, hi],
                "n": len(idx),
                "mean_predicted": mean(probs[j] for j in idx),
                "empirical_success": mean(labels[j] for j in idx),
            })
    return bins


def classification_metrics(rows):
    labels = [int(r["y"]) for r in rows]
    probs = [float(r["p"]) for r in rows]
    if not labels:
        return {}
    pos_rate = mean(labels)
    baseline_p = min(1 - 1e-12, max(1e-12, pos_rate))
    return {
        "n": len(labels),
        "positives": sum(labels),
        "positive_rate": pos_rate,
        "mean_predicted": mean(probs),
        "mean_predicted_positive": mean(p for y, p in zip(labels, probs) if y),
        "mean_predicted_negative": mean(p for y, p in zip(labels, probs) if not y),
        "auc": auc(labels, probs),
        "brier": mean((p - y) ** 2 for y, p in zip(labels, probs)),
        "logloss": safe_logloss(labels, probs),
        "constant_prevalence_logloss": safe_logloss(labels, [baseline_p] * len(labels)),
        "predicted_positive_at_0.5": sum(p >= 0.5 for p in probs),
        "calibration_bins": calibration_bins(labels, probs),
    }


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def collect_rows(dataset_root: Path, source: str):
    rows = []
    files = sorted(dataset_root.glob("groups/*/attempt_*/job/COLLECTED_ROWS.json"))
    for path in files:
        accepted = path.parents[2] / "ACCEPTED.json"
        if accepted.exists() and read(accepted).get("accepted") is False:
            continue
        for raw in read(path):
            rows.append({
                "source": source,
                "source_file": str(path),
                "context_id": raw["context_id"],
                "split": raw["split"],
                "friction": float(raw["true_mu_training_only"]),
                "force": float(raw["force"]),
                "y": int(raw["full_task_success_y"]),
                "feature_sha256": raw["feature_sha256"],
                "result_sha256": raw["result_sha256"],
            })
    return files, rows


def group_summary(rows):
    by_context = defaultdict(list)
    for row in rows:
        by_context[row["context_id"]].append(row)
    summaries = []
    for context_id, group in sorted(by_context.items()):
        group = sorted(group, key=lambda r: r["force"])
        ys = [r["y"] for r in group]
        successes = [r["force"] for r in group if r["y"]]
        y_by_force = {r["force"]: r["y"] for r in group}
        downward_transitions = sum(ys[i] > ys[i + 1] for i in range(len(ys) - 1))
        summaries.append({
            "context_id": context_id,
            "source": group[0]["source"],
            "split": group[0]["split"],
            "friction": group[0]["friction"],
            "feature_sha256": group[0]["feature_sha256"],
            "forces": [r["force"] for r in group],
            "labels": ys,
            "successes": sum(ys),
            "successful_forces": successes,
            "all_failed_right_censored_at_8N": not successes,
            "eight_N_success": bool(y_by_force.get(8.0, 0)),
            "eight_N_failure_with_lower_force_success": (not y_by_force.get(8.0, 0)) and bool(successes),
            "eight_N_only_success": successes == [8.0],
            "nonmonotone_downward_transition": downward_transitions > 0,
            "downward_transitions": downward_transitions,
        })
    return summaries


def prediction_at(decision, force):
    grid = decision["force_grid_N"]
    probs = decision["p_success"]
    idx = min(range(len(grid)), key=lambda i: abs(float(grid[i]) - force))
    if abs(float(grid[idx]) - force) > 1e-8:
        raise AssertionError((decision["id"], force, grid[idx]))
    return float(probs[idx])


def model_diagnostics(rows, decisions):
    decision_by_id = {d["id"]: d for d in decisions}
    scored = []
    unmatched = []
    for row in rows:
        decision = decision_by_id.get(row["context_id"])
        if decision is None:
            unmatched.append(row["context_id"])
            continue
        scored.append({**row, "p": prediction_at(decision, row["force"])})

    by_split = {}
    for split in sorted({r["split"] for r in scored}):
        by_split[split] = classification_metrics([r for r in scored if r["split"] == split])

    decision_rows = []
    for d in decisions:
        grid = [float(v) for v in d["force_grid_N"]]
        probs = [float(v) for v in d["p_success"]]
        max_idx = max(range(len(probs)), key=probs.__getitem__)
        p8 = prediction_at(d, 8.0)
        p75 = prediction_at(d, 7.5)
        decision_rows.append({
            "id": d["id"],
            "split": d["split"],
            "selected_force_N": float(d["selected_force_N"]),
            "selected_probability": float(d["predicted_success"]),
            "probability_peak_force_N": grid[max_idx],
            "probability_peak": probs[max_idx],
            "p_at_8N": p8,
            "p_at_7_5N": p75,
            "upper_edge_delta_p_8_minus_7_5": p8 - p75,
            "probability_still_increasing_at_upper_edge": p8 > p75,
        })
    return {
        "scored_rows": len(scored),
        "unmatched_contexts": sorted(set(unmatched)),
        "by_split": by_split,
        "decisions": decision_rows,
        "selected_force_distribution": dict(sorted(Counter(r["selected_force_N"] for r in decision_rows).items())),
        "selected_at_8N": sum(r["selected_force_N"] == 8.0 for r in decision_rows),
        "probability_peak_at_8N": sum(r["probability_peak_force_N"] == 8.0 for r in decision_rows),
        "probability_still_increasing_at_upper_edge": sum(r["probability_still_increasing_at_upper_edge"] for r in decision_rows),
    }


def formal_diagnostics(core, deep):
    compact = {}
    for record in core["records"]:
        context_id = record["context"]["id"]
        for outcome in record["outcomes"]:
            compact[(context_id, outcome["method"])] = outcome
    rows = []
    for context in deep["contexts"]:
        for branch in context["branch_summaries"]:
            key = (context["context_id"], branch["method"])
            outcome = compact[key]
            contact = float(branch["target_contact_fraction"])
            squeeze = float(branch["measured_mean_squeeze_N"])
            rows.append({
                "context_id": context["context_id"],
                "method": branch["method"],
                "success": int(branch["success"]),
                "commanded_force_N": outcome["commanded_force_N"],
                "measured_mean_squeeze_N": squeeze,
                "target_contact_fraction": contact,
                "squeeze_per_contact_fraction_N": squeeze / contact if contact > 0 else None,
                "sustained_contact_loss": bool(branch["sustained_contact_loss_50_steps"]),
            })
    by_method = {}
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        success = [r for r in group if r["success"]]
        fail = [r for r in group if not r["success"]]
        by_method[method] = {
            "rollouts": len(group),
            "successes": len(success),
            "mean_commanded_force_N": mean(r["commanded_force_N"] for r in group if r["commanded_force_N"] is not None),
            "mean_squeeze_all": mean(r["measured_mean_squeeze_N"] for r in group),
            "mean_squeeze_success": mean(r["measured_mean_squeeze_N"] for r in success),
            "mean_squeeze_failure": mean(r["measured_mean_squeeze_N"] for r in fail),
            "mean_contact_fraction_all": mean(r["target_contact_fraction"] for r in group),
            "mean_contact_fraction_success": mean(r["target_contact_fraction"] for r in success),
            "mean_contact_fraction_failure": mean(r["target_contact_fraction"] for r in fail),
            "mean_squeeze_per_contact_fraction_N": mean(r["squeeze_per_contact_fraction_N"] for r in group if r["squeeze_per_contact_fraction_N"] is not None),
            "squeeze_success_correlation": pearson([r["measured_mean_squeeze_N"] for r in group], [r["success"] for r in group]),
            "contact_fraction_success_correlation": pearson([r["target_contact_fraction"] for r in group], [r["success"] for r in group]),
            "sustained_contact_loss": sum(r["sustained_contact_loss"] for r in group),
        }
    return {"by_method": by_method, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original_files, original_rows = collect_rows(args.original, "reused_original")
    additional_files, additional_rows = collect_rows(args.additional, "new_full_group")
    rows = original_rows + additional_rows
    contexts = group_summary(rows)

    by_feature_force = defaultdict(list)
    by_result = defaultdict(list)
    for row in rows:
        by_feature_force[(row["feature_sha256"], row["force"])].append(row)
        by_result[row["result_sha256"]].append(row)
    feature_force_dupes = [v for v in by_feature_force.values() if len(v) > 1]
    split_leakage = []
    by_feature = defaultdict(list)
    for row in rows:
        by_feature[row["feature_sha256"]].append(row)
    for feature, group in by_feature.items():
        splits = sorted(set(r["split"] for r in group))
        if len(splits) > 1:
            split_leakage.append({"feature_sha256": feature, "splits": splits, "contexts": sorted(set(r["context_id"] for r in group))})

    label_by_force = {}
    for force in sorted(set(r["force"] for r in rows)):
        group = [r for r in rows if r["force"] == force]
        label_by_force[str(force)] = {"n": len(group), "successes": sum(r["y"] for r in group), "rate": mean(r["y"] for r in group)}
    label_by_split = {}
    for split in sorted(set(r["split"] for r in rows)):
        group = [r for r in rows if r["split"] == split]
        label_by_split[split] = {"n": len(group), "successes": sum(r["y"] for r in group), "rate": mean(r["y"] for r in group)}

    result = {
        "audit_version": "FORCE_RANGE_ROOT_CAUSE_DIAGNOSTIC_V1",
        "read_only": True,
        "dataset": {
            "row_count": len(rows),
            "context_count": len(contexts),
            "source_files": {"reused_original": len(original_files), "new_full_group": len(additional_files)},
            "required_fields_complete": all(all(row.get(k) is not None for k in ("context_id", "split", "friction", "force", "y", "feature_sha256", "result_sha256")) for row in rows),
            "duplicate_context_force_keys": len(rows) - len({(r["context_id"], r["force"]) for r in rows}),
            "exact_result_hash_duplicate_groups": sum(len(v) > 1 for v in by_result.values()),
            "unique_feature_hashes": len(by_feature),
            "feature_force_duplicate_groups": len(feature_force_dupes),
            "feature_force_duplicate_rows": sum(len(v) - 1 for v in feature_force_dupes),
            "feature_force_conflicting_label_groups": sum(len({r["y"] for r in v}) > 1 for v in feature_force_dupes),
            "train_val_feature_leakage": split_leakage,
            "labels_by_split": label_by_split,
            "labels_by_force": label_by_force,
            "contexts_all_failed_right_censored_at_8N": sum(c["all_failed_right_censored_at_8N"] for c in contexts),
            "contexts_with_any_success": sum(c["successes"] > 0 for c in contexts),
            "contexts_8N_success": sum(c["eight_N_success"] for c in contexts),
            "contexts_8N_only_success": sum(c["eight_N_only_success"] for c in contexts),
            "contexts_8N_failure_but_lower_success": sum(c["eight_N_failure_with_lower_force_success"] for c in contexts),
            "contexts_nonmonotone": sum(c["nonmonotone_downward_transition"] for c in contexts),
            "context_summaries": contexts,
        },
        "model": model_diagnostics(rows, read(args.decisions)),
        "formal": formal_diagnostics(read(args.core), read(args.deep)),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload)
    print(args.output)
    print(hashlib.sha256(payload.encode()).hexdigest())


if __name__ == "__main__":
    main()
