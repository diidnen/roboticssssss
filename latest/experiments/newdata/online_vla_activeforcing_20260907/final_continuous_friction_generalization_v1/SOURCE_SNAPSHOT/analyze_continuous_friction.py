"""Offline-only analysis for the frozen continuous-friction campaign."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, spearmanr


METHODS = ("ACTIVEFORCING", "GT_PHYSICS", "FIXED_4")
METHOD_LABEL = {"ACTIVEFORCING": "ActiveForcing", "GT_PHYSICS": "GT-Physics", "FIXED_4": "Fixed-4"}
TASKS = (0, 1, 5, 6)


def read(path: Path):
    return json.loads(Path(path).read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_spearman(x, y):
    value = spearmanr(x, y)
    rho = float(value.statistic)
    p = float(value.pvalue)
    return {"rho": None if not np.isfinite(rho) else rho, "p": None if not np.isfinite(p) else p}


def positive_mixture_logpdf(x, means, log_sigmas):
    means = np.asarray(means, float)
    sigmas = np.exp(np.asarray(log_sigmas, float))
    weights = np.full(len(means), 1.0 / len(means))
    retained = logsumexp(np.log(weights) + norm.logsf(-means / sigmas))
    if x <= 0:
        return -math.inf
    return float(logsumexp(np.log(weights) + norm.logpdf(x, means, sigmas)) - retained)


def branch_squeeze(trace):
    values = []
    for row in trace:
        if int(row["branch_step"]) >= 1 and not row["vla_release_intent"]:
            left, right = row["normal_force_N"]
            values.append(2.0 * min(abs(float(left)), abs(float(right))))
    if not values:
        return None
    return float(np.mean(values))


def metric(rows, key):
    values = [float(row[key]) for row in rows if row[key] is not None]
    return None if not values else float(np.mean(values))


def paired_counts(left, right):
    pairs = []
    for context_id in sorted(set(left) & set(right)):
        a, b = left[context_id], right[context_id]
        pairs.append((a, b))
    return {
        "contexts": len(pairs),
        "both_success": sum(a["full_task_success"] and b["full_task_success"] for a, b in pairs),
        "left_only": sum(a["full_task_success"] and not b["full_task_success"] for a, b in pairs),
        "right_only": sum(not a["full_task_success"] and b["full_task_success"] for a, b in pairs),
        "both_fail": sum(not a["full_task_success"] and not b["full_task_success"] for a, b in pairs),
        "mean_selected_force_difference_left_minus_right": float(np.mean([a["selected_force"] - b["selected_force"] for a, b in pairs])),
        "mean_measured_squeeze_difference_left_minus_right": float(np.mean([a["measured_squeeze"] - b["measured_squeeze"] for a, b in pairs])),
    }


def load_rows(out: Path):
    complete = read(out / "CONTINUOUS_FRICTION_EXECUTION_COMPLETE.json")
    if complete.get("valid_full_task_branches") != 144 or complete.get("physics_stopped") is not True:
        raise RuntimeError("Exactly 144 valid branches must complete before offline analysis")
    rows = []
    for item in complete["rows"]:
        job = Path(item["job"])
        admission = read(job / "FINAL_EVIDENCE_ADMISSION.json")
        if admission.get("admitted") is not True:
            raise RuntimeError(f"Unadmitted branch in final set: {job}")
        result = read(job / "BRANCH_RESULT.json")
        outcome = result["outcome"]
        decision = read(job / "PLANNER_DECISION.json")
        posterior = read(job / "PREACTION_POSTERIOR.json")
        trace = read(job / "BRANCH_TRACE.json")
        intervention = read(job / "FRICTION_INTERVENTION_READBACK.json")
        squeeze = branch_squeeze(trace)
        if squeeze is None:
            raise RuntimeError(f"Empty primary squeeze window: {job}")
        plan = result["plan"]
        rows.append({
            "execution_index": int(item["execution_index"]),
            "context_id": item["context_id"],
            "root": int(plan["root"]),
            "task": int(plan["task"]),
            "mu_test": float(plan["mu"]),
            "mu_index": int(plan["mu_index"]),
            "interpolation_segment": plan["interpolation_segment"],
            "interpolation_fraction": float(plan["interpolation_fraction"]),
            "method": decision["method"],
            "full_task_success": int(outcome["full_task_success_y"]),
            "lift_success": int(outcome["lift_success"]),
            "drop": int(outcome["dropped"]),
            "failure_reasons": "|".join(outcome["failure_reasons"]),
            "selected_force": float(decision["executed_force_N"]),
            "model_selected_force": float(decision["model_selected_force_N"]),
            "predicted_success_at_model_selection": float(decision["predicted_success"]),
            "expected_utility_at_model_selection": float(max(decision["expected_utility"])),
            "measured_squeeze": squeeze,
            "posterior_mean": float(posterior["posterior_moments"]["mean"]),
            "posterior_std": float(posterior["posterior_moments"]["std"]),
            "posterior_median": float(posterior["posterior_quantiles"]["0.5"]),
            "posterior_interval_68_low": float(posterior["interval_68"][0]),
            "posterior_interval_68_high": float(posterior["interval_68"][1]),
            "posterior_interval_90_low": float(posterior["interval_90"][0]),
            "posterior_interval_90_high": float(posterior["interval_90"][1]),
            "posterior_interval_95_low": float(posterior["interval_95"][0]),
            "posterior_interval_95_high": float(posterior["interval_95"][1]),
            "member_means": json.dumps(posterior["member_means"], separators=(",", ":")),
            "member_sigmas": json.dumps(posterior["member_sigmas"], separators=(",", ":")),
            "request_count": int(result["rpc_count"]),
            "branch_steps": len(trace),
            "request_ids": json.dumps(sorted({x["request_id"] for x in trace}), separators=(",", ":")),
            "observation_hashes": json.dumps([sha(p) for p in sorted((job / "RPC").glob("*.npz"))], separators=(",", ":")),
            "action_trace_sha256": sha(job / "ACTION_TRACE.jsonl"),
            "branch_trace_sha256": sha(job / "BRANCH_TRACE.json"),
            "actual_material_readback": json.dumps(intervention["object_material_properties_static_dynamic_restitution"], separators=(",", ":")),
            "DOWNSTREAM_ACTION_SOURCE": admission["online_provenance"]["DOWNSTREAM_ACTION_SOURCE"],
            "VLA_CHECKPOINT_LOADED": admission["online_provenance"]["VLA_CHECKPOINT_LOADED"],
            "ONLINE_POLICY_INFERENCE": admission["online_provenance"]["ONLINE_POLICY_INFERENCE"],
            "VLA_ACTION_PROVENANCE_VERIFIED": admission["online_provenance"]["VLA_ACTION_PROVENANCE_VERIFIED"],
            "job": str(job),
        })
    if len(rows) != 144 or {row["method"] for row in rows} != set(METHODS):
        raise RuntimeError("Final trace rows are incomplete")
    return sorted(rows, key=lambda row: row["execution_index"])


def main(out: Path):
    out = Path(out)
    rows = load_rows(out)
    write_csv(out / "TABLE_CONTINUOUS_FRICTION_FULL_TRACE.csv", rows)
    by_method = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    end_to_end = []
    for method in METHODS:
        values = by_method[method]
        end_to_end.append({
            "method": METHOD_LABEL[method], "n": len(values),
            "full_task_success_count": sum(x["full_task_success"] for x in values),
            "FULL_TASK_SR": metric(values, "full_task_success"),
            "lift_success_count": sum(x["lift_success"] for x in values),
            "LIFT_SR": metric(values, "lift_success"),
            "drop_count": sum(x["drop"] for x in values),
            "DROP_RATE": metric(values, "drop"),
            "MEAN_SELECTED_FORCE": metric(values, "selected_force"),
            "MEAN_MEASURED_BILATERAL_SQUEEZE": metric(values, "measured_squeeze"),
        })
    write_csv(out / "TABLE_CONTINUOUS_FRICTION_END_TO_END.csv", end_to_end)

    af = by_method["ACTIVEFORCING"]
    belief_rows = []
    for row in af:
        error = row["posterior_mean"] - row["mu_test"]
        logpdf = positive_mixture_logpdf(row["mu_test"], json.loads(row["member_means"]),
                                         [math.log(x) for x in json.loads(row["member_sigmas"])])
        belief_rows.append({
            "context_id": row["context_id"], "root": row["root"], "task": row["task"],
            "mu_index": row["mu_index"], "mu_test": row["mu_test"],
            "interpolation_segment": row["interpolation_segment"],
            "interpolation_fraction": row["interpolation_fraction"],
            "posterior_mean": row["posterior_mean"], "posterior_median": row["posterior_median"],
            "posterior_std": row["posterior_std"], "error": error, "absolute_error": abs(error),
            "gaussian_mixture_nll": -logpdf,
            "member_means": row["member_means"], "member_sigmas": row["member_sigmas"],
            "interval_68_low": row["posterior_interval_68_low"], "interval_68_high": row["posterior_interval_68_high"],
            "interval_90_low": row["posterior_interval_90_low"], "interval_90_high": row["posterior_interval_90_high"],
            "interval_95_low": row["posterior_interval_95_low"], "interval_95_high": row["posterior_interval_95_high"],
            "covered_68": int(row["posterior_interval_68_low"] <= row["mu_test"] <= row["posterior_interval_68_high"]),
            "covered_90": int(row["posterior_interval_90_low"] <= row["mu_test"] <= row["posterior_interval_90_high"]),
            "covered_95": int(row["posterior_interval_95_low"] <= row["mu_test"] <= row["posterior_interval_95_high"]),
        })
    write_csv(out / "TABLE_CONTINUOUS_FRICTION_BELIEF.csv", belief_rows)
    overall_spearman = safe_spearman([x["mu_test"] for x in belief_rows], [x["posterior_mean"] for x in belief_rows])
    belief_summary = {
        "n": 48,
        "BELIEF_UNSEEN_MAE": float(np.mean([x["absolute_error"] for x in belief_rows])),
        "BELIEF_UNSEEN_RMSE": float(np.sqrt(np.mean([x["error"] ** 2 for x in belief_rows]))),
        "BELIEF_UNSEEN_BIAS": float(np.mean([x["error"] for x in belief_rows])),
        "BELIEF_UNSEEN_SPEARMAN": overall_spearman,
        "BELIEF_UNSEEN_GAUSSIAN_NLL": float(np.mean([x["gaussian_mixture_nll"] for x in belief_rows])),
        "coverage_68": metric(belief_rows, "covered_68"),
        "coverage_90": metric(belief_rows, "covered_90"),
        "coverage_95": metric(belief_rows, "covered_95"),
    }

    joined = defaultdict(dict)
    for row in rows:
        joined[row["context_id"]][row["method"]] = row
    continuity = []
    for context_id, methods in sorted(joined.items(), key=lambda x: (x[1]["ACTIVEFORCING"]["task"], x[1]["ACTIVEFORCING"]["mu_test"], x[1]["ACTIVEFORCING"]["root"])):
        a, g = methods["ACTIVEFORCING"], methods["GT_PHYSICS"]
        continuity.append({
            "TASK": a["task"], "ROOT": a["root"], "MU": a["mu_test"],
            "POSTERIOR_MEAN": a["posterior_mean"], "POSTERIOR_STD": a["posterior_std"],
            "F_AF": a["selected_force"], "F_GT": g["selected_force"],
            "AF_SUCCESS": a["full_task_success"], "GT_SUCCESS": g["full_task_success"],
        })
    write_csv(out / "TABLE_CONTINUOUS_FRICTION_CONTINUITY.csv", continuity)

    per_task = []
    ranking = []
    force_to_gt = []
    for task in TASKS:
        b = [x for x in belief_rows if x["task"] == task]
        b_spear = safe_spearman([x["mu_test"] for x in b], [x["posterior_mean"] for x in b])
        aggregated = []
        for mu_index in range(1, 7):
            group = [x for x in b if x["mu_index"] == mu_index]
            aggregated.append((float(np.mean([x["mu_test"] for x in group])), float(np.mean([x["posterior_mean"] for x in group]))))
        concordance = []
        for i in range(len(aggregated)):
            for j in range(i + 1, len(aggregated)):
                truth = np.sign(aggregated[j][0] - aggregated[i][0])
                pred = np.sign(aggregated[j][1] - aggregated[i][1])
                concordance.append(1.0 if truth == pred else (0.5 if pred == 0 else 0.0))
        rank_spear = safe_spearman([x[0] for x in aggregated], [x[1] for x in aggregated])
        ranking.append({
            "task": task, "NUM_UNIQUE_POSTERIOR_MEANS": len(set(x[1] for x in aggregated)),
            "POSTERIOR_MEAN_RANGE": max(x[1] for x in aggregated) - min(x[1] for x in aggregated),
            "WITHIN_TASK_RANKING_ACCURACY": float(np.mean(concordance)),
            "WITHIN_TASK_SPEARMAN": rank_spear["rho"], "WITHIN_TASK_SPEARMAN_P": rank_spear["p"],
        })
        task_cont = [x for x in continuity if x["TASK"] == task]
        force_spear = safe_spearman([x["MU"] for x in task_cont], [x["F_AF"] for x in task_cont])
        diffs = [abs(x["F_AF"] - x["F_GT"]) for x in task_cont]
        force_to_gt.extend(diffs)
        for method in METHODS:
            values = [x for x in by_method[method] if x["task"] == task]
            per_task.append({
                "task": task, "method": METHOD_LABEL[method], "n": len(values),
                "full_success_count": sum(x["full_task_success"] for x in values),
                "FULL_TASK_SR": metric(values, "full_task_success"),
                "LIFT_SR": metric(values, "lift_success"), "DROP_RATE": metric(values, "drop"),
                "MEAN_SELECTED_FORCE": metric(values, "selected_force"),
                "MEAN_MEASURED_BILATERAL_SQUEEZE": metric(values, "measured_squeeze"),
                "BELIEF_MAE": (float(np.mean([x["absolute_error"] for x in b])) if method == "ACTIVEFORCING" else None),
                "BELIEF_BIAS": (float(np.mean([x["error"] for x in b])) if method == "ACTIVEFORCING" else None),
                "BELIEF_SPEARMAN": (b_spear["rho"] if method == "ACTIVEFORCING" else None),
                "MU_VS_AF_FORCE_SPEARMAN": (force_spear["rho"] if method == "ACTIVEFORCING" else None),
            })
    write_csv(out / "TABLE_CONTINUOUS_FRICTION_PER_TASK.csv", per_task)
    write_csv(out / "TABLE_CONTINUOUS_FRICTION_RANKING.csv", ranking)

    maps = {method: {x["context_id"]: x for x in by_method[method]} for method in METHODS}
    paired_rows = []
    comparisons = (("AF_VS_GT", "ACTIVEFORCING", "GT_PHYSICS"),
                   ("AF_VS_FIXED4", "ACTIVEFORCING", "FIXED_4"))
    strata = [("ALL", lambda x: True),
              ("LOWER_INTERPOLATION_HALF", lambda x: x["interpolation_segment"] == "LOW_MID"),
              ("UPPER_INTERPOLATION_HALF", lambda x: x["interpolation_segment"] == "MID_HIGH")]
    strata += [(f"TASK_{task}", lambda x, task=task: x["task"] == task) for task in TASKS]
    for comparison, left_method, right_method in comparisons:
        for stratum, include in strata:
            left = {k: v for k, v in maps[left_method].items() if include(v)}
            right = {k: v for k, v in maps[right_method].items() if include(v)}
            value = paired_counts(left, right)
            paired_rows.append({"comparison": comparison, "stratum": stratum,
                                "left_method": METHOD_LABEL[left_method], "right_method": METHOD_LABEL[right_method], **value})
    write_csv(out / "TABLE_CONTINUOUS_FRICTION_PAIRED.csv", paired_rows)

    af_gt = next(x for x in paired_rows if x["comparison"] == "AF_VS_GT" and x["stratum"] == "ALL")
    af_fixed = next(x for x in paired_rows if x["comparison"] == "AF_VS_FIXED4" and x["stratum"] == "ALL")
    force_summary = {
        "MAE_FORCE_TO_GT": float(np.mean(force_to_gt)),
        "MEDIAN_FORCE_TO_GT": float(np.median(force_to_gt)),
        "fraction_within_0.10_N": float(np.mean(np.asarray(force_to_gt) <= 0.10 + 1e-12)),
        "fraction_within_0.25_N": float(np.mean(np.asarray(force_to_gt) <= 0.25 + 1e-12)),
        "fraction_within_0.50_N": float(np.mean(np.asarray(force_to_gt) <= 0.50 + 1e-12)),
    }

    failure_rows = []
    for a in af:
        if a["full_task_success"]:
            continue
        g = maps["GT_PHYSICS"][a["context_id"]]
        f = maps["FIXED_4"][a["context_id"]]
        reasons = set(a["failure_reasons"].split("|"))
        if a["lift_success"] and not a["drop"]:
            category = "POST_LIFT_GEOMETRIC"
        elif (a["drop"] or "NO_LIFT" in reasons) and a["selected_force"] + 0.05 < g["selected_force"]:
            category = "UNDER_FORCE"
        elif (g["full_task_success"] or f["full_task_success"]) and a["selected_force"] >= g["selected_force"] - 0.05:
            category = "VLA_EXECUTION_VARIANCE"
        else:
            category = "OTHER"
        failure_rows.append({
            "context_id": a["context_id"], "root": a["root"], "task": a["task"], "mu_test": a["mu_test"],
            "category": category, "failure_reasons": a["failure_reasons"],
            "posterior_error": a["posterior_mean"] - a["mu_test"],
            "F_AF": a["selected_force"], "F_GT": g["selected_force"], "F_FIXED4": f["selected_force"],
            "GT_success": g["full_task_success"], "Fixed4_success": f["full_task_success"],
            "model_probability": a["predicted_success_at_model_selection"],
            "utility_choice": a["expected_utility_at_model_selection"],
        })
    write_csv(out / "TABLE_CONTINUOUS_FRICTION_FAILURES.csv", failure_rows)

    root_results = []
    for root in sorted({x["root"] for x in rows}):
        for method in METHODS:
            values = [x for x in by_method[method] if x["root"] == root]
            root_results.append({"root": root, "method": METHOD_LABEL[method], "n": len(values),
                                 "full_success_count": sum(x["full_task_success"] for x in values),
                                 "FULL_TASK_SR": metric(values, "full_task_success"),
                                 "MEAN_SELECTED_FORCE": metric(values, "selected_force"),
                                 "MEAN_MEASURED_BILATERAL_SQUEEZE": metric(values, "measured_squeeze")})
    write_csv(out / "TABLE_CONTINUOUS_FRICTION_PER_ROOT.csv", root_results)

    anchors = read(out / "FINAL_UNSEEN_FRICTION_PLAN.json")
    anchor_map = {int(x["task"]): [x["low_anchor"], x["mid_anchor"], x["high_anchor"]] for x in anchors["tasks"]}
    colors = {0: "#2563eb", 1: "#d97706", 5: "#be185d", 6: "#4d7c0f"}
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), sharex=False, sharey=False)
    for ax, task in zip(axes.ravel(), TASKS):
        values = [x for x in belief_rows if x["task"] == task]
        x = np.asarray([v["mu_test"] for v in values])
        y = np.asarray([v["posterior_mean"] for v in values])
        err_low = y - np.asarray([v["interval_68_low"] for v in values])
        err_high = np.asarray([v["interval_68_high"] for v in values]) - y
        ax.errorbar(x, y, yerr=np.vstack([err_low, err_high]), fmt="o", ms=4, color=colors[task],
                    ecolor=colors[task], alpha=0.75, capsize=2, label="posterior mean ±68%")
        lo = min(min(x), min(y)); hi = max(max(x), max(y))
        ax.plot([lo, hi], [lo, hi], color="#374151", linestyle="--", linewidth=1, label="y=x")
        for anchor in anchor_map[task]:
            ax.axvline(anchor, color="#9ca3af", linestyle=":", linewidth=0.9)
        ax.set_title(f"Task {task}")
        ax.set_xlabel("True unseen object-side μ")
        ax.set_ylabel("Posterior mean")
        ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Continuous-friction belief on 48 unseen contexts", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "FIGURE_CONTINUOUS_FRICTION_BELIEF.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), sharey=True)
    styles = {"ACTIVEFORCING": ("#2563eb", "o", "-"), "GT_PHYSICS": ("#d97706", "s", "--")}
    for ax, task in zip(axes.ravel(), TASKS):
        for method in ("ACTIVEFORCING", "GT_PHYSICS"):
            series = [x for x in by_method[method] if x["task"] == task]
            grouped = []
            for mu in sorted({x["mu_test"] for x in series}):
                vals = [x["selected_force"] for x in series if x["mu_test"] == mu]
                grouped.append((mu, float(np.mean(vals)), min(vals), max(vals)))
            color, marker, line = styles[method]
            ax.plot([x[0] for x in grouped], [x[1] for x in grouped], line, color=color, marker=marker,
                    linewidth=1.6, markersize=4, label=METHOD_LABEL[method])
            ax.fill_between([x[0] for x in grouped], [x[2] for x in grouped], [x[3] for x in grouped],
                            color=color, alpha=0.10)
        ax.axhline(4.0, color="#374151", linestyle=":", linewidth=1.4, label="Fixed-4")
        ax.set_title(f"Task {task}")
        ax.set_xlabel("True unseen object-side μ")
        ax.set_ylabel("Selected force (N)")
        ax.set_ylim(2.9, 5.1)
        ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Force selection on unseen continuous friction", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "FIGURE_CONTINUOUS_FRICTION_FORCE.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    pareto_colors = {"ACTIVEFORCING": "#2563eb", "GT_PHYSICS": "#d97706", "FIXED_4": "#374151"}
    for item in end_to_end:
        method = next(k for k, v in METHOD_LABEL.items() if v == item["method"])
        ax.scatter(item["MEAN_MEASURED_BILATERAL_SQUEEZE"], 100 * item["FULL_TASK_SR"], s=70,
                   color=pareto_colors[method], label=item["method"])
        ax.annotate(item["method"], (item["MEAN_MEASURED_BILATERAL_SQUEEZE"], 100 * item["FULL_TASK_SR"]),
                    xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel("Mean measured bilateral squeeze (N)")
    ax.set_ylabel("Full-task success (%)")
    ax.set_title("Unseen continuous-friction operating points (48 contexts)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "FIGURE_CONTINUOUS_FRICTION_PARETO.pdf", bbox_inches="tight")
    plt.close(fig)

    task_force_spearman = {str(task): next(x["MU_VS_AF_FORCE_SPEARMAN"] for x in per_task
                                           if x["task"] == task and x["method"] == "ActiveForcing") for task in TASKS}
    positive_belief_tasks = sum((x["WITHIN_TASK_SPEARMAN"] or 0) > 0 for x in ranking)
    negative_force_tasks = sum((task_force_spearman[str(task)] or 0) < 0 for task in TASKS)
    unique_all = all(x["NUM_UNIQUE_POSTERIOR_MEANS"] >= 5 for x in ranking)
    belief_claim = "SUPPORTED" if overall_spearman["rho"] is not None and overall_spearman["rho"] >= 0.5 and positive_belief_tasks == 4 and unique_all else ("MIXED" if positive_belief_tasks >= 2 else "NOT_SUPPORTED")
    interpolation_claim = belief_claim
    force_claim = "SUPPORTED" if negative_force_tasks >= 3 else ("MIXED" if negative_force_tasks >= 2 else "NOT_SUPPORTED")
    gt_success = sum(x["full_task_success"] for x in by_method["GT_PHYSICS"])
    af_success = sum(x["full_task_success"] for x in af)
    gt_claim = "SUPPORTED" if af_gt["left_only"] == af_gt["right_only"] == 0 and force_summary["MAE_FORCE_TO_GT"] <= 0.10 else ("MIXED" if abs(af_success - gt_success) <= 4 and force_summary["MAE_FORCE_TO_GT"] <= 0.50 else "NOT_SUPPORTED")
    fixed_success = sum(x["full_task_success"] for x in by_method["FIXED_4"])
    af_squeeze = metric(af, "measured_squeeze")
    fixed_squeeze = metric(by_method["FIXED_4"], "measured_squeeze")
    fixed_claim = "SUPPORTED" if af_success > fixed_success and af_squeeze < fixed_squeeze else ("MIXED" if (af_success > fixed_success) != (af_squeeze < fixed_squeeze) else "NOT_SUPPORTED")
    main_paper = belief_claim == "SUPPORTED" and interpolation_claim == "SUPPORTED"
    claims = {
        "CLAIM_CONTINUOUS_PHYSICAL_BELIEF": belief_claim,
        "CLAIM_UNSEEN_FRICTION_INTERPOLATION": interpolation_claim,
        "CLAIM_CONTINUOUS_FORCE_ADAPTATION": force_claim,
        "CLAIM_AF_APPROACHES_GT_PHYSICS": gt_claim,
        "CLAIM_AF_OUTPERFORMS_FIXED4_ON_UNSEEN_FRICTION": fixed_claim,
        "MAIN_PAPER_WORTHY": "YES" if main_paper else "NO",
        "decision_rules": {
            "belief": "SUPPORTED requires pooled Spearman>=0.5, positive within-task ordering in all four tasks, and >=5 unique aggregated means per task.",
            "force": "SUPPORTED requires negative mu-vs-AF-force Spearman in at least three of four tasks; MIXED requires two.",
            "AF_approaches_GT": "SUPPORTED requires identical paired outcomes and force MAE<=0.10N; MIXED permits <=4/48 success-count gap and force MAE<=0.50N.",
            "AF_outperforms_Fixed4": "SUPPORTED requires both more successes and lower mean measured squeeze; MIXED when only one improves.",
        },
    }
    write_json(out / "CONTINUOUS_FRICTION_CLAIM_AUDIT.json", claims)

    end_map = {x["method"]: x for x in end_to_end}
    failure_counts = {category: sum(x["category"] == category for x in failure_rows)
                      for category in ("UNDER_FORCE", "POST_LIFT_GEOMETRIC", "VLA_EXECUTION_VARIANCE", "OTHER")}
    support = read(out / "TRAINING_FRICTION_SUPPORT_AUDIT.json")
    unseen_values = {str(x["task"]): [y["mu"] for y in x["unseen_values"]] for x in anchors["tasks"]}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "belief": belief_summary, "ranking": ranking, "force_to_gt": force_summary,
        "task_force_spearman": task_force_spearman, "end_to_end": end_to_end,
        "AF_vs_GT_paired": af_gt, "AF_vs_Fixed4_paired": af_fixed,
        "failure_counts": failure_counts, "claims": claims,
    }
    write_json(out / "FINAL_CONTINUOUS_FRICTION_RESULTS.json", summary)

    def pct(value): return f"{100 * value:.1f}%"
    af_end, gt_end, fx_end = end_map["ActiveForcing"], end_map["GT-Physics"], end_map["Fixed-4"]
    report = f"""# Final Unseen Continuous-Friction Generalization

Technical summary: the frozen experiment completed 48 predeclared unseen-friction contexts and 144 valid online-VLA branches with no retraining or tuning. The actual training audit shows that both frozen models had already seen root-specific continuous draws—9 μ values per task for belief and 12 per task for feasibility—so this study tests interpolation to exact values absent from finite training sets, not training from only three fixed anchors. Belief interpolation is classified **{belief_claim}**, force adaptation **{force_claim}**, descriptive proximity to GT-Physics **{gt_claim}**, and superiority to Fixed-4 **{fixed_claim}**.

# 1. Training Friction Support Audit

The final 58D belief TRAIN split used 9 distinct object-side μ values per task on roots 5101/5104/5109. The frozen feasibility TRAIN split used 12 distinct μ values per task on roots 5100/5101/5102/5106. Both were generated by continuous uniform draws within LOW/MID/HIGH intervals. Therefore `BELIEF_CONTINUOUSLY_SAMPLED_DURING_TRAINING=YES` and `FEASIBILITY_CONTINUOUSLY_SAMPLED_DURING_TRAINING=YES`; describing either model as trained only at three fixed friction points would be false.

# 2. Frozen Unseen-Friction Test Values

For every task, six values were deterministically fixed at fractions 0.25/0.50/0.75 of the canonical L–M and M–H intervals. Exact membership checks against both frozen training supports and all canonical evaluation anchors passed before physics. Roots 170052 and 170053 were selected outcome-blind and collision-audited before execution. The intervention changed only object static/dynamic material friction to μ_test; no effective contact-pair coefficient is claimed.

# 3. Belief Generalization

Across 48 contexts, MAE={belief_summary['BELIEF_UNSEEN_MAE']:.4f}, RMSE={belief_summary['BELIEF_UNSEEN_RMSE']:.4f}, bias={belief_summary['BELIEF_UNSEEN_BIAS']:+.4f}, Spearman ρ={belief_summary['BELIEF_UNSEEN_SPEARMAN']['rho']:.3f} (p={belief_summary['BELIEF_UNSEEN_SPEARMAN']['p']:.3g}), and positive-mixture NLL={belief_summary['BELIEF_UNSEEN_GAUSSIAN_NLL']:.4f}. Coverage is 68%={pct(belief_summary['coverage_68'])}, 90%={pct(belief_summary['coverage_90'])}, and 95%={pct(belief_summary['coverage_95'])}.

# 4. Continuous Interpolation Analysis

The primary analysis keeps all six ordered intermediate μ values continuous. Per-task unique posterior means, range, pairwise ranking accuracy, and Spearman are in `TABLE_CONTINUOUS_FRICTION_RANKING.csv`; the full root-resolved sequence is in `TABLE_CONTINUOUS_FRICTION_CONTINUITY.csv`. This evidence is classified `{interpolation_claim}` and should not be reframed as LOW/MID/HIGH classification.

# 5. Force-Selection Generalization

AF-to-GT force MAE is {force_summary['MAE_FORCE_TO_GT']:.3f} N (median {force_summary['MEDIAN_FORCE_TO_GT']:.3f} N); {pct(force_summary['fraction_within_0.10_N'])}, {pct(force_summary['fraction_within_0.25_N'])}, and {pct(force_summary['fraction_within_0.50_N'])} fall within 0.10/0.25/0.50 N. Task-wise μ-versus-AF-force Spearman values are {task_force_spearman}. No sign was imposed during analysis.

# 6. ActiveForcing vs GT-Physics

Paired outcomes: both success={af_gt['both_success']}, AF only={af_gt['left_only']}, GT only={af_gt['right_only']}, both fail={af_gt['both_fail']}. AF-minus-GT selected force is {af_gt['mean_selected_force_difference_left_minus_right']:+.3f} N and measured squeeze is {af_gt['mean_measured_squeeze_difference_left_minus_right']:+.3f} N. Without a predeclared non-inferiority margin and with only two roots, this is descriptive mechanistic evidence, not population-level equivalence.

# 7. ActiveForcing vs Fixed-4

Paired outcomes: both success={af_fixed['both_success']}, AF only={af_fixed['left_only']}, Fixed-4 only={af_fixed['right_only']}, both fail={af_fixed['both_fail']}. AF-minus-Fixed-4 measured squeeze is {af_fixed['mean_measured_squeeze_difference_left_minus_right']:+.3f} N. Lower/upper interpolation and task strata are reported without denominator pooling in `TABLE_CONTINUOUS_FRICTION_PAIRED.csv`.

# 8. End-to-End Results

ActiveForcing: {af_end['full_task_success_count']}/48 full success ({pct(af_end['FULL_TASK_SR'])}), lift {pct(af_end['LIFT_SR'])}, drop {pct(af_end['DROP_RATE'])}, mean selected {af_end['MEAN_SELECTED_FORCE']:.3f} N, mean measured squeeze {af_end['MEAN_MEASURED_BILATERAL_SQUEEZE']:.3f} N. GT-Physics: {gt_end['full_task_success_count']}/48 ({pct(gt_end['FULL_TASK_SR'])}), selected {gt_end['MEAN_SELECTED_FORCE']:.3f} N, measured {gt_end['MEAN_MEASURED_BILATERAL_SQUEEZE']:.3f} N. Fixed-4: {fx_end['full_task_success_count']}/48 ({pct(fx_end['FULL_TASK_SR'])}), measured {fx_end['MEAN_MEASURED_BILATERAL_SQUEEZE']:.3f} N.

The primary squeeze is recomputed branch-wise over `branch_step>=1 AND vla_release_intent=false` as 2×min(|N_L|,|N_R|), retaining zero-contact frames, then averaged equally across branches.

# 9. Per-Task Results

Exact 12-context task/method results are in `TABLE_CONTINUOUS_FRICTION_PER_TASK.csv`; two 24-context root-group readouts are in `TABLE_CONTINUOUS_FRICTION_PER_ROOT.csv`. Root is the only statistical cluster, so no population-level significance claim is made.

# 10. Failure Analysis

AF failures are classified by a frozen post-hoc diagnostic rule: {failure_counts}. `UNDER_FORCE` requires no-lift/drop plus AF at least 0.05 N below GT; successful comparator branches at comparable force are labeled `VLA_EXECUTION_VARIANCE`; successful-lift non-drop terminal failures are `POST_LIFT_GEOMETRIC`; remaining cases are `OTHER`. These categories are diagnostic, not causal proof.

# 11. What This Experiment Proves

It directly measures how the frozen probe→58D belief→positive-support posterior→phase-free feasibility→posterior marginalization→0.05 N dense search→online VLA chain behaves at 24 exact object-side μ values absent from both training datasets, replicated on two untouched roots. It can support continuous interpolation claims only at the strength recorded in the claim audit.

# 12. What It Does NOT Prove

It does not show learning from only three friction anchors, because training used continuous within-band draws. It does not test extrapolation, new tasks, new objects, hardware transfer, the resolved pair coefficient, population-level significance, or statistical non-inferiority to GT/Fix-4. Two roots limit generality and VLA stochasticity remains entangled with force-dependent trajectories.

# 13. Recommended Main-Paper Placement

{'Place the belief/force interpolation figure and compact AF/GT/Fixed-4 table in the main paper as a mechanistic generalization result; state explicitly that training used finite continuous within-band samples and that all test μ values were exact held-out values.' if main_paper else 'Keep the full result in the appendix or limitations section. The main paper may report it as a frozen held-out diagnostic, but should not headline continuous physical-identification generalization.'}
"""
    (out / "FINAL_CONTINUOUS_FRICTION_GENERALIZATION_REPORT.md").write_text(report)

    terminal = {
        "FINAL_CONTINUOUS_FRICTION_STATUS": "COMPLETE_144_VALID_BRANCHES_STOPPED",
        "BELIEF_CONTINUOUSLY_SAMPLED_DURING_TRAINING": "YES",
        "FEASIBILITY_CONTINUOUSLY_SAMPLED_DURING_TRAINING": "YES",
        "NUM_TRAIN_FRICTION_VALUES_PER_TASK": {"belief": 9, "feasibility": 12},
        "NUM_NEW_ROOTS": 2, "NUM_UNSEEN_MU_PER_TASK": 6, "NUM_NEW_CONTEXTS": 48,
        "NUM_NEW_VALID_BRANCHES": 144, "UNSEEN_MU_VALUES_PER_TASK": unseen_values,
        "BELIEF_UNSEEN_MAE": belief_summary["BELIEF_UNSEEN_MAE"],
        "BELIEF_UNSEEN_RMSE": belief_summary["BELIEF_UNSEEN_RMSE"],
        "BELIEF_UNSEEN_SPEARMAN": belief_summary["BELIEF_UNSEEN_SPEARMAN"],
        "BELIEF_UNSEEN_90_COVERAGE": belief_summary["coverage_90"],
        "AF_FULL_SR": af_end["FULL_TASK_SR"], "AF_MEAN_SELECTED_FORCE": af_end["MEAN_SELECTED_FORCE"],
        "AF_MEASURED_FORCE": af_end["MEAN_MEASURED_BILATERAL_SQUEEZE"],
        "GT_FULL_SR": gt_end["FULL_TASK_SR"], "GT_MEAN_SELECTED_FORCE": gt_end["MEAN_SELECTED_FORCE"],
        "GT_MEASURED_FORCE": gt_end["MEAN_MEASURED_BILATERAL_SQUEEZE"],
        "FIXED4_FULL_SR": fx_end["FULL_TASK_SR"], "FIXED4_MEASURED_FORCE": fx_end["MEAN_MEASURED_BILATERAL_SQUEEZE"],
        "AF_GT_FORCE_MAE": force_summary["MAE_FORCE_TO_GT"],
        "AF_VS_GT_PAIRED": af_gt, "AF_VS_FIXED4_PAIRED": af_fixed,
        "CONTINUOUS_PHYSICAL_BELIEF_CLAIM": belief_claim,
        "UNSEEN_FRICTION_GENERALIZATION_CLAIM": interpolation_claim,
        "CONTINUOUS_FORCE_ADAPTATION_CLAIM": force_claim,
        "MAIN_PAPER_WORTHY": "YES" if main_paper else "NO",
        "REMAINING_MUST_RUN_EXPERIMENTS": "NONE",
    }
    write_json(out / "FINAL_TERMINAL_OUTPUT.json", terminal)
    manifest = {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.name.startswith(("TABLE_", "FIGURE_", "FINAL_", "CONTINUOUS_FRICTION_CLAIM"))}
    write_json(out / "FINAL_ANALYSIS_ARTIFACT_SHA256.json", manifest)
    print(json.dumps(terminal, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.out)
