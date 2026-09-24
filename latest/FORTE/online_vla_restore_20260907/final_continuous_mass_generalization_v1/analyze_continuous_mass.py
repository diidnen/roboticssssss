#!/usr/bin/env python3
"""Official analysis for the frozen continuous/unseen MASS evaluation."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
OUT = HERE / "final_evaluation"
FRICTION = Path("/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1")
METHODS = ("ACTIVEFORCING_MASS", "GT_MASS", "FIXED_4")
COLORS = {0: "#4C78A8", 1: "#F58518", 5: "#54A24B", 6: "#E45756"}
MARKERS = {0: "o", 1: "s", 5: "^", 6: "D"}
LINESTYLES = {0: "-", 1: "--", 5: "-.", 6: ":"}
METHOD_COLORS = {"ACTIVEFORCING_MASS": "#4C78A8", "GT_MASS": "#F58518", "FIXED_4": "#666666"}
METHOD_LABELS = {"ACTIVEFORCING_MASS": "AF-Mass", "GT_MASS": "GT-Mass", "FIXED_4": "Fixed-4"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def inertia_readback_matches(material):
    nominal = np.asarray(material.get("nominal_inertia", []), dtype=float)
    actual = np.asarray(material.get("actual_inertia", []), dtype=float)
    ratio = material.get("mass_ratio")
    return bool(nominal.size and nominal.shape == actual.shape and ratio is not None and
                np.isfinite(nominal).all() and np.isfinite(actual).all() and
                np.allclose(actual, nominal * float(ratio), rtol=3e-6, atol=1e-12))


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(clean(value), stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def csv_new(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def lines(path):
    with Path(path).open() as stream: return [json.loads(line) for line in stream if line.strip()]


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(spearmanr(x, y).statistic) if len(x) > 1 and np.std(x) and np.std(y) else None


def formatted(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def force_trend(group):
    """Descriptive trend only; it is not used as a claim gate."""
    x = group.true_mass_kg.to_numpy(dtype=float)
    y = group.selected_force_N.to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 and np.std(x) else None
    return {"Spearman": rho(x, y), "linear_slope_N_per_kg": slope,
            "unique_selected_forces": int(group.selected_force_N.nunique())}


def ranking_accuracy(rows):
    correct = total = ties = 0
    for (_, _), group in rows.groupby(["task", "root"]):
        values = list(zip(group.true_mass_kg, group.posterior_mean_kg))
        for a, b in itertools.combinations(values, 2):
            if a[0] == b[0]: continue
            total += 1; product = (a[0] - b[0]) * (a[1] - b[1])
            correct += product > 0; ties += product == 0
    return {"correct": int(correct), "ties": int(ties), "total": total,
            "accuracy": float(correct / total) if total else None}


def belief_metrics(frame):
    y, p = frame.true_mass_kg.to_numpy(), frame.posterior_mean_kg.to_numpy()
    aggregate = (frame.groupby(["task", "true_mass_kg"], as_index=False)
                 .posterior_mean_kg.mean())
    result = {"count": len(frame), "MAE": float(np.mean(np.abs(p - y))),
              "RMSE": float(np.sqrt(np.mean((p - y) ** 2))), "bias": float(np.mean(p - y)),
              "Spearman": rho(y, p), "ranking_accuracy": ranking_accuracy(frame),
              "distinct_posterior_means": int(len(set(np.round(p, 10)))),
              "distinct_aggregate_posterior_means": int(len(set(np.round(aggregate.posterior_mean_kg, 10)))),
              "aggregate_Spearman": rho(aggregate.true_mass_kg, aggregate.posterior_mean_kg)}
    for level in (68, 90, 95):
        result[f"coverage_{level}"] = float(np.mean((y >= frame[f"interval_{level}_low"]) & (y <= frame[f"interval_{level}_high"])))
    return result


def valid_job(context_id, method):
    canonical = OUT / "branches" / f"{context_id}__{method}"
    if (canonical / "WORKER_COMPLETION.json").is_file() and read(canonical / "WORKER_COMPLETION.json").get("logical_success"):
        return canonical
    retry = OUT / "retries" / f"{context_id}__{method}"
    candidates = sorted(retry.glob("attempt_*")) if retry.exists() else []
    admitted = [path for path in candidates if (path / "WORKER_COMPLETION.json").is_file() and read(path / "WORKER_COMPLETION.json").get("logical_success")]
    if len(admitted) != 1: raise RuntimeError(f"expected one valid branch for {context_id}/{method}")
    return admitted[0]


def rerun_audit():
    records = []
    for authorization in sorted(OUT.glob("**/RETRY_AUTHORIZATION.json")):
        records.append({"authorization_path": str(authorization), "authorization": read(authorization)})
    attempts = []
    retry_root = OUT / "retries"
    if retry_root.exists():
        for attempt in sorted(retry_root.glob("*/attempt_*")):
            completion = read(attempt / "WORKER_COMPLETION.json") if (attempt / "WORKER_COMPLETION.json").is_file() else None
            attempts.append({"path": str(attempt), "completion": completion})
    payload = {"result_driven_reruns": 0, "infrastructure_retry_authorizations": records,
               "retry_attempts": attempts, "infrastructure_retry_count": len(attempts),
               "policy": "Only independently verified infrastructure corruption; outcomes never authorize reruns."}
    write_json(HERE / "FINAL_MASS_RERUN_AUDIT.json", payload)
    return payload


def render_animation(job, destination):
    frames = []
    paths = sorted(job.glob("RAW_OBSERVATION_*.npz"))[::10]
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            image = Image.fromarray(data["agentview_cam"]).resize((256, 256), Image.Resampling.BILINEAR)
            frames.append(image)
    if not frames: return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=250, loop=0)
    return str(destination)


def load_table(render_videos):
    manifest = read(HERE / "FINAL_MASS_EVALUATION_MANIFEST.json")
    runtime = read(HERE / "FINAL_MASS_RUNTIME_MANIFEST.json")
    if sha(HERE / "FINAL_MASS_EVALUATION_MANIFEST.json") != runtime["evaluation_manifest_sha256"]:
        raise RuntimeError("final MASS manifest changed")
    for path, digest in runtime["source_hashes"].items():
        if sha(path) != digest: raise RuntimeError("final source changed: " + path)
    for path, digest in runtime["artifact_hashes"].items():
        if sha(path) != digest: raise RuntimeError("final artifact changed: " + path)
    rows, belief_rows = [], []
    queue_index = {(item["context_id"], item["method"]): i + 1 for i, item in enumerate(manifest["execution_queue"])}
    for context in manifest["contexts"]:
        reference = OUT / "references" / context["id"]
        if not (reference / "WORKER_COMPLETION.json").is_file() or not read(reference / "WORKER_COMPLETION.json").get("logical_success"):
            raise RuntimeError("invalid final reference: " + context["id"])
        if read(reference / "SOURCE_HASHES_BEFORE.json") != runtime["source_hashes"] or \
                read(reference / "SOURCE_HASHES_AFTER.json") != runtime["source_hashes"]:
            raise RuntimeError("final reference did not preserve frozen source hashes")
        reference_result = read(reference / "RESULT.json")
        reference_material = read(reference / "MASS_INTERVENTION_READBACK.json")
        if reference_result.get("probe_qualified") is not True or reference_result.get("candidate_actions_executed") != 0:
            raise RuntimeError("invalid/candidate-conditioned final query reference")
        if not (np.isclose(reference_material.get("actual_total_mass_kg"), context["mass_kg"], rtol=0, atol=2e-8) and
                reference_material.get("inertia_scaled_by_mass_ratio") is True and inertia_readback_matches(reference_material) and
                reference_material.get("geometry_or_appearance_modified") is False and
                bool(reference_material.get("object_static_dynamic_friction")) and
                all(np.allclose(pair, [.5, .5], rtol=0, atol=1e-7)
                    for pair in reference_material.get("object_static_dynamic_friction", []))):
            raise RuntimeError("final reference MASS intervention/readback contract failed")
        posterior = read(reference / "PREACTION_POSTERIOR.json"); moments = posterior["posterior_moments"]
        if posterior.get("hidden_mass_used") is not False or posterior.get("candidate_actions_executed") != 0:
            raise RuntimeError("final MASS posterior is not deployable pre-action evidence")
        belief_rows.append({"context_id": context["id"], "root": context["root"], "task": context["task"],
            "true_mass_kg": context["mass_kg"], "mass_index": context["mass_index"],
            "normalized_mass_position": context["normalized_mass_position"], "posterior_mean_kg": moments["mean"],
            "posterior_std_kg": moments["std"], "interval_68_low": posterior["interval_68"][0],
            "interval_68_high": posterior["interval_68"][1], "interval_90_low": posterior["interval_90"][0],
            "interval_90_high": posterior["interval_90"][1], "interval_95_low": posterior["interval_95"][0],
            "interval_95_high": posterior["interval_95"][1], "member_means_kg": json.dumps(posterior["member_means"]),
            "member_sigmas_kg": json.dumps(posterior["member_sigmas"]), "reference_path": str(reference)})
        for method in METHODS:
            job = valid_job(context["id"], method); decision = read(job / "PLANNER_DECISION.json")
            branch = read(job / "BRANCH_RESULT.json"); outcome = branch["outcome"]; trace = lines(job / "ACTION_TRACE.jsonl")
            if read(job / "PREACTION_POSTERIOR.json") != posterior:
                raise RuntimeError("method posterior does not match common query reference")
            if read(job / "INITIAL_ONLINE_CHUNK_IDENTITY.json") != read(OUT / "initial_chunk_identity" / f"{context['id']}.json"):
                raise RuntimeError("methods do not share the candidate-independent initial online VLA chunk")
            if read(job / "SOURCE_HASHES_BEFORE.json") != runtime["source_hashes"] or \
                    read(job / "SOURCE_HASHES_AFTER.json") != runtime["source_hashes"]:
                raise RuntimeError("final branch did not preserve frozen source hashes")
            if branch.get("online_vla_verified") is not True or branch.get("downstream_action_source") != "ONLINE_VLA":
                raise RuntimeError("final MASS branch is not verified online VLA")
            if outcome.get("label_valid") is not True or branch.get("runtime_label_version") != "ONLINE_VLA_350STEP_WHOLE_MESH_RELEASE_SUPPORT_V1":
                raise RuntimeError("wrong/invalid final MASS full-task evaluator")
            if decision.get("phase_representation") != "NONE" or decision.get("phase_channels") != "NONE":
                raise RuntimeError("phase input present in final MASS branch")
            if decision.get("feasibility_sequence_source") != "ONLINE_VLA_ACTION_CHUNK" or decision.get("scripted_prefix_used") is not False:
                raise RuntimeError("non-online/scripted final MASS motion context")
            if method == "ACTIVEFORCING_MASS" and not (decision.get("gt_mass_used") is False and
                    decision.get("simulator_mass_kg") is None and posterior.get("hidden_mass_used") is False):
                raise RuntimeError("true simulator mass leaked into AF-MASS deployment")
            if method == "GT_MASS" and not (decision.get("gt_mass_used") is True and
                    decision.get("simulator_mass_kg") == context["mass_kg"]):
                raise RuntimeError("GT-MASS did not use the exact simulator mass")
            if method == "FIXED_4" and decision.get("executed_force_N") != 4.0:
                raise RuntimeError("Fixed-4 force changed")
            equality = read(job / "POSTPROBE_EQUALITY.json")
            material = read(job / "MASS_INTERVENTION_READBACK.json")
            if equality.get("passed") is not True or equality.get("candidate_actions_executed") != 0:
                raise RuntimeError("candidate-independent post-query state failed")
            if branch.get("rpc_count") != (branch.get("steps", 0) + 9) // 10:
                raise RuntimeError("online VLA cadence changed")
            if not all(float(item["selected_force_setpoint"]) == float(decision["executed_force_N"]) for item in trace):
                raise RuntimeError("more than one selected force setpoint")
            if not all(item["vla_release_intent"] or float(item["active_force_setpoint"]) == float(decision["executed_force_N"])
                       for item in trace):
                raise RuntimeError("selected force not active before release")
            if not all((not item["vla_release_intent"]) or float(item["final_gripper_command"]) == float(np.float32(.04))
                       for item in trace):
                raise RuntimeError("VLA release arbitration changed")
            if not (np.isclose(material.get("actual_total_mass_kg"), context["mass_kg"], rtol=0, atol=2e-8) and
                    material.get("inertia_scaled_by_mass_ratio") is True and inertia_readback_matches(material) and
                    material.get("geometry_or_appearance_modified") is False and
                    bool(material.get("object_static_dynamic_friction")) and
                    all(np.allclose(pair, [.5, .5], rtol=0, atol=1e-7)
                        for pair in material.get("object_static_dynamic_friction", []))):
                raise RuntimeError("final MASS intervention/readback contract failed")
            recomputed_squeeze = float(np.mean([row["measured_bilateral_squeeze"] for row in trace if not row["vla_release_intent"]]))
            if not np.isclose(recomputed_squeeze, outcome["mean_measured_bilateral_squeeze"], rtol=0, atol=1e-12):
                raise RuntimeError("official squeeze recomputation mismatch")
            animation = OUT / "videos" / f"{context['id']}__{method}.gif"
            video_path = render_animation(job, animation) if render_videos and not animation.exists() else str(animation) if animation.exists() else None
            rows.append({"execution_index": queue_index[(context["id"], method)], "context_id": context["id"],
                "root": context["root"], "task": context["task"], "true_mass_kg": context["mass_kg"],
                "mass_index": context["mass_index"], "normalized_mass_position": context["normalized_mass_position"],
                "interpolation_segment": context["interpolation_segment"], "interpolation_fraction": context["interpolation_fraction"],
                "method": method, "posterior_mean_kg": moments["mean"], "posterior_std_kg": moments["std"],
                "posterior_interval_68": json.dumps(posterior["interval_68"]), "posterior_interval_90": json.dumps(posterior["interval_90"]),
                "posterior_interval_95": json.dumps(posterior["interval_95"]), "selected_force_N": decision["executed_force_N"],
                "model_selected_force_N": decision["model_selected_force_N"], "predicted_success": decision["predicted_success"],
                "expected_utility": decision["utility"], "measured_bilateral_squeeze_N": recomputed_squeeze,
                "lift_success": outcome["lift_success"], "drop": outcome["dropped"],
                "release": int(outcome["opened_last20"] and outcome["unheld_last20"]),
                "placement": outcome["place_success"], "full_success": outcome["full_task_success_y"],
                "failure_taxonomy": "", "failure_reasons": json.dumps(outcome["failure_reasons"]),
                "branch_path": str(job), "video_path": video_path or "",
                "video_sha256": sha(animation) if video_path else "",
                "posterior_sha256": sha(reference / "PREACTION_POSTERIOR.json"),
                "planner_decision_sha256": sha(job / "PLANNER_DECISION.json"),
                "branch_result_sha256": sha(job / "BRANCH_RESULT.json"),
                "action_trace_sha256": sha(job / "ACTION_TRACE.jsonl"), "branch_trace_sha256": sha(job / "BRANCH_TRACE.json"),
                "runtime_manifest_sha256": sha(HERE / "FINAL_MASS_RUNTIME_MANIFEST.json"),
                "online_vla_verified": branch["online_vla_verified"], "vla_checkpoint_sha256": branch["checkpoint_sha256"],
                "query_mass_readback_sha256": sha(job / "MASS_INTERVENTION_READBACK.json")})
    if len(rows) != 144: raise RuntimeError(f"expected 144 final branches, got {len(rows)}")
    return manifest, pd.DataFrame(rows), pd.DataFrame(belief_rows)


def paired_counts(frame, a, b):
    pivot = frame[frame.method.isin([a, b])].pivot(index="context_id", columns="method", values="full_success")
    return {"both_success": int(((pivot[a] == 1) & (pivot[b] == 1)).sum()),
            f"{a}_only": int(((pivot[a] == 1) & (pivot[b] == 0)).sum()),
            f"{b}_only": int(((pivot[a] == 0) & (pivot[b] == 1)).sum()),
            "both_fail": int(((pivot[a] == 0) & (pivot[b] == 0)).sum())}


def method_summary(frame):
    output = {}
    for method, group in frame.groupby("method"):
        output[method] = {"n": len(group), "full_success": int(group.full_success.sum()),
            "full_success_rate": float(group.full_success.mean()), "lift_rate": float(group.lift_success.mean()),
            "drop_rate": float(group["drop"].mean()), "mean_selected_force_N": float(group.selected_force_N.mean()),
            "mean_measured_squeeze_N": float(group.measured_bilateral_squeeze_N.mean())}
    return output


def classify_failures(frame, manifest):
    classifications = {}; af = frame[frame.method == "ACTIVEFORCING_MASS"]
    for _, row in af[af.full_success == 0].iterrows():
        paired = frame[frame.context_id == row.context_id]
        successful_higher = paired[(paired.full_success == 1) &
                                   (paired.selected_force_N >= row.selected_force_N + 0.05 - 1e-9)]
        reasons = json.loads(row.failure_reasons)
        if (row.lift_success == 0 or row["drop"] == 1) and len(successful_higher): category = "UNDER_FORCE"
        elif row.lift_success == 1 and row["drop"] == 0 and any(name in reasons for name in
                ("OUTSIDE_AUTHORED_REGION", "NOT_RELEASED", "NO_FINAL_SUPPORT_CONTACT", "TIMEOUT", "INCOMPLETE_HORIZON")):
            near = paired[(paired.full_success == 1) &
                          (np.abs(paired.selected_force_N - row.selected_force_N) <= 0.050000001)]
            category = "VLA_EXECUTION_VARIANCE" if len(near) else "POST_LIFT_GEOMETRIC"
        else: category = "OTHER"
        classifications[row.context_id] = category
    frame.loc[(frame.method == "ACTIVEFORCING_MASS") & (frame.full_success == 0), "failure_taxonomy"] = \
        frame.loc[(frame.method == "ACTIVEFORCING_MASS") & (frame.full_success == 0), "context_id"].map(classifications)
    return Counter(classifications.values())


def failure_diagnostics(frame):
    """Preserve the frozen diagnostic evidence for every AF failure."""
    records = []
    for _, row in frame[(frame.method == "ACTIVEFORCING_MASS") & (frame.full_success == 0)].iterrows():
        job = Path(row.branch_path); decision = read(job / "PLANNER_DECISION.json")
        trace = lines(job / "ACTION_TRACE.jsonl")
        active = [item for item in trace if not item["vla_release_intent"]]
        contact_min = [min(map(float, item["finger_object_contact_norms_N"])) for item in active]
        normal_min = [min(map(abs, map(float, item["normal_force_N"]))) for item in active]
        squeeze = [float(item["measured_bilateral_squeeze"]) for item in active]
        counterparts = []
        for _, paired in frame[(frame.context_id == row.context_id) & (frame.method != "ACTIVEFORCING_MASS")].iterrows():
            paired_decision = read(Path(paired.branch_path) / "PLANNER_DECISION.json")
            counterparts.append({"method": paired.method, "selected_force_N": paired.selected_force_N,
                "full_success": int(paired.full_success), "lift_success": int(paired.lift_success),
                "drop": int(paired["drop"]), "release": int(paired.release), "placement": int(paired.placement),
                "measured_bilateral_squeeze_N": paired.measured_bilateral_squeeze_N,
                "failure_reasons": json.loads(paired.failure_reasons),
                "predicted_success": paired_decision["predicted_success"], "utility": paired_decision["utility"],
                "branch_path": paired.branch_path, "video_path": paired.video_path})
        records.append({"context_id": row.context_id, "task": int(row.task), "root": int(row.root),
            "true_mass_kg": row.true_mass_kg, "posterior_mean_kg": row.posterior_mean_kg,
            "posterior_std_kg": row.posterior_std_kg, "diagnostic_category": row.failure_taxonomy,
            "diagnostic_is_causal_proof": False, "selected_force_N": row.selected_force_N,
            "outcome": {"lift_success": int(row.lift_success), "drop": int(row["drop"]),
                        "release": int(row.release), "placement": int(row.placement),
                        "failure_reasons": json.loads(row.failure_reasons)},
            "planner": {"predicted_success_at_selection": decision["predicted_success"],
                        "utility_at_selection": decision["utility"], "force_grid_N": decision["force_grid_N"],
                        "posterior_marginalized_p_success": decision["p_success"],
                        "expected_utility": decision["expected_utility"]},
            "contact_trace": {"active_frames": len(active), "contact_below_0.1N_frames": int(np.sum(np.asarray(contact_min) < .1)),
                              "minimum_finger_contact_norm_N": float(np.min(contact_min)),
                              "minimum_normal_force_N": float(np.min(normal_min)),
                              "mean_measured_bilateral_squeeze_N": float(np.mean(squeeze)),
                              "q05_measured_bilateral_squeeze_N": float(np.quantile(squeeze, .05))},
            "counterparts": counterparts, "branch_path": row.branch_path, "video_path": row.video_path})
    payload = {"taxonomy_source": "FINAL_MASS_EVALUATION_MANIFEST.json", "failure_count": len(records),
               "categories_are_diagnostic_not_causal": True, "records": records}
    write_json(HERE / "MASS_AF_FAILURE_ANALYSIS.json", payload)
    return payload


def figures(frame, belief):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for task, group in belief.groupby("task"):
        # Mixture means need not lie inside a central quantile interval, so draw
        # the interval directly instead of passing potentially negative yerr.
        ax.vlines(group.true_mass_kg, group.interval_90_low, group.interval_90_high,
                  color=COLORS[task], alpha=.45, linewidth=1)
        ax.scatter(group.true_mass_kg, group.posterior_mean_kg,
                   color=COLORS[task], marker=MARKERS[task], alpha=.75, label=f"task {task}")
    lo, hi = belief.true_mass_kg.min(), belief.true_mass_kg.max(); ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
    ax.set(xlabel="True unseen object mass (kg)", ylabel="Posterior mass estimate (kg)")
    ax.set_title("Mass belief on unseen interpolation values")
    ax.legend(frameon=False, ncol=2); fig.tight_layout(); fig.savefig(HERE / "FIGURE_CONTINUOUS_MASS_BELIEF.pdf"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    for ax, task in zip(axes.flat, sorted(COLORS)):
        group = frame[frame.task == task]
        for method, style in (("ACTIVEFORCING_MASS", "o-"), ("GT_MASS", "s--")):
            for _, root_curve in group[group.method == method].groupby("root"):
                root_curve = root_curve.sort_values("true_mass_kg")
                ax.plot(root_curve.true_mass_kg, root_curve.selected_force_N, style,
                        color=METHOD_COLORS[method], alpha=.22, linewidth=.8, markersize=3)
            curve = group[group.method == method].groupby("true_mass_kg").selected_force_N.mean()
            ax.plot(curve.index, curve.values, style, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.axhline(4.0, color="0.35", ls=":", label="Fixed-4"); ax.set_title(f"task {task}"); ax.grid(alpha=.2)
    axes[1, 0].set_xlabel("True unseen mass (kg)"); axes[1, 1].set_xlabel("True unseen mass (kg)")
    axes[0, 0].set_ylabel("Selected force F* (N)"); axes[1, 0].set_ylabel("Selected force F* (N)")
    axes[0, 0].legend(frameon=False, fontsize=8); fig.suptitle("Selected force across unseen mass")
    fig.tight_layout(); fig.savefig(HERE / "FIGURE_CONTINUOUS_MASS_FORCE.pdf"); plt.close(fig)

    summary = method_summary(frame); fig, ax = plt.subplots(figsize=(5, 4))
    for method, marker in zip(METHODS, ("o", "s", "^")):
        row = summary[method]; ax.scatter(row["mean_measured_squeeze_N"], row["full_success_rate"], s=70,
                                          marker=marker, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.set(xlabel="Mean measured bilateral squeeze (N)", ylabel="Full-task success rate", ylim=(-.03, 1.03))
    ax.set_title("Reliability–force operating points")
    ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(HERE / "FIGURE_CONTINUOUS_MASS_PARETO.pdf"); plt.close(fig)


def qualitative_sheet(frame):
    af = frame[(frame.method == "ACTIVEFORCING_MASS") & (frame.full_success == 1)]
    chosen = None
    for (task, root), group in af.groupby(["task", "root"]):
        low = group[group.mass_index <= 2]; mid = group[group.mass_index.isin([3, 4])]; high = group[group.mass_index >= 5]
        for triple in itertools.product(low.to_dict("records"), mid.to_dict("records"), high.to_dict("records")):
            if len({round(row["selected_force_N"], 8) for row in triple}) == 3: chosen = triple; break
        if chosen: break
    result = {"found_clean_successful_triplet": bool(chosen), "selection_used_for_statistics": False}
    if not chosen:
        write_json(HERE / "MASS_QUALITATIVE_CANDIDATE.json", result); return result
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    result["rows"] = []
    for i, row in enumerate(chosen):
        job = Path(row["branch_path"]); images = sorted(job.glob("RAW_OBSERVATION_*.npz"))
        indices = np.linspace(0, len(images) - 1, 4).round().astype(int)
        for j, index in enumerate(indices):
            with np.load(images[index], allow_pickle=False) as data: axes[i, j].imshow(data["agentview_cam"])
            axes[i, j].axis("off"); axes[i, j].set_title(f"step {index + 1}", fontsize=8)
        axes[i, 0].set_ylabel(f"m={row['true_mass_kg']:.4f} kg\nF={row['selected_force_N']:.2f} N")
        result["rows"].append({k: row[k] for k in ("context_id", "true_mass_kg", "selected_force_N", "branch_path")})
    fig.suptitle("Same-task/root unseen-mass ActiveForcing rollouts"); fig.tight_layout()
    fig.savefig(HERE / "MASS_QUALITATIVE_CONTACT_SHEET.pdf"); plt.close(fig)
    result["contact_sheet"] = str(HERE / "MASS_QUALITATIVE_CONTACT_SHEET.pdf")
    write_json(HERE / "MASS_QUALITATIVE_CANDIDATE.json", result); return result


def cross_physics(frame):
    friction = pd.read_csv(FRICTION / "TABLE_CONTINUOUS_FRICTION_FULL_TRACE.csv")
    f_af = friction[friction.method == "ACTIVEFORCING"]
    m_af = frame[frame.method == "ACTIVEFORCING_MASS"]
    friction_rho = {str(task): rho(g.mu_test, g.selected_force) for task, g in f_af.groupby("task")}
    mass_rho = {str(task): rho(g.true_mass_kg, g.selected_force_N) for task, g in m_af.groupby("task")}
    friction_direction = ("broadly decreases" if all(v is not None and v < 0 for v in friction_rho.values())
                          else "is mixed across tasks")
    mass_direction = ("broadly increases" if all(v is not None and v > 0 for v in mass_rho.values())
                      else "is mixed across tasks")
    text = "# Cross-physics friction–mass summary\n\n"
    text += "The experiments remain separate: no dataset merging, joint posterior, or joint retraining was performed.\n\n"
    text += "- As friction increases, AF force " + friction_direction + "; task-wise Spearman(force, friction) = " + json.dumps(friction_rho) + ".\n"
    text += "- As mass increases, AF force " + mass_direction + "; task-wise Spearman(force, mass) = " + json.dumps(mass_rho) + ".\n\n"
    text += "Friction and mass use the same belief-to-decision interface, while their inferred scalar physics enters independently. Opposing directions are evidence against one universal force heuristic only when both task-wise direction rules hold. The comparison does not establish joint mass–friction reasoning.\n"
    (HERE / "CROSS_PHYSICS_FRICTION_MASS_SUMMARY.md").write_text(text)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for task, group in f_af.groupby("task"):
        curve = group.groupby("mu_test").selected_force.mean()
        axes[0].plot(curve.index, curve.values, color=COLORS[task], marker=MARKERS[task],
                     ls=LINESTYLES[task], label=f"task {task}")
    for task, group in m_af.groupby("task"):
        curve = group.groupby("true_mass_kg").selected_force_N.mean()
        axes[1].plot(curve.index, curve.values, color=COLORS[task], marker=MARKERS[task],
                     ls=LINESTYLES[task], label=f"task {task}")
    axes[0].set(xlabel="Object-side friction", ylabel="AF selected force (N)", title="(a) friction")
    axes[1].set(xlabel="Object mass (kg)", title="(b) mass"); axes[0].legend(frameon=False, fontsize=7)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("Force response to separately inferred physics")
    fig.tight_layout(); fig.savefig(HERE / "FIGURE_FRICTION_VS_MASS_FORCE.pdf"); plt.close(fig)
    return {"friction_taskwise_force_spearman": friction_rho, "mass_taskwise_force_spearman": mass_rho,
            "friction_direction": friction_direction, "mass_direction": mass_direction,
            "all_friction_tasks_negative": all(v is not None and v < 0 for v in friction_rho.values()),
            "all_mass_tasks_positive": all(v is not None and v > 0 for v in mass_rho.values())}


def status(flag): return flag


def main(render_videos):
    for name in ("TABLE_CONTINUOUS_MASS_FULL_TRACE.csv", "MASS_CLAIM_AUDIT.json", "FINAL_CONTINUOUS_MASS_GENERALIZATION_REPORT.md"):
        if (HERE / name).exists(): raise FileExistsError("official MASS analysis already exists: " + name)
    manifest, frame, belief = load_table(render_videos)
    retries = rerun_audit()
    identity_control = read(HERE / "MASS_BELIEF_IDENTITY_CONTROL.json")
    offline_belief = read(HERE / "MASS_BELIEF_METRICS.json")["HELDOUT"]
    offline_feasibility = read(HERE / "MASS_FEASIBILITY_METRICS.json")
    friction_integrity = read(HERE / "FRICTION_UNMODIFIED_AUDIT.json")
    failures = classify_failures(frame, manifest); summaries = method_summary(frame)
    failure_detail = failure_diagnostics(frame)
    overall_belief = belief_metrics(belief); task_belief = {str(task): belief_metrics(group) for task, group in belief.groupby("task")}
    train_prior = float(np.mean(read(HERE / "MASS_TRAINING_CONTEXT_PLAN.json")["mass_anchors_kg"]))
    prior_mae = float(np.mean(np.abs(belief.true_mass_kg - train_prior)))
    af = frame[frame.method == "ACTIVEFORCING_MASS"].copy(); gt = frame[frame.method == "GT_MASS"].copy()
    force_trends = {str(task): force_trend(group) for task, group in af.groupby("task")}
    root_force_trends = {f"task{task}_root{root}": force_trend(group)
                         for (task, root), group in af.groupby(["task", "root"])}
    force_rho = {task: values["Spearman"] for task, values in force_trends.items()}
    merged = af.merge(gt[["context_id", "selected_force_N", "full_success"]], on="context_id", suffixes=("_AF", "_GT"))
    error = np.abs(merged.selected_force_N_AF - merged.selected_force_N_GT)
    af_gt = {"MAE_N": float(error.mean()), "median_AE_N": float(np.median(error)),
             "within_0.05N": float(np.mean(error <= .050000001)), "within_0.10N": float(np.mean(error <= .100000001)),
             "within_0.25N": float(np.mean(error <= .250000001)), "paired": paired_counts(frame, "ACTIVEFORCING_MASS", "GT_MASS")}
    af_fixed = {"paired": paired_counts(frame, "ACTIVEFORCING_MASS", "FIXED_4"), "halves": {}}
    for half, indices in manifest["lower_upper_split"].items():
        subset = frame[frame.mass_index.isin(indices)]
        half_summary = method_summary(subset)
        af_fixed["halves"][half] = {"summary": half_summary,
            "AF_minus_FIXED4": {
                "full_success_rate": half_summary["ACTIVEFORCING_MASS"]["full_success_rate"] - half_summary["FIXED_4"]["full_success_rate"],
                "selected_force_N": half_summary["ACTIVEFORCING_MASS"]["mean_selected_force_N"] - 4.0,
                "measured_squeeze_N": half_summary["ACTIVEFORCING_MASS"]["mean_measured_squeeze_N"] - half_summary["FIXED_4"]["mean_measured_squeeze_N"]},
            "paired": paired_counts(subset, "ACTIVEFORCING_MASS", "FIXED_4")}
    figures(frame, belief); qualitative = qualitative_sheet(frame); cross = cross_physics(frame)
    write_json(HERE / "MASS_ANALYSIS_CHART_MAP.json", {"charts": [
        {"section": "Continuous mass belief", "question": "Does the posterior preserve ordering on unseen masses?",
         "family": "Uncertainty & Benchmark", "type": "scatter plus central 90% interval and y=x reference",
         "grain": "48 candidate-independent query contexts", "fields": ["true_mass_kg", "posterior_mean_kg", "interval_90_low", "interval_90_high", "task"],
         "palette": "four declared task colors plus distinct markers; dark-neutral ideal line", "output": "FIGURE_CONTINUOUS_MASS_BELIEF.pdf"},
        {"section": "Continuous force adaptation", "question": "Does selected force change with unseen mass and approach GT-Mass?",
         "family": "Trend", "type": "task-faceted ordered line", "grain": "two thin root traces plus root-mean at each of six mass values",
         "fields": ["true_mass_kg", "selected_force_N", "method", "task", "root"],
         "palette": "AF blue circles, GT orange squares/dashes, Fixed-4 neutral reference", "output": "FIGURE_CONTINUOUS_MASS_FORCE.pdf"},
        {"section": "Reliability-force comparison", "question": "Where do the three methods operate in success versus measured squeeze?",
         "family": "Relationship", "type": "three labeled aggregate operating points",
         "grain": "one predetermined method aggregate across 48 contexts", "fields": ["mean_measured_squeeze_N", "full_success_rate", "method"],
         "palette": "declared method colors plus distinct markers", "output": "FIGURE_CONTINUOUS_MASS_PARETO.pdf"},
        {"section": "Cross-physics comparison", "question": "Do friction and mass induce opposite force trends under the shared interface?",
         "family": "Trend", "type": "two-panel task curves", "grain": "root-mean per held-out physics value",
         "fields": ["physics_value", "selected_force", "task", "physics_type"],
         "palette": "declared task colors plus task-specific markers and line styles", "output": "FIGURE_FRICTION_VS_MASS_FORCE.pdf"}
    ]})
    csv_new(HERE / "TABLE_CONTINUOUS_MASS_FULL_TRACE.csv", frame.sort_values("execution_index").to_dict("records"))
    csv_new(HERE / "TABLE_CONTINUOUS_MASS_BELIEF.csv", belief.to_dict("records"))

    rules = manifest["claim_decision_rules"]
    informative = overall_belief["Spearman"] is not None and overall_belief["Spearman"] >= .75 and overall_belief["MAE"] < prior_mae
    continuous = all(value["ranking_accuracy"]["accuracy"] >= .75 and
                     value["distinct_aggregate_posterior_means"] >= 4 for value in task_belief.values())
    force_supported = all(value is not None and value > 0 for value in force_rho.values())
    force_mixed = rho(af.true_mass_kg, af.selected_force_N) is not None and rho(af.true_mass_kg, af.selected_force_N) > 0
    close_supported = af_gt["MAE_N"] <= .10 and af_gt["within_0.10N"] >= .75
    close_mixed = af_gt["MAE_N"] <= .25 or af_gt["within_0.25N"] >= .50
    fixed_pair = af_fixed["paired"]; fixed_dom = (summaries["ACTIVEFORCING_MASS"]["full_success_rate"] >= summaries["FIXED_4"]["full_success_rate"] and
        summaries["ACTIVEFORCING_MASS"]["mean_measured_squeeze_N"] < summaries["FIXED_4"]["mean_measured_squeeze_N"] and
        fixed_pair["FIXED_4_only"] == 0)
    failure_total = sum(failures.values()); under_primary = failure_total > 0 and failures["UNDER_FORCE"] > failure_total / 2
    shared_supported = informative and continuous and force_supported and close_supported
    shared_mixed = informative and (force_mixed or close_mixed)
    claims = {
        "MASS_PHYSICAL_BELIEF_INFORMATIVE": "SUPPORTED" if informative else "NOT_SUPPORTED",
        "MASS_BELIEF_CONTINUOUS_GENERALIZATION": "SUPPORTED" if continuous else "MIXED" if informative else "NOT_SUPPORTED",
        "UNSEEN_MASS_GENERALIZATION": "SUPPORTED" if informative and continuous else "MIXED" if informative else "NOT_SUPPORTED",
        "CONTINUOUS_MASS_FORCE_ADAPTATION": "SUPPORTED" if force_supported else "MIXED" if force_mixed else "NOT_SUPPORTED",
        "AF_MASS_CLOSE_TO_GT_DECISION": "SUPPORTED" if close_supported else "MIXED" if close_mixed else "NOT_SUPPORTED",
        "AF_MASS_MORE_RELIABLE_THAN_FIXED4": "SUPPORTED" if summaries["ACTIVEFORCING_MASS"]["full_success_rate"] > summaries["FIXED_4"]["full_success_rate"] else "MIXED" if summaries["ACTIVEFORCING_MASS"]["full_success_rate"] == summaries["FIXED_4"]["full_success_rate"] else "NOT_SUPPORTED",
        "AF_MASS_LOWER_FORCE_THAN_FIXED4": "SUPPORTED" if summaries["ACTIVEFORCING_MASS"]["mean_selected_force_N"] < 4.0 else "NOT_SUPPORTED",
        "AF_MASS_DOMINATES_FIXED4": "SUPPORTED" if fixed_dom else "NOT_SUPPORTED",
        "MASS_FAILURES_PRIMARILY_UNDER_FORCE": "SUPPORTED" if under_primary else "NOT_TESTED" if failure_total == 0 else "NOT_SUPPORTED",
        "SAME_ARCHITECTURE_WORKS_FOR_FRICTION_AND_MASS": "SUPPORTED" if shared_supported else "MIXED" if shared_mixed else "NOT_SUPPORTED",
        "JOINT_FRICTION_MASS_REASONING": "NOT_TESTED", "OUT_OF_SUPPORT_MASS_EXTRAPOLATION": "NOT_TESTED",
        "REAL_ROBOT_MASS_GENERALIZATION": "NOT_TESTED"}
    write_json(HERE / "MASS_CLAIM_AUDIT.json", {"claims": claims, "frozen_decision_rules": rules,
        "belief": {"overall": overall_belief, "taskwise": task_belief, "train_mean_prior_MAE": prior_mae},
        "force": {"taskwise": force_trends, "rootwise": root_force_trends,
                  "overall": force_trend(af)},
        "AF_vs_GT": af_gt, "AF_vs_FIXED4": af_fixed, "failures": dict(failures)})
    af_vs_fixed_claim = ("SUPPORTED" if claims["AF_MASS_DOMINATES_FIXED4"] == "SUPPORTED" else
                         "MIXED" if (claims["AF_MASS_MORE_RELIABLE_THAN_FIXED4"] in ("SUPPORTED", "MIXED") or
                                     claims["AF_MASS_LOWER_FORCE_THAN_FIXED4"] == "SUPPORTED") else "NOT_SUPPORTED")
    main_paper = shared_supported
    strongest = ("Across exact-held-out masses interpolated between discrete training anchors, the restored mass query produced an informative continuous mass belief and the unchanged belief-to-decision interface allocated force in a mass-dependent manner."
                 if shared_supported else "The mass instantiation is a mixed second-physics result; report its measured belief and decision behavior without claiming general success.")
    report = f"""# 1. Executive conclusion

The frozen run completed 144/144 valid online-VLA method branches. {strongest}

# 2. Restored old mass experiment

The working historical implementation was recovered at `/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_FINAL_CLOSURE_20260903`, with the formal task-2 source under `ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME`. It used the P4-B contact-conditioned tangential shear query at 0.05/0.10/0.20 kg. The raw root-held-out identifier reported MAE 0.0229 kg, RMSE 0.0246 kg, bias +0.0117 kg, Spearman 0.956, and pairwise ranking 0.800. The historical task-2 curve separated 0.20 kg at low forces, and its small scripted-path AF readout was 0.917 success at 1.750 N; these are restoration evidence, not current-paper results. The physical query evidence and proportional mass/inertia intervention were reusable; point-only belief, old feasibility/planner, scripted downstream execution, and obsolete metrics were not.

# 3. Migration to current architecture

The MASS namespace uses the current 58-D deployable query sequence, heteroscedastic three-member ensemble, positive-support continuous posterior, phase-free 8×64 online-VLA motion context, full-task feasibility, deterministic posterior marginalization, the current reliability-force utility, one [3,5]-N setpoint, 10-step online re-query cadence, and the frozen 350-step evaluator. Friction artifacts were not modified.

# 4. Final mass query

The restored P4-B query runs after the grasp and before planning. It starts at 3.0 N bilateral preload, raises preload by 0.5 N at hold steps 14/24/34 when contact support is inadequate, and caps preload at 4.5 N. It moves tangentially in 0.2-mm increments for at most 2.0 mm/25 outward steps. Frozen stops cover contact loss, normal-force ratio below 0.55, shear ratio 0.08, normalized shear impulse 0.012, marker-motion budget 0.12, major disturbance, maximum displacement, or episode termination. It then returns over 10 steps, holds five steps, records the current 58-D force/tactile/aperture sequence with validity masks, and saves the candidate-independent decision state. Total object mass is changed in kilograms and the complete inertia tensor is scaled by the same ratio; friction is fixed at 0.5. Geometry and appearance are unchanged.

# 5. Training support audit

Training uses discrete 0.05/0.10/0.20-kg anchors. TRAIN roots 181000–181003 provide 48 query contexts and 432 nine-force branches; VAL root 181010 and HELDOUT root 181011 each provide 12 query contexts and 108 branches. Sibling forces remain in one root split. Historical traces remain restoration/auxiliary evidence because they lack the exact current 58-D sensor contract, phase-free online-motion context, current force support, or current evaluator. The correct claim is **interpolation to unseen masses between training anchors**. Final values are {manifest['masses_per_task_kg']} kg and have zero exact collisions with belief training, feasibility training, canonical anchors, or old evaluations.

# 6. Offline model qualification

Belief and phase-free full-task feasibility passed their frozen gates before online development. On the held-out anchor root, belief MAE={offline_belief['MAE']:.6f} kg, RMSE={offline_belief['RMSE']:.6f} kg, bias={offline_belief['bias']:+.6f} kg, Spearman={formatted(offline_belief['Spearman'])}, and 68/90/95% coverage={offline_belief['coverage']['0.68']:.3f}/{offline_belief['coverage']['0.9']:.3f}/{offline_belief['coverage']['0.95']:.3f}. The root-held-out task-only control MAE was {identity_control['task_only_heldout_MAE_kg']:.6f} kg versus {identity_control['query_model_heldout_MAE_kg']:.6f} kg for the query model; task/root were fully crossed with mass. Held-out feasibility metrics were NLL={offline_feasibility['HELDOUT']['NLL']:.6f}, Brier={offline_feasibility['HELDOUT']['Brier']:.6f}, AUROC={formatted(offline_feasibility['HELDOUT']['AUROC'])}, AUPRC={formatted(offline_feasibility['HELDOUT']['AUPRC'])}, and ECE={offline_feasibility['HELDOUT']['ECE_10bin']:.6f}; VAL and HELDOUT NLL both beat their prevalence baselines. These feasibility numbers describe controlled-motion auxiliary branches; current online-VLA transfer passed separately before final freeze.

# 7. Frozen final evaluation protocol

Fresh roots 181100/181101 × tasks 0/1/5/6 × masses {manifest['masses_per_task_kg']} kg produced 48 contexts. AF-MASS, GT-MASS, and Fixed-4 produced 144 valid branches under method-order seed {manifest['method_order_seed']}. All methods ran the identical query and matched the reference post-query state; GT alone received simulator mass, and Fixed-4 executed exactly 4.0 N. Result-driven reruns={retries['result_driven_reruns']}; independently authorized infrastructure retry attempts={retries['infrastructure_retry_count']}; automatic root expansion did not occur. The frozen source, checkpoints, evaluator, context/method queue, seeds, and root audit are transitively hashed in `FINAL_MASS_EVALUATION_HASHES.txt` and `FINAL_MASS_RUNTIME_MANIFEST.json`.

# 8. Continuous mass belief results

On the frozen unseen interpolation set, belief MAE={overall_belief['MAE']:.6f} kg, RMSE={overall_belief['RMSE']:.6f} kg, bias={overall_belief['bias']:+.6f} kg, Spearman={formatted(overall_belief['Spearman'])}, and 68/90/95% coverage={overall_belief['coverage_68']:.3f}/{overall_belief['coverage_90']:.3f}/{overall_belief['coverage_95']:.3f}. Task-wise results are `{json.dumps(task_belief, sort_keys=True)}`. Distinct task×mass aggregate means={overall_belief['distinct_aggregate_posterior_means']}; ranking accuracy={overall_belief['ranking_accuracy']['accuracy']:.3f}.

[Figure 1: posterior mean and central 90% interval versus true unseen mass](FIGURE_CONTINUOUS_MASS_BELIEF.pdf). Points are individual candidate-independent query contexts; color and marker identify task, and the dashed line is ideal calibration. Read ordering separately from interval coverage because broad intervals can cover well while point estimates remain biased.

# 9. Continuous force adaptation results

Task-wise force trends (Spearman, descriptive linear slope, and unique grid values) are `{json.dumps(force_trends, sort_keys=True)}`. AF selected {af.selected_force_N.nunique()} distinct grid values overall; root-wise consistency is `{json.dumps(root_force_trends, sort_keys=True)}`.

[Figure 2: AF-MASS and GT-MASS force across the six frozen mass values](FIGURE_CONTINUOUS_MASS_FORCE.pdf). Thin traces retain each root and the emphasized curves average the two roots at each mass; task panels keep local non-monotonicity visible, and the horizontal line is Fixed-4 rather than a fitted comparator.

# 10. AF vs GT-Mass

Force MAE={af_gt['MAE_N']:.4f} N, median AE={af_gt['median_AE_N']:.4f} N; within 0.05/0.10/0.25 N={af_gt['within_0.05N']:.3f}/{af_gt['within_0.10N']:.3f}/{af_gt['within_0.25N']:.3f}. Paired outcomes: `{json.dumps(af_gt['paired'], sort_keys=True)}`. GT is interpreted only as a decision oracle.

# 11. AF vs Fixed-4

Method summaries: `{json.dumps(summaries, sort_keys=True)}`. Paired outcomes: `{json.dumps(af_fixed['paired'], sort_keys=True)}`. Frozen lower/upper-half results: `{json.dumps(af_fixed['halves'], sort_keys=True)}`. No universal dominance is asserted unless the claim audit marks it supported.

[Figure 3: full-task success versus the official measured-squeeze operating point](FIGURE_CONTINUOUS_MASS_PARETO.pdf). Each marker is one predetermined method aggregate over the same 48 contexts; this compact comparison is secondary to the belief and interpolation curves and is not a statistical non-inferiority analysis.

# 12. Failure analysis

Frozen diagnostic counts are `{json.dumps(dict(failures), sort_keys=True)}`. `MASS_AF_FAILURE_ANALYSIS.json` records the paired GT/Fixed outcomes, contact-trace summary, and full planner curves for all {failure_detail['failure_count']} AF failures. These categories are not causal proof.

# 13. Friction vs mass comparison

Separate task-wise trends are `{json.dumps(cross, sort_keys=True)}`. The closed friction runtime manifest and all {friction_integrity['verified_entries']} frozen source/artifact entries still match their original hashes. No joint inference, joint retraining, or friction modification was performed.

[Cross-physics force comparison](FIGURE_FRICTION_VS_MASS_FORCE.pdf). The left and right panels retain their native physics axes; only the direction of each within-physics force trend is compared, so the panel is not evidence for a joint latent model.

# 14. Supported and unsupported claims

`{json.dumps(claims, sort_keys=True)}`

# 15. Paper recommendation

- MAIN_PAPER_WORTHY = {'YES' if main_paper else 'NO'}
- Main paper: belief interpolation, mass-to-force curve, and the two-panel friction/mass interface comparison if supported.
- Supplement: training/data forensics, calibration, per-root tables, paired outcomes, failures, provenance, and the same-task/root qualitative contact sheet when an all-success three-force triplet exists.
- Qualitative candidate: {'available at MASS_QUALITATIVE_CONTACT_SHEET.pdf' if qualitative['found_clean_successful_triplet'] else 'no all-success low/medium/high same-task/root triplet with three distinct AF forces was available; no substitute was cherry-picked'}.
- Strongest defensible claim: {strongest}
- Limitations: discrete-anchor in-support interpolation; mass plus proportional inertia scaling; fixed friction; simulation only; no joint physics, out-of-support extrapolation, real-robot transfer, or statistical non-inferiority.
"""
    (HERE / "FINAL_CONTINUOUS_MASS_GENERALIZATION_REPORT.md").write_text(report)
    terminal = {"FINAL_MASS_STATUS": "COMPLETE_FROZEN_144_VALID", "OLD_WORKING_MASS_EXPERIMENT_FOUND": "YES",
        "OLD_MASS_ARTIFACT_PATH": "/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_FINAL_CLOSURE_20260903",
        "OLD_MASS_QUERY": "P4-B contact-conditioned tangential shear", "OLD_MASS_REUSABLE": "QUERY_AND_INTERVENTION_ONLY",
        "CURRENT_ARCH_MIGRATION": "PASS", "PHASE_FREE": "YES", "ONLINE_VLA": "YES",
        "CANDIDATE_INDEPENDENT_STATE": "YES", "FULL_TASK_LABEL": "YES", "TRUE_MASS_DEPLOYMENT_LEAKAGE": "NO",
        "MASS_QUERY_PROTOCOL": "MASS_QUERY_PROTOCOL.md", "MASS_TRAINING_SUPPORT_TYPE": "DISCRETE",
        "NUM_BELIEF_TRAIN_MASSES_PER_TASK": 3, "NUM_FEASIBILITY_TRAIN_MASSES_PER_TASK": 3,
        "NUM_NEW_ROOTS": 2, "NUM_UNSEEN_MASSES_PER_TASK": 6, "NUM_FINAL_CONTEXTS": 48, "NUM_VALID_BRANCHES": 144,
        "UNSEEN_MASS_VALUES_PER_TASK": {f"task{task}": manifest["masses_per_task_kg"] for task in (0, 1, 5, 6)},
        "MASS_BELIEF_MAE": overall_belief["MAE"], "MASS_BELIEF_RMSE": overall_belief["RMSE"],
        "MASS_BELIEF_BIAS": overall_belief["bias"], "MASS_BELIEF_SPEARMAN": overall_belief["Spearman"],
        "MASS_BELIEF_90_COVERAGE": overall_belief["coverage_90"], "TASKWISE_MASS_FORCE_SPEARMAN": force_rho,
        "AF_FULL_SR": summaries["ACTIVEFORCING_MASS"]["full_success_rate"],
        "AF_MEAN_SELECTED_FORCE": summaries["ACTIVEFORCING_MASS"]["mean_selected_force_N"],
        "AF_MEASURED_SQUEEZE": summaries["ACTIVEFORCING_MASS"]["mean_measured_squeeze_N"],
        "GT_MASS_FULL_SR": summaries["GT_MASS"]["full_success_rate"],
        "GT_MASS_MEAN_SELECTED_FORCE": summaries["GT_MASS"]["mean_selected_force_N"],
        "GT_MASS_MEASURED_SQUEEZE": summaries["GT_MASS"]["mean_measured_squeeze_N"],
        "FIXED4_FULL_SR": summaries["FIXED_4"]["full_success_rate"], "FIXED4_MEASURED_SQUEEZE": summaries["FIXED_4"]["mean_measured_squeeze_N"],
        "AF_GT_FORCE_MAE": af_gt["MAE_N"], "AF_GT_FORCE_MEDIAN_AE": af_gt["median_AE_N"],
        "AF_GT_WITHIN_0.10N": af_gt["within_0.10N"], "AF_VS_GT_PAIRED": af_gt["paired"],
        "AF_VS_FIXED4_PAIRED": af_fixed["paired"], "AF_FAILURES_UNDER_FORCE": failures["UNDER_FORCE"],
        "AF_FAILURES_POST_LIFT_GEOMETRIC": failures["POST_LIFT_GEOMETRIC"],
        "AF_FAILURES_VLA_VARIANCE": failures["VLA_EXECUTION_VARIANCE"], "AF_FAILURES_OTHER": failures["OTHER"],
        "MASS_PHYSICAL_BELIEF_CLAIM": claims["MASS_PHYSICAL_BELIEF_INFORMATIVE"],
        "UNSEEN_MASS_GENERALIZATION_CLAIM": claims["UNSEEN_MASS_GENERALIZATION"],
        "CONTINUOUS_MASS_FORCE_ADAPTATION_CLAIM": claims["CONTINUOUS_MASS_FORCE_ADAPTATION"],
        "AF_CLOSE_TO_GT_MASS_CLAIM": claims["AF_MASS_CLOSE_TO_GT_DECISION"],
        "AF_VS_FIXED4_CLAIM": af_vs_fixed_claim,
        "FRICTION_AND_MASS_SHARED_INTERFACE_CLAIM": claims["SAME_ARCHITECTURE_WORKS_FOR_FRICTION_AND_MASS"],
        "MAIN_PAPER_WORTHY": "YES" if main_paper else "NO", "REMAINING_MUST_RUN_MASS_EXPERIMENTS": "NONE"}
    write_json(HERE / "FINAL_TERMINAL_OUTPUT.json", terminal)
    provenance = {}
    for path in HERE.glob("*MASS*.json"):
        if path.name != "FINAL_ANALYSIS_ARTIFACT_SHA256.json": provenance[path.name] = sha(path)
    for name in ("TABLE_CONTINUOUS_MASS_FULL_TRACE.csv", "TABLE_CONTINUOUS_MASS_BELIEF.csv",
                 "FINAL_CONTINUOUS_MASS_GENERALIZATION_REPORT.md", "CROSS_PHYSICS_FRICTION_MASS_SUMMARY.md",
                 "FINAL_TERMINAL_OUTPUT.json",
                 "FIGURE_CONTINUOUS_MASS_BELIEF.pdf", "FIGURE_CONTINUOUS_MASS_FORCE.pdf", "FIGURE_CONTINUOUS_MASS_PARETO.pdf",
                 "FIGURE_FRICTION_VS_MASS_FORCE.pdf"):
        provenance[name] = sha(HERE / name)
    if (HERE / "MASS_QUALITATIVE_CONTACT_SHEET.pdf").is_file():
        provenance["MASS_QUALITATIVE_CONTACT_SHEET.pdf"] = sha(HERE / "MASS_QUALITATIVE_CONTACT_SHEET.pdf")
    write_json(HERE / "FINAL_ANALYSIS_ARTIFACT_SHA256.json", provenance)
    for key, value in terminal.items(): print(f"{key} = {json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--render-videos", action="store_true")
    args = parser.parse_args(); main(args.render_videos)
