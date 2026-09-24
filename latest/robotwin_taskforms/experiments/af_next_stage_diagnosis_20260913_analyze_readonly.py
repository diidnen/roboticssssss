"""Read-only Phase 0-3/6 diagnostics for the frozen dump_bin_bigbin V4 run.

This script never writes into the source experiment directories and never invokes
the simulator.  It reconstructs the rollout grain from original JSON and full
physics traces, recomputes offline decision curves, audits matched comparisons,
and quantifies motion-distribution coverage.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


BASE = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911")
EXPERIMENTS = BASE / "experiments"
V4 = EXPERIMENTS / "af_dump_maxf8_20260913"
INFERENCE = EXPERIMENTS / "af_dump_maxf8_confirmation_storage_20260913/inference_v4"
MECHANISM = EXPERIMENTS / "af_dump_mechanism_20260913"
OLD_DATA = EXPERIMENTS / "af_dump_original_restore_20260912/original_rootlocal_dataset_v3_rim20"
NEW_DATA = V4 / "additional_data_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=lambda item: item.item() if isinstance(item, np.generic) else item.tolist(),
        )
        + "\n"
    )


def norm(v) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def quat_angle_deg(a, b) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa /= max(np.linalg.norm(qa), 1e-12)
    qb /= max(np.linalg.norm(qb), 1e-12)
    return float(np.degrees(2 * np.arccos(np.clip(abs(float(np.dot(qa, qb))), 0.0, 1.0))))


def trace_summary(args):
    context_id, method, branch = args
    branch = Path(branch)
    result = read(branch / "result.json")
    digest = hashlib.sha256()
    forces, target_forces = [], []
    bin_positions, bin_quats, bin_linear_speeds, bin_angular_speeds = [], [], [], []
    gripper_centers, relative_offsets = [], []
    streak = 0
    first_loss = None
    last = None
    with gzip.open(branch / "physics_trace.jsonl.gz", "rb") as stream:
        for raw in stream:
            digest.update(raw.rstrip(b"\n"))
            row = json.loads(raw)
            last = row
            contact = row["contact"]
            force = float(contact["measured_squeeze_n"])
            target_force = float(contact["target_measured_squeeze_n"])
            forces.append(force)
            target_forces.append(target_force)
            if target_force < 0.5:
                streak += 1
                if streak == 25 and first_loss is None:
                    first_loss = len(target_forces) - 25
            else:
                streak = 0
            actor = next(a for a in row["state"]["actors"] if a["name"] == "063_tabletrashbin")
            pose = actor["pose"]
            pos = np.asarray(pose[:3], dtype=np.float64)
            bin_positions.append(pos)
            bin_quats.append(pose[3:7])
            bin_linear_speeds.append(norm(actor["native_com_linear_velocity"]))
            bin_angular_speeds.append(norm(actor["native_angular_velocity"]))
            fingers = contact["fingers"][:2]
            centers = [f["native_joint_readback"]["com_position_world"] for f in fingers]
            center = np.mean(np.asarray(centers, dtype=np.float64), axis=0)
            gripper_centers.append(center)
            relative_offsets.append(center - pos)
    if digest.hexdigest() != result["trace_sha256"]:
        raise ValueError(f"Trace hash mismatch: {branch}")
    arr_force = np.asarray(forces)
    arr_target = np.asarray(target_forces)
    arr_pos = np.stack(bin_positions)
    arr_grip = np.stack(gripper_centers)
    arr_rel = np.stack(relative_offsets)
    final_garbage = [
        a["pose"][:3] for a in last["state"]["actors"] if a["name"] == "garbage"
    ]
    motion_file = branch / "original_motion_feature.json"
    return {
        "context": context_id,
        "method": method,
        "branch": branch.name,
        "commanded_force_N": float(result["force_setpoint_bilateral_n"]),
        "success": int(result["success"]),
        "native_actions": int(result["native_actions"]),
        "physics_steps": int(result["physics_steps"]),
        "measured_force_mean_N": float(arr_force.mean()),
        "measured_force_first1000_mean_N": float(arr_force[:1000].mean()),
        "measured_force_p99_N": float(np.quantile(arr_force, 0.99)),
        "measured_force_max_N": float(arr_force.max()),
        "contact_fraction": float(np.mean(arr_target >= 0.5)),
        "sustained_contact_loss": first_loss is not None,
        "first_sustained_contact_loss_step": first_loss,
        "diagnostic_lift_reached_1m": bool(float(arr_pos[:, 2].max()) >= 1.0),
        "object_initial_position_xyz": arr_pos[0].tolist(),
        "object_final_position_xyz": arr_pos[-1].tolist(),
        "object_max_z_m": float(arr_pos[:, 2].max()),
        "object_max_translation_from_initial_m": float(np.max(np.linalg.norm(arr_pos - arr_pos[0], axis=1))),
        "object_max_linear_speed_mps": float(max(bin_linear_speeds)),
        "object_max_angular_speed_radps": float(max(bin_angular_speeds)),
        "gripper_center_max_translation_m": float(np.max(np.linalg.norm(arr_grip - arr_grip[0], axis=1))),
        "relative_offset_max_change_m": float(np.max(np.linalg.norm(arr_rel - arr_rel[0], axis=1))),
        "final_garbage_positions_xyz": final_garbage,
        "trajectory_path": str(branch / "physics_trace.jsonl.gz"),
        "motion_descriptor_path": str(motion_file),
        "motion_descriptor_sha256": sha(motion_file),
        "trace_sha256": digest.hexdigest(),
    }


def cdf(x, means, sigmas):
    return float(np.mean([0.5 * (1 + math.erf((x - a) / (b * math.sqrt(2)))) for a, b in zip(means, sigmas)]))


def posterior_stats(posterior, true_mu):
    means = posterior["member_means"]
    sigmas = np.exp(posterior["member_log_sigmas"])
    c0 = cdf(0, means, sigmas)

    def quantile(q):
        lo, hi = 0.0, max(means) + 12 * max(sigmas)
        for _ in range(75):
            mid = (lo + hi) / 2
            if (cdf(mid, means, sigmas) - c0) / (1 - c0) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    mean = float(posterior["posterior_moments"]["mean"])
    q05, q95 = quantile(0.05), quantile(0.95)
    return {
        "true_mu": float(true_mu),
        "posterior_mean": mean,
        "posterior_std": float(posterior["posterior_moments"]["std"]),
        "absolute_error": abs(mean - true_mu),
        "q05": q05,
        "q95": q95,
        "covered90": q05 <= true_mu <= q95,
    }


def point_curve(model, feature, mu):
    sequence = np.broadcast_to(np.asarray(feature["sequence"], np.float32), (151, 8, 64)).copy()
    sequence[:, :, 10] = model.force_grid[:, None] / 8
    sequence[:, :, 11] = mu
    sequence = (sequence - model.mean[None, None]) / model.std[None, None]
    tensor = torch.as_tensor(sequence, dtype=torch.float32)
    with torch.no_grad():
        probability = np.mean(
            [torch.sigmoid(net(tensor[:, :, :10], tensor[:, 0, 10:])).numpy() for net in model.models],
            axis=0,
        )
    utility = probability * (2 - model.force_grid / 8) - 1
    index = int(np.argmax(utility))
    return {
        "selected_force_N": float(model.force_grid[index]),
        "selected_p_success": float(probability[index]),
        "selected_utility": float(utility[index]),
        "p_success": probability.tolist(),
        "utility": utility.tolist(),
    }


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def correlation(a, b, spearman=False):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if spearman:
        a, b = rankdata(a), rankdata(b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def load_motion_contexts():
    records = []
    for dataset in [OLD_DATA, NEW_DATA]:
        complete = read(dataset / "COLLECTION_COMPLETE.json")
        for entry in complete["contexts"]:
            job = Path(entry["attempt"]) / "job"
            collected = job / "COLLECTED_ROWS.json"
            if not collected.exists():
                continue
            branches = read(collected)
            feature = read(Path(branches[0]["job"]) / "original_motion_feature.json")
            records.append({
                "id": entry["context"]["id"],
                "split": entry["context"]["split"],
                "policy_seed": entry["context"].get("policy_seed"),
                "sequence": np.asarray(feature["sequence"], dtype=np.float64),
            })
    return records


def compact_lock(lock):
    return {k: lock.get(k) for k in ["state_sha256", "first_chunk_sha256"]}


def load_aligned_trace(branch: Path):
    rows = []
    with gzip.open(branch / "physics_trace.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            actor = next(a for a in row["state"]["actors"] if a["name"] == "063_tabletrashbin")
            fingers = row["contact"]["fingers"][:2]
            centers = [f["native_joint_readback"]["com_position_world"] for f in fingers]
            center = np.mean(np.asarray(centers, dtype=np.float64), axis=0)
            pos = np.asarray(actor["pose"][:3], dtype=np.float64)
            rows.append({
                "position": pos,
                "quaternion": np.asarray(actor["pose"][3:7], dtype=np.float64),
                "linear_velocity": np.asarray(actor["native_com_linear_velocity"], dtype=np.float64),
                "angular_velocity": np.asarray(actor["native_angular_velocity"], dtype=np.float64),
                "gripper_center": center,
                "relative_offset": center - pos,
                "target_force": float(row["contact"]["target_measured_squeeze_n"]),
                "measured_force": float(row["contact"]["measured_squeeze_n"]),
            })
    return rows


def first_threshold(values, threshold):
    for index, value in enumerate(values):
        if value > threshold:
            return index
    return None


def mechanism_alignment():
    job = INFERENCE / "test_mu0.675_root200002_ps50200002/job"
    branches = {}
    for force in [5.0, 6.0, 8.0]:
        branch = next(job.glob(f"branch_*_{force:g}N"))
        branches[force] = load_aligned_trace(branch)
    output = {"context": job.parent.name, "comparisons": []}
    for lower in [5.0, 6.0]:
        a, b = branches[lower], branches[8.0]
        n = min(len(a), len(b))
        translation = [norm(a[i]["position"] - b[i]["position"]) for i in range(n)]
        rotation = [quat_angle_deg(a[i]["quaternion"], b[i]["quaternion"]) for i in range(n)]
        linear_velocity = [norm(a[i]["linear_velocity"] - b[i]["linear_velocity"]) for i in range(n)]
        angular_velocity = [norm(a[i]["angular_velocity"] - b[i]["angular_velocity"]) for i in range(n)]
        gripper = [norm(a[i]["gripper_center"] - b[i]["gripper_center"]) for i in range(n)]
        relative = [norm(a[i]["relative_offset"] - b[i]["relative_offset"]) for i in range(n)]
        streak, loss8 = 0, None
        for i in range(n):
            if b[i]["target_force"] < 0.5:
                streak += 1
                if streak == 25:
                    loss8 = i - 24
                    break
            else:
                streak = 0
        before = max(0, (loss8 or n) - 1)
        output["comparisons"].append({
            "lower_force_N": lower,
            "high_force_N": 8.0,
            "aligned_steps": n,
            "first_sustained_contact_loss_step_8N": loss8,
            "first_object_translation_delta_gt_1mm": first_threshold(translation, 0.001),
            "first_object_rotation_delta_gt_1deg": first_threshold(rotation, 1.0),
            "first_object_linear_velocity_delta_gt_0.05mps": first_threshold(linear_velocity, 0.05),
            "first_object_angular_velocity_delta_gt_0.1radps": first_threshold(angular_velocity, 0.1),
            "first_gripper_center_delta_gt_1mm": first_threshold(gripper, 0.001),
            "first_relative_offset_delta_gt_1mm": first_threshold(relative, 0.001),
            "at_step_before_8N_loss": {
                "step": before,
                "object_translation_delta_m": translation[before],
                "object_rotation_delta_deg": rotation[before],
                "object_linear_velocity_delta_mps": linear_velocity[before],
                "object_angular_velocity_delta_radps": angular_velocity[before],
                "gripper_center_delta_m": gripper[before],
                "relative_offset_delta_m": relative[before],
            },
            "thresholds_are_exploratory_not_success_criteria": True,
        })
    return output


def main(output: Path):
    sys.path.insert(0, str(V4))
    from maxf8_runtime import MaxF8Feasibility

    manifest = V4 / "models_v4/feasibility/FEASIBILITY_MANIFEST.json"
    model = MaxF8Feasibility(manifest, manifest_sha256=sha(manifest))
    torch.set_num_threads(1)

    contexts = []
    trace_jobs = []
    method_order = ["ActiveForcing", "Fixed-1N", "Fixed-3N", "Fixed-5N", "Fixed-6N", "Fixed-8N"]
    for context_dir in sorted(INFERENCE.glob("test_*")):
        job = context_dir / "job"
        spec = read(context_dir / "SPEC.json")["context"]
        result = read(job / "AF_INFERENCE_RESULT.json")
        posterior = read(job / "PREACTION_POSTERIOR.json")
        decision = read(job / "PREACTION_AF_DECISION.json")
        feature = read(job / "PREACTION_FEATURE.json")
        lock = read(job / "PREACTION_SELECTION_LOCK.json")
        point_mean = point_curve(model, feature, posterior["posterior_moments"]["mean"])
        oracle = point_curve(model, feature, spec["friction"])
        full_curve = {
            "selected_force_N": float(decision["selected_force_N"]),
            "selected_p_success": float(decision["predicted_success"]),
            "selected_utility": float(decision["utility"]),
            "p_success": decision["p_success"],
            "utility": decision["expected_utility"],
        }
        branches = {int(re.match(r"branch_(\d+)_", b.name).group(1)): b for b in job.glob("branch_*")}
        if sorted(branches) != list(range(6)):
            raise ValueError(f"Unexpected branches in {job}")
        outcomes = []
        for i, outcome in enumerate(result["outcomes"]):
            if outcome["method"] != method_order[i]:
                raise ValueError("Method order drift")
            trace_jobs.append((context_dir.name, outcome["method"], str(branches[i])))
            grid_index = int(np.argmin(np.abs(np.asarray(decision["force_grid_N"]) - outcome["force_N"])))
            outcomes.append({**outcome, "predicted_p_success": float(decision["p_success"][grid_index])})
        contexts.append({
            "id": context_dir.name,
            "root": int(spec["root"]),
            "reset_id": f"root{spec['root']}:post_query_state:{lock['state_sha256']}",
            "policy_seed": int(spec["policy_seed"]),
            "simulator_seed": int(spec["root"]),
            "friction_condition": float(spec["friction"]),
            "belief": posterior_stats(posterior, spec["friction"]),
            "preaction_lock": compact_lock(lock),
            "motion_descriptor_sha256": sha(job / "PREACTION_FEATURE.json"),
            "curves": {"full_posterior": full_curve, "point_mean": point_mean, "oracle_true_mu": oracle},
            "outcomes": outcomes,
        })

    trace_rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(trace_summary, task) for task in trace_jobs]
        for future in as_completed(futures):
            trace_rows.append(future.result())
    trace_lookup = {(r["context"], r["method"]): r for r in trace_rows}

    motion_records = load_motion_contexts()
    train = [r for r in motion_records if r["split"] == "TRAIN"]
    val = [r for r in motion_records if r["split"] == "VAL"]
    test_sequences = {c["id"]: np.asarray(read(INFERENCE / c["id"] / "job/PREACTION_FEATURE.json")["sequence"], dtype=np.float64) for c in contexts}
    checkpoint = torch.load(V4 / "models_v4/feasibility/CURRENT_FULLTASK_FEAS_seed0.pt", map_location="cpu", weights_only=True)
    channel_mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float64)[:6]
    channel_std = np.asarray(checkpoint["normalization_std"], dtype=np.float64)[:6]
    train_matrix = np.stack([((r["sequence"][:, :6] - channel_mean) / channel_std).reshape(-1) for r in train])
    val_matrix = np.stack([((r["sequence"][:, :6] - channel_mean) / channel_std).reshape(-1) for r in val])
    test_ids = [c["id"] for c in contexts]
    test_matrix = np.stack([((test_sequences[i][:, :6] - channel_mean) / channel_std).reshape(-1) for i in test_ids])
    center = train_matrix.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(train_matrix - center, full_matrices=False)
    components = vt[:2]
    variance = singular_values ** 2 / max(len(train_matrix) - 1, 1)
    explained = variance[:2] / variance.sum()
    project = lambda x: (x - center) @ components.T
    train_pca, val_pca, test_pca = project(train_matrix), project(val_matrix), project(test_matrix)

    all_sequences = np.stack([r["sequence"][:, :6] for r in train])
    train_min, train_max = all_sequences.min(axis=0), all_sequences.max(axis=0)
    feature_ranges = []
    for step in range(8):
        for channel in range(6):
            values = np.asarray([test_sequences[i][step, channel] for i in test_ids])
            feature_ranges.append({
                "step": step,
                "motion_channel": channel,
                "train_min": float(train_min[step, channel]),
                "train_max": float(train_max[step, channel]),
                "test_min": float(values.min()),
                "test_max": float(values.max()),
                "test_below_train_count": int(np.sum(values < train_min[step, channel])),
                "test_above_train_count": int(np.sum(values > train_max[step, channel])),
            })

    context_metrics = []
    rollout_rows = []
    for idx, context in enumerate(contexts):
        distances = np.sqrt(np.mean((train_matrix - test_matrix[idx]) ** 2, axis=1))
        nearest_index = int(np.argmin(distances))
        predictions = [o["predicted_p_success"] for o in context["outcomes"]]
        labels = [int(o["success"]) for o in context["outcomes"]]
        absolute_errors = [abs(p - y) for p, y in zip(predictions, labels)]
        classes = ["FP" if p >= 0.5 and y == 0 else "FN" if p < 0.5 and y == 1 else "correct" for p, y in zip(predictions, labels)]
        metric = {
            "id": context["id"],
            "policy_seed": context["policy_seed"],
            "nearest_train_motion_standardized_rms": float(distances[nearest_index]),
            "nearest_train_id": train[nearest_index]["id"],
            "max_abs_motion_z": float(np.max(np.abs(test_matrix[idx]))),
            "motion_elements_outside_train_range": int(np.sum((test_sequences[context["id"]][:, :6] < train_min) | (test_sequences[context["id"]][:, :6] > train_max))),
            "pca1": float(test_pca[idx, 0]),
            "pca2": float(test_pca[idx, 1]),
            "mean_absolute_prediction_error": float(np.mean(absolute_errors)),
            "brier": float(np.mean([(p - y) ** 2 for p, y in zip(predictions, labels)])),
            "false_positive_count": classes.count("FP"),
            "false_negative_count": classes.count("FN"),
        }
        context_metrics.append(metric)
        belief = context["belief"]
        for outcome, classification in zip(context["outcomes"], classes):
            trace = trace_lookup[(context["id"], outcome["method"])]
            rollout_rows.append({
                "context": context["id"],
                "root": context["root"],
                "reset_id": context["reset_id"],
                "policy_seed": context["policy_seed"],
                "simulator_seed": context["simulator_seed"],
                "friction_condition": context["friction_condition"],
                "true_mu": belief["true_mu"],
                "estimated_mu": belief["posterior_mean"],
                "posterior_std": belief["posterior_std"],
                "posterior_q05": belief["q05"],
                "posterior_q95": belief["q95"],
                "method": outcome["method"],
                "commanded_force_N": float(outcome["force_N"]),
                "predicted_p_success": float(outcome["predicted_p_success"]),
                "prediction_class_at_0.5": classification,
                "success": int(outcome["success"]),
                "motion_nn_distance": metric["nearest_train_motion_standardized_rms"],
                "motion_pca1": metric["pca1"],
                "motion_pca2": metric["pca2"],
                **{k: v for k, v in trace.items() if k not in ["context", "method"]},
            })

    distances = [m["nearest_train_motion_standardized_rms"] for m in context_metrics]
    context_mae = [m["mean_absolute_prediction_error"] for m in context_metrics]
    rollout_distance = [r["motion_nn_distance"] for r in rollout_rows]
    rollout_error = [abs(r["predicted_p_success"] - r["success"]) for r in rollout_rows]
    ood_error_association = {
        "context_n": len(context_metrics),
        "effective_unique_policy_seeds": len(set(m["policy_seed"] for m in context_metrics)),
        "context_pearson_nn_vs_mae": correlation(distances, context_mae),
        "context_spearman_nn_vs_mae": correlation(distances, context_mae, spearman=True),
        "rollout_n_nonindependent": len(rollout_rows),
        "rollout_pearson_nn_vs_abs_error": correlation(rollout_distance, rollout_error),
        "rollout_spearman_nn_vs_abs_error": correlation(rollout_distance, rollout_error, spearman=True),
        "by_policy_seed": {},
        "caveat": "Eight contexts contain only two downstream policy seeds; 48 rollout rows are not independent motion samples.",
    }
    for seed in sorted(set(m["policy_seed"] for m in context_metrics)):
        subset = [m for m in context_metrics if m["policy_seed"] == seed]
        ood_error_association["by_policy_seed"][str(seed)] = {
            "contexts": len(subset),
            "mean_nn_distance": float(np.mean([m["nearest_train_motion_standardized_rms"] for m in subset])),
            "mean_absolute_prediction_error": float(np.mean([m["mean_absolute_prediction_error"] for m in subset])),
            "mean_brier": float(np.mean([m["brier"] for m in subset])),
            "false_positive_count": int(sum(m["false_positive_count"] for m in subset)),
            "false_negative_count": int(sum(m["false_negative_count"] for m in subset)),
        }

    paired = []
    for context in contexts:
        by_method = {o["method"]: o for o in context["outcomes"]}
        paired.append({
            "context": context["id"],
            "af_success": int(by_method["ActiveForcing"]["success"]),
            "fixed8_success": int(by_method["Fixed-8N"]["success"]),
            "af_force_N": float(context["curves"]["full_posterior"]["selected_force_N"]),
            "delta_force_vs_fixed8_N": 8.0 - float(context["curves"]["full_posterior"]["selected_force_N"]),
        })

    duplicate_groups = {}
    for row in rollout_rows:
        duplicate_groups.setdefault(row["trace_sha256"], []).append({"context": row["context"], "method": row["method"]})
    exact_duplicates = [v for v in duplicate_groups.values() if len(v) > 1]

    cross_motion_pairs = []
    for mu in sorted(set(c["friction_condition"] for c in contexts)):
        pair = [c for c in contexts if c["friction_condition"] == mu]
        if len(pair) != 2:
            raise ValueError("Expected two policy seeds per friction")
        cross_motion_pairs.append({
            "true_mu": mu,
            "same_preaction_state": pair[0]["preaction_lock"]["state_sha256"] == pair[1]["preaction_lock"]["state_sha256"],
            "different_first_chunk": pair[0]["preaction_lock"]["first_chunk_sha256"] != pair[1]["preaction_lock"]["first_chunk_sha256"],
            "posterior_exactly_equal": pair[0]["belief"] == pair[1]["belief"],
            "policy_seeds": [p["policy_seed"] for p in pair],
            "selected_forces_N": [p["curves"]["full_posterior"]["selected_force_N"] for p in pair],
        })

    mechanism = read(MECHANISM / "STAGE_A_COMPLETE.json")
    evidence = {
        "scope": "Read-only original-log reconstruction plus completed declared post-hoc development intervention; no model, label, criterion, or source experiment modified.",
        "source_experiments": {"v4": str(V4), "inference": str(INFERENCE), "mechanism": str(MECHANISM)},
        "counts": {
            "rollout_rows": len(rollout_rows),
            "decision_contexts": len(contexts),
            "independent_probe_cases": len(set(c["friction_condition"] for c in contexts)),
            "unique_policy_seeds": len(set(c["policy_seed"] for c in contexts)),
            "training_contexts": len(train),
            "validation_contexts": len(val),
            "development_intervention_rollouts": mechanism["paired_rollouts"],
        },
        "data_quality": {
            "primary_key_unique": len({(r["context"], r["method"]) for r in rollout_rows}) == len(rollout_rows),
            "all_trace_hashes_verified": all(r["trace_sha256"] for r in rollout_rows),
            "exact_duplicate_trace_groups": exact_duplicates,
            "all_contexts_six_methods": all(len(c["outcomes"]) == 6 for c in contexts),
            "matched_comparison_status": "Same post-query state, simulator root, policy seed, deterministic policy and controller within each context; only force branch changes at intervention. Later closed-loop actions may diverge as observations diverge.",
        },
        "paired_af_fixed8": paired,
        "paired_counts": {
            "both_success": sum(r["af_success"] and r["fixed8_success"] for r in paired),
            "af_only": sum(r["af_success"] and not r["fixed8_success"] for r in paired),
            "fixed8_only": sum(not r["af_success"] and r["fixed8_success"] for r in paired),
            "both_failure": sum(not r["af_success"] and not r["fixed8_success"] for r in paired),
        },
        "contexts": contexts,
        "motion": {
            "feature_ranges_48": feature_ranges,
            "pca_explained_variance_ratio": explained.tolist(),
            "train_pca": [{"id": r["id"], "policy_seed": r["policy_seed"], "pca1": float(x[0]), "pca2": float(x[1])} for r, x in zip(train, train_pca)],
            "val_pca": [{"id": r["id"], "policy_seed": r["policy_seed"], "pca1": float(x[0]), "pca2": float(x[1])} for r, x in zip(val, val_pca)],
            "test_context_metrics": context_metrics,
            "ood_error_association": ood_error_association,
            "cross_motion_isolation": cross_motion_pairs,
        },
        "mechanism_alignment": mechanism_alignment(),
        "intermediate_force_replication": mechanism,
        "source_hashes": {
            "final_results": sha(V4 / "FINAL_INFERENCE_RESULTS.json"),
            "final_audit": sha(V4 / "INDEPENDENT_FINAL_RESULT_AUDIT.json"),
            "mechanism_complete": sha(MECHANISM / "STAGE_A_COMPLETE.json"),
            "feasibility_manifest": sha(manifest),
        },
    }
    write(output / "NEXT_STAGE_EVIDENCE.json", evidence)

    csv_columns = [
        "context", "root", "reset_id", "policy_seed", "simulator_seed", "friction_condition", "true_mu",
        "estimated_mu", "posterior_std", "posterior_q05", "posterior_q95", "method", "commanded_force_N",
        "measured_force_mean_N", "measured_force_first1000_mean_N", "predicted_p_success", "prediction_class_at_0.5",
        "success", "diagnostic_lift_reached_1m", "sustained_contact_loss", "first_sustained_contact_loss_step",
        "contact_fraction", "object_initial_position_xyz", "object_final_position_xyz", "object_max_z_m",
        "object_max_translation_from_initial_m", "object_max_linear_speed_mps", "object_max_angular_speed_radps",
        "motion_nn_distance", "motion_pca1", "motion_pca2", "motion_descriptor_sha256", "trajectory_path",
    ]
    with (output / "AUTHORITATIVE_ROLLOUT_TABLE.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rollout_rows, key=lambda r: (r["context"], method_order.index(r["method"]))):
            writer.writerow(row)

    with (output / "DECISION_CURVES.csv").open("w", newline="") as stream:
        columns = ["context", "variant", "force_N", "predicted_p_success", "utility"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for context in contexts:
            force_grid = read(INFERENCE / context["id"] / "job/PREACTION_AF_DECISION.json")["force_grid_N"]
            for variant, curve in context["curves"].items():
                for force, probability, utility in zip(force_grid, curve["p_success"], curve["utility"]):
                    writer.writerow({"context": context["id"], "variant": variant, "force_N": force, "predicted_p_success": probability, "utility": utility})

    with (output / "MOTION_PCA.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "id", "policy_seed", "pca1", "pca2", "prediction_status"])
        writer.writeheader()
        for split, rows in [("TRAIN", evidence["motion"]["train_pca"]), ("VAL", evidence["motion"]["val_pca"])]:
            for row in rows:
                writer.writerow({"split": split, **row, "prediction_status": "not_test"})
        for row in context_metrics:
            status = "has_FP" if row["false_positive_count"] else "has_FN" if row["false_negative_count"] else "correct_only"
            writer.writerow({"split": "TEST", "id": row["id"], "policy_seed": row["policy_seed"], "pca1": row["pca1"], "pca2": row["pca2"], "prediction_status": status})

    print(json.dumps({"completed": True, "output": str(output), "counts": evidence["counts"], "paired_counts": evidence["paired_counts"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
