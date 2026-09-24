"""Lift-style dump sanity sweep with grasp-surface-only friction.

Not official 18/19. Not a retrain. Not Fig.B.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
sys.path[:0] = [str(HERE), str(V4), str(REPO)]

from patch_grasp_surface import apply_grasp_surface_materials, contact_material_audit, install_patches
from run_liftstyle_context import (
    GRASP_FORCE_N,
    QUERY_FORCE_N,
    PrefixSnapshot,
    build_hold_feature,
    now,
    scripted_dump_remainder,
    scripted_establish_grasp,
    sequence_hash,
    write,
)

FIXED_MU = 0.3
FINGER_MU = 0.3


def mu_eff(object_side: float) -> float:
    return 0.5 * (FINGER_MU + float(object_side))


def ball_band_count(task) -> int:
    count = 0
    zs = []
    for sphere in task.sphere_lst:
        z = float(sphere.get_pose().p[2])
        zs.append(z)
        if 0.13 <= z <= 0.25:
            count += 1
    return count, zs


def make_task(seed: int, friction: float, episode_id: int = 0):
    install_patches(HERE / "collision_splits")
    from scripts.eval_policy_xpolicylab import class_decorator, load_task_args

    user_args = {
        "task_name": "dump_bin_bigbin",
        "task_config": "demo_clean",
        "policy_name": "expert",
        "ckpt_setting": "grasp-surface-friction-only",
        "activeforcing_enabled": True,
        "af_dynamic_evaluator": True,
        "af_contact_friction": float(friction),
        "af_force_limit_n": float(GRASP_FORCE_N),
    }
    task_args, _ = load_task_args(user_args)
    task_args["eval_mode"] = True
    task_args["render_freq"] = 0
    task = class_decorator("dump_bin_bigbin")
    task.setup_demo(now_ep_num=episode_id, seed=int(seed), is_test=True, **task_args)
    return task


def labels_from_metrics(task, metrics: dict) -> dict:
    official = bool(task.check_success()) and bool(task.plan_success)
    contact_ratio = float(metrics.get("contact_ratio") or 0.0)
    irrecoverable = bool(metrics.get("irrecoverable_failure"))
    n_balls, ball_z = ball_band_count(task)
    pose = np.asarray(task.deskbin.get_pose().p, dtype=float)
    quat = np.asarray(task.deskbin.get_pose().q, dtype=float)
    retention = bool(contact_ratio >= 0.8 and not irrecoverable)
    trace = list(getattr(task, "_af_force_trace", []) or [])
    squeeze = np.asarray(
        [max(float(sample["left_force_n"]), float(sample["right_force_n"])) for sample in trace],
        dtype=float,
    ) if trace else np.zeros(0)
    left_n = np.asarray([float(sample["left_force_n"]) for sample in trace], dtype=float) if trace else np.zeros(0)
    right_n = np.asarray([float(sample["right_force_n"]) for sample in trace], dtype=float) if trace else np.zeros(0)
    bilateral = np.asarray(
        [bool(sample["left_bilateral"] or sample["right_bilateral"]) for sample in trace],
        dtype=bool,
    ) if trace else np.zeros(0, dtype=bool)
    return {
        "official_full_task_success": int(official),
        "retention_success": int(retention),
        "contact_ratio": contact_ratio,
        "persistent_contact": bool(contact_ratio >= 0.8),
        "irrecoverable_failure": irrecoverable,
        "drop": int(irrecoverable or contact_ratio < 0.5),
        "n_balls_in_official_band": int(n_balls),
        "ball_z": ball_z,
        "deskbin_xyz": pose.tolist(),
        "deskbin_xyzw": quat.tolist(),
        "deskbin_z": float(pose[2]),
        "deskbin_z_ge_1": bool(pose[2] >= 1.0),
        "measured_force_mean_n": metrics.get("measured_force_mean_n"),
        "measured_force_min_n": float(np.min(squeeze)) if len(squeeze) else 0.0,
        "measured_force_p95_n": metrics.get("measured_force_p95_n"),
        "measured_left_mean_n": float(np.mean(left_n)) if len(left_n) else 0.0,
        "measured_right_mean_n": float(np.mean(right_n)) if len(right_n) else 0.0,
        "bilateral_fraction": float(np.mean(bilateral)) if len(bilateral) else 0.0,
        "left_force_limit_n": metrics.get("left_force_limit_n"),
        "right_force_limit_n": metrics.get("right_force_limit_n"),
        "samples": metrics.get("samples"),
    }


def run_remainder(task, force_n: float, friction: float, feature: dict, feature_sha: str, seed: int) -> dict:
    task.plan_success = True
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    task.af_force_limit_n = float(force_n)
    apply_grasp_surface_materials(task, friction)
    task.activate_activeforcing_candidate_force()
    scripted_dump_remainder(task)
    metrics = task.compute_activeforcing_dynamic_metrics()
    labels = labels_from_metrics(task, metrics)
    result = {
        "completed": True,
        "success": int(labels["official_full_task_success"]),
        "plan_success": bool(task.plan_success),
        "official_final_check": bool(task.check_success()),
        "force_setpoint_single_finger_N": float(force_n),
        "grasp_force_N": GRASP_FORCE_N,
        "query_force_N": QUERY_FORCE_N,
        "friction_object_side_grasp": float(friction),
        "friction_object_side_inner_bottom": FIXED_MU,
        "friction_finger": FINGER_MU,
        "mu_eff_finger_grasp_average": mu_eff(friction),
        "seed": int(seed),
        "feature": feature,
        "feature_sha256": feature_sha,
        "motion_mode": "liftstyle_snapshot_fork_remainder_grasp_surface_only",
        "finished_utc": now(),
    }
    result.update(labels)
    return result


def run_context(seed: int, friction: float, forces: list[float], episode_id: int = 0) -> dict:
    task = None
    try:
        task = make_task(seed, friction, episode_id)
        if float(task.af_contact_friction) != float(friction):
            raise RuntimeError("Friction not applied")
        assignment = apply_grasp_surface_materials(task, friction)
        task.activate_activeforcing_candidate_force()
        scripted_establish_grasp(task)
        for _ in range(20):
            task.scene.step()
        contact_audit = contact_material_audit(task)
        pre_query = PrefixSnapshot(task)
        query = None
        query_error = None
        for attempt in range(3):
            try:
                query = task.run_activeforcing_query(query_force_n=QUERY_FORCE_N, displacement_m=0.012)
                query_error = None
                break
            except RuntimeError as exc:
                query_error = repr(exc)
                if "planning failed" not in query_error:
                    raise
                pre_query.restore()
                apply_grasp_surface_materials(task, friction)
        if query is None:
            query = {
                "contact_ratio": None,
                "final_bilateral_contact": None,
                "query_failed": True,
                "error": query_error,
            }
        feature = build_hold_feature(task)
        sequence = np.asarray(feature["sequence"], np.float32)
        if sequence.shape != (8, 64) or not np.isfinite(sequence).all():
            raise RuntimeError(f"Bad feature shape {sequence.shape}")
        feature_sha = sequence_hash(feature["sequence"])
        snapshot = PrefixSnapshot(task)
        branches = []
        for force in forces:
            snapshot.restore()
            apply_grasp_surface_materials(task, friction)
            try:
                result = run_remainder(task, force, friction, feature, feature_sha, seed)
                result["query_contact_ratio"] = query.get("contact_ratio")
                result["query_final_bilateral_contact"] = query.get("final_bilateral_contact")
            except Exception as exc:
                unstable = type(exc).__name__ == "UnStableError" or "UnStableError" in repr(exc)
                if not unstable:
                    raise
                result = {
                    "completed": True,
                    "success": 0,
                    "official_full_task_success": 0,
                    "retention_success": 0,
                    "unstable_layout": True,
                    "error": repr(exc),
                    "force_setpoint_single_finger_N": float(force),
                    "friction_object_side_grasp": float(friction),
                    "seed": int(seed),
                    "feature": feature,
                    "feature_sha256": feature_sha,
                    "motion_mode": "liftstyle_snapshot_fork_remainder_grasp_surface_only",
                    "finished_utc": now(),
                }
            branches.append(result)
        return {
            "completed": True,
            "feature": feature,
            "feature_sha256": feature_sha,
            "deskbin_id": int(task.deskbin_id),
            "material_assignment": assignment,
            "contact_material_audit_post_grasp": contact_audit,
            "query_contact_ratio": query.get("contact_ratio"),
            "query_failed": bool(query.get("query_failed")),
            "query_error": query.get("error"),
            "branches": branches,
        }
    finally:
        if task is not None:
            try:
                task.close_env()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    context = json.loads(args.context.read_text())["context"]
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    write(out / "CONTEXT.json", context)
    packed = run_context(
        seed=int(context["seed"]),
        friction=float(context["friction"]),
        forces=[float(x) for x in context["forces_N"]],
        episode_id=0,
    )
    feature = packed["feature"]
    feature_sha = packed["feature_sha256"]
    write(out / "SHARED_PREACTION_FEATURE.json", feature)
    write(out / "MATERIAL_ASSIGNMENT.json", packed["material_assignment"])
    write(out / "CONTACT_MATERIAL_AUDIT.json", packed["contact_material_audit_post_grasp"])
    outcomes = []
    for index, result in enumerate(packed["branches"]):
        force = float(result["force_setpoint_single_finger_N"])
        branch = out / f"branch_{index}_{force:g}N"
        branch.mkdir()
        write(branch / "result.json", result)
        write(branch / "PREACTION_FEATURE.json", feature)
        outcomes.append(
            {
                "method": f"GraspSurface-{force:g}N",
                "force_N": force,
                "success": int(result.get("official_full_task_success") or result.get("success") or 0),
                "official_full_task_success": int(result.get("official_full_task_success") or 0),
                "retention_success": int(result.get("retention_success") or 0),
                "unstable_layout": bool(result.get("unstable_layout")),
                "feature_sha256": feature_sha,
                "contact_ratio": result.get("contact_ratio"),
                "persistent_contact": result.get("persistent_contact"),
                "measured_force_mean_n": result.get("measured_force_mean_n"),
                "measured_force_min_n": result.get("measured_force_min_n"),
                "bilateral_fraction": result.get("bilateral_fraction"),
                "irrecoverable_failure": result.get("irrecoverable_failure"),
                "n_balls_in_official_band": result.get("n_balls_in_official_band"),
                "deskbin_z": result.get("deskbin_z"),
                "drop": result.get("drop"),
            }
        )
    write(
        out / "GRASP_SURFACE_CONTEXT_RESULT.json",
        {
            "completed": True,
            "context": context,
            "deskbin_id": packed["deskbin_id"],
            "outcomes": outcomes,
            "paired_rollouts": len(outcomes),
            "shared_feature_sha256": feature_sha,
            "contact_material_audit_post_grasp": packed["contact_material_audit_post_grasp"],
            "query_failed": packed.get("query_failed"),
            "query_error": packed.get("query_error"),
            "claim_boundary": "grasp-surface-only friction isolation; lift-style remainder; not official 18/19; not retrain",
            "finished_utc": now(),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
