"""Force-realize dump sweep: commanded F must be actual squeeze before remainder.

Does not overwrite official 18/19, Fig.B, checkpoints, or the previous
grasp-surface isolation directory. Official dump success is recorded but is
not the Coulomb target.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ISO = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
sys.path[:0] = [str(ISO), str(V4), str(REPO)]

from patch_grasp_surface import apply_grasp_surface_materials, contact_material_audit
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
from run_sweep_context import FIXED_MU, FINGER_MU, labels_from_metrics, make_task, mu_eff

sys.path.insert(0, str(HERE))
from force_realize import (
    capture_stock_drives,
    install_lift_squeeze,
    realize_commanded_force,
    restore_stock_drives,
    uninstall_lift_squeeze,
)


def run_remainder(task, force_n: float, friction: float, feature: dict, feature_sha: str, seed: int, stock) -> dict:
    task.plan_success = True
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    apply_grasp_surface_materials(task, friction)
    realize = realize_commanded_force(task, force_n, stock)
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    task.activate_activeforcing_candidate_force()
    restore_stock_drives(stock)
    scripted_dump_remainder(task)
    uninstall_lift_squeeze(task)
    metrics = task.compute_activeforcing_dynamic_metrics()
    labels = labels_from_metrics(task, metrics)
    mean = float(metrics.get("measured_force_mean_n") or 0.0)
    rel = abs(mean - float(force_n)) / max(float(force_n), 1e-6)
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
        "motion_mode": "lift_original_squeeze_inner_then_remainder",
        "remainder_squeeze_rel_error": rel,
        "remainder_force_tracked": bool(rel <= 0.35 and float(labels.get("contact_ratio") or 0.0) >= 0.8),
        "finished_utc": now(),
    }
    result.update(labels)
    result.update({f"realize_{key}": value for key, value in realize.items()})
    return result


def run_context(seed: int, friction: float, forces: list[float], episode_id: int = 0) -> dict:
    task = None
    try:
        task = make_task(seed, friction, episode_id)
        stock = capture_stock_drives(task)
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
                result = run_remainder(task, force, friction, feature, feature_sha, seed, stock)
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
                    "motion_mode": "lift_original_squeeze_inner_then_remainder",
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
                "method": f"ForceRealize-{force:g}N",
                "force_N": force,
                "success": int(result.get("official_full_task_success") or 0),
                "official_full_task_success": int(result.get("official_full_task_success") or 0),
                "retention_success": int(result.get("retention_success") or 0),
                "force_realized_before_motion": bool(result.get("realize_force_realized_before_motion")),
                "pre_motion_squeeze_mean_n": result.get("realize_pre_motion_squeeze_mean_n"),
                "remainder_force_tracked": bool(result.get("remainder_force_tracked")),
                "measured_force_mean_n": result.get("measured_force_mean_n"),
                "contact_ratio": result.get("contact_ratio"),
                "drop": result.get("drop"),
                "n_balls_in_official_band": result.get("n_balls_in_official_band"),
                "deskbin_z": result.get("deskbin_z"),
                "unstable_layout": bool(result.get("unstable_layout")),
                "feature_sha256": feature_sha,
            }
        )
    write(
        out / "FORCE_REALIZE_CONTEXT_RESULT.json",
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
            "claim_boundary": (
                "lift/Tabero original squeeze inner on dump remainder; F is a force "
                "reference not an actuator cap; grasp-surface friction isolation; "
                "retention is the Coulomb target; official dump is recorded separately; "
                "not official 18/19; not retrain; not 32x20"
            ),
            "finished_utc": now(),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
