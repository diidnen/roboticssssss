"""Filtered grasp-geometry canaries. Original squeeze inner unchanged."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ISO = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
sys.path[:0] = [str(ISO), str(V4), str(REPO), str(ROOT), str(HERE)]

from envs.utils import ArmTag
from patch_grasp_surface import apply_grasp_surface_materials
from run_liftstyle_context import PrefixSnapshot, build_hold_feature, scripted_establish_grasp, sequence_hash, write
from run_sweep_context import labels_from_metrics, make_task
from force_realize import capture_stock_drives, realize_commanded_force, restore_stock_drives, uninstall_lift_squeeze
from collision_filter import apply_retention_collision_filters
from grasp_geom_logger import GraspGeomLogger
from plot_geom_timelines import plot_seven_panel, plot_overlay_425, plot_overlay_575, plot_slab_maps

SEED = 200014
CANARIES = {
    0.425: [0.50, 0.75, 1.00],
    0.575: [1.00, 1.25, 1.50],
    0.85: [0.50, 1.00],
}


def remainder_phased(task) -> None:
    place = ArmTag("left")
    task._af_diag_phase = "lift"
    task.move(task.move_by_displacement(arm_tag=place, z=0.08, move_axis="arm"))
    task._af_diag_phase = "wrist-rotation-start"
    task.move(task.pour_actions)
    task._af_diag_phase = "pour-start"
    task.move(task.pour_actions)
    task._af_diag_phase = "pour"
    task.move(task.pour_actions)
    task._af_diag_phase = "terminal"
    task.delay(6)


def run_one(task, force: float, friction: float, stock, out: Path) -> dict:
    apply_grasp_surface_materials(task, friction)
    apply_retention_collision_filters(task)
    task.plan_success = True
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    logger = GraspGeomLogger(task, out, force)
    logger.install()
    task._af_diag_phase = "settle"
    realize = realize_commanded_force(task, force, stock)
    logger.save_frame("settle_end")
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    task.activate_activeforcing_candidate_force()
    restore_stock_drives(stock)
    remainder_phased(task)
    uninstall_lift_squeeze(task)
    metrics = task.compute_activeforcing_dynamic_metrics()
    labels = labels_from_metrics(task, metrics)
    logger.detach()
    dumped = logger.finish(int(labels["retention_success"]))
    plot_seven_panel(out)
    plot_slab_maps(out)
    result = {
        "force_N": float(force),
        "friction": float(friction),
        "retention_success": int(labels["retention_success"]),
        "contact_ratio": labels.get("contact_ratio"),
        "drop": labels.get("drop"),
        "measured_force_mean_n": labels.get("measured_force_mean_n"),
        "deskbin_z": labels.get("deskbin_z"),
        "realize": realize,
        "event_order": dumped.get("event_order"),
        "flags": dumped.get("flags"),
    }
    write(out / "result.json", result)
    print(json.dumps({"done": f"mu{friction:g}_F{force:g}", "retention": result["retention_success"], "event_order": result["event_order"]}), flush=True)
    return result


def run_mu(friction: float, forces: list[float], out_root: Path) -> dict:
    task = None
    stock = None
    grasp_error = None
    for attempt in range(3):
        if task is not None:
            try:
                task.close_env()
            except Exception:
                pass
        task = make_task(SEED, friction, attempt)
        stock = capture_stock_drives(task)
        apply_retention_collision_filters(task)
        apply_grasp_surface_materials(task, friction)
        task.activate_activeforcing_candidate_force()
        try:
            scripted_establish_grasp(task)
            grasp_error = None
            break
        except Exception as exc:
            grasp_error = exc
    if grasp_error is not None:
        raise grasp_error
    for _ in range(20):
        task.scene.step()
    apply_retention_collision_filters(task)
    pre_query = PrefixSnapshot(task)
    query = None
    for _ in range(3):
        try:
            query = task.run_activeforcing_query(query_force_n=4.0, displacement_m=0.012)
            break
        except RuntimeError as exc:
            if "planning failed" not in repr(exc):
                raise
            pre_query.restore()
            apply_grasp_surface_materials(task, friction)
            apply_retention_collision_filters(task)
    if query is None:
        raise RuntimeError("query failed")
    feature = build_hold_feature(task)
    feature_sha = sequence_hash(feature["sequence"])
    snapshot = PrefixSnapshot(task)
    outcomes = []
    try:
        for force in forces:
            snapshot.restore()
            apply_grasp_surface_materials(task, friction)
            apply_retention_collision_filters(task)
            branch = out_root / f"mu{friction:g}_F{force:g}"
            branch.mkdir(parents=True, exist_ok=True)
            result = run_one(task, force, friction, stock, branch)
            result["feature_sha256"] = feature_sha
            outcomes.append(result)
    finally:
        try:
            task.close_env()
        except Exception:
            pass
    payload = {
        "seed": SEED,
        "friction": friction,
        "feature_sha256": feature_sha,
        "query_contact_ratio": query.get("contact_ratio"),
        "outcomes": outcomes,
        "collision_filter": "fingers ignore inner/bottom; fl_link6 ignores deskbin",
    }
    write(out_root / f"mu{friction:g}_CONTEXT.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "geom_runs")
    parser.add_argument("--mu", type=float, action="append")
    args = parser.parse_args()
    if args.mu is None and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    selected = args.mu or list(CANARIES)
    jobs = []
    for mu in selected:
        payload = run_mu(float(mu), CANARIES[float(mu)], args.out)
        jobs.append({"mu": float(mu), "feature_sha256": payload["feature_sha256"]})
    plot_overlay_425(args.out)
    plot_overlay_575(args.out)
    from analyze_geom import main as analyze

    analyze(args.out)
    write(args.out / "INDEX.json", {"seed": SEED, "jobs": jobs})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
