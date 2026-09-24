"""Tiny contact-attribution canaries. Does not change original squeeze inner."""
from __future__ import annotations

import argparse
import json
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
from patch_grasp_surface import apply_grasp_surface_materials, contact_material_audit
from run_liftstyle_context import (
    PrefixSnapshot,
    build_hold_feature,
    scripted_establish_grasp,
    sequence_hash,
    write,
)
from run_sweep_context import labels_from_metrics, make_task
from force_realize import capture_stock_drives, realize_commanded_force, restore_stock_drives, uninstall_lift_squeeze
from contact_sampler import ContactLogger
from plot_timelines import plot_timeline

CANARIES = {
    0.575: [0.50, 1.00, 1.50],
    0.85: [0.75, 1.50, 3.00],
    0.425: [0.50, 1.00],
}
SEED = 200014


def remainder_phased(task) -> None:
    place = ArmTag("left")
    task._af_diag_phase = "lift"
    task.move(task.move_by_displacement(arm_tag=place, z=0.08, move_axis="arm"))
    task._af_diag_phase = "pour-start"
    for index in range(3):
        if index == 1:
            task._af_diag_phase = "pour"
        task.move(task.pour_actions)
    task._af_diag_phase = "terminal"
    task.delay(6)


def run_one(task, force: float, friction: float, stock, out: Path) -> dict:
    task.plan_success = True
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    apply_grasp_surface_materials(task, friction)
    logger = ContactLogger(task, out, force)
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
    logger.save_frame("terminal")
    if not any(sample["bilateral"] for sample in logger.steps if sample["phase"] in {"pour", "terminal"}):
        last = logger.steps[-1] if logger.steps else {}
        logger.events.append(
            {
                "step": last.get("i"),
                "t": last.get("t"),
                "event": "retention_loss",
                "phase": last.get("phase"),
                "squeeze": last.get("squeeze"),
            }
        )
        logger.flags["retention_loss"] = {"step": last.get("i"), "t": last.get("t")}
    uninstall_lift_squeeze(task)
    logger.detach()
    dumped = logger.dump()
    plot_timeline(out)
    metrics = task.compute_activeforcing_dynamic_metrics()
    labels = labels_from_metrics(task, metrics)
    result = {
        "force_N": float(force),
        "friction": float(friction),
        "retention_success": int(labels["retention_success"]),
        "contact_ratio": labels.get("contact_ratio"),
        "drop": labels.get("drop"),
        "measured_force_mean_n": labels.get("measured_force_mean_n"),
        "realize": realize,
        "log": dumped,
        "deskbin_z": labels.get("deskbin_z"),
    }
    write(out / "result.json", result)
    return result


def run_mu(friction: float, forces: list[float], out_root: Path) -> dict:
    task = make_task(SEED, friction, 0)
    stock = capture_stock_drives(task)
    apply_grasp_surface_materials(task, friction)
    task.activate_activeforcing_candidate_force()
    grasp_error = None
    for attempt in range(3):
        try:
            scripted_establish_grasp(task)
            grasp_error = None
            break
        except Exception as exc:
            grasp_error = exc
            try:
                task.close_env()
            except Exception:
                pass
            task = make_task(SEED, friction, attempt + 1)
            stock = capture_stock_drives(task)
            apply_grasp_surface_materials(task, friction)
            task.activate_activeforcing_candidate_force()
    if grasp_error is not None:
        raise grasp_error
    for _ in range(20):
        task.scene.step()
    audit = contact_material_audit(task)
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
            branch = out_root / f"mu{friction:g}_F{force:g}"
            branch.mkdir(parents=True, exist_ok=False)
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
        "contact_audit": audit,
        "outcomes": outcomes,
    }
    write(out_root / f"mu{friction:g}_CONTEXT.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    parser.add_argument("--mu", type=float, action="append")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selected = args.mu or list(CANARIES)
    all_payloads = []
    for mu in selected:
        forces = CANARIES[float(mu)]
        payload = run_mu(float(mu), forces, args.out)
        all_payloads.append({"mu": float(mu), "forces": forces, "feature_sha256": payload["feature_sha256"]})
    write(args.out / "INDEX.json", {"seed": SEED, "jobs": all_payloads})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
