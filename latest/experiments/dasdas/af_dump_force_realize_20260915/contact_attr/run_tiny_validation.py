"""Tiny post-filter validation. Does not change original squeeze inner."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ISO = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
sys.path[:0] = [str(ISO), str(V4), str(REPO), str(ROOT), str(HERE)]

from patch_grasp_surface import apply_grasp_surface_materials, contact_material_audit
from run_liftstyle_context import PrefixSnapshot, build_hold_feature, scripted_establish_grasp, sequence_hash, write
from run_sweep_context import labels_from_metrics, make_task
from force_realize import capture_stock_drives, realize_commanded_force, restore_stock_drives, uninstall_lift_squeeze
from collision_filter import apply_retention_collision_filters
from contact_sampler import sample_contacts, deskbin_shape_index
from run_contact_canary import remainder_phased

SEED = 200014
GRID = {
    0.425: [0.50, 0.75, 1.00],
    0.575: [0.75, 1.00, 1.25, 1.50],
    0.85: [0.50, 0.75, 1.00, 1.25],
}


def unintended_snapshot(task, shape_map, dt) -> dict:
    pairs = sample_contacts(task, shape_map, dt)
    grouped = defaultdict(float)
    for pair in pairs:
        grouped[pair["kind"]] += pair["fn"]
    finger_inner = sum(pair["fn"] for pair in pairs if pair["kind"].startswith("finger-inner"))
    palm = sum(pair["fn"] for pair in pairs if pair["kind"].startswith("robot_nonfinger"))
    finger_grasp = sum(pair["fn"] for pair in pairs if pair["kind"].startswith("finger-grasp"))
    return {
        "finger_inner_n": float(finger_inner),
        "finger_grasp_n": float(finger_grasp),
        "palm_n": float(palm),
        "kinds": {key: float(value) for key, value in grouped.items()},
        "n_pairs": len(pairs),
    }


def run_one(task, force, friction, stock, shape_map) -> dict:
    dt = float(getattr(task, "physics_timestep", 1.0 / 250.0))
    apply_grasp_surface_materials(task, friction)
    task.plan_success = True
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    task._af_diag_phase = "settle"
    realize = realize_commanded_force(task, force, stock)
    settle_unint = unintended_snapshot(task, shape_map, dt)
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
    terminal_unint = unintended_snapshot(task, shape_map, dt)
    return {
        "mu": float(friction),
        "F_cmd": float(force),
        "settle_squeeze": realize.get("pre_motion_squeeze_mean_n"),
        "settle_realized": bool(realize.get("force_realized_before_motion")),
        "pour_squeeze": labels.get("measured_force_mean_n"),
        "retention": int(labels.get("retention_success") or 0),
        "contact_ratio": labels.get("contact_ratio"),
        "drop": labels.get("drop"),
        "deskbin_z": labels.get("deskbin_z"),
        "settle_finger_inner_n": settle_unint["finger_inner_n"],
        "settle_finger_grasp_n": settle_unint["finger_grasp_n"],
        "settle_palm_n": settle_unint["palm_n"],
        "terminal_finger_inner_n": terminal_unint["finger_inner_n"],
        "terminal_palm_n": terminal_unint["palm_n"],
        "unintended_contact": bool(settle_unint["finger_inner_n"] > 0.05 or settle_unint["palm_n"] > 0.05),
        "settle_kinds": settle_unint["kinds"],
    }


def run_mu(friction: float, forces: list[float]) -> dict:
    task = make_task(SEED, friction, 0)
    stock = capture_stock_drives(task)
    filt = apply_retention_collision_filters(task)
    apply_grasp_surface_materials(task, friction)
    task.activate_activeforcing_candidate_force()
    scripted_establish_grasp(task)
    for _ in range(20):
        task.scene.step()
    shape_map = deskbin_shape_index(task)
    audit = contact_material_audit(task)
    query = task.run_activeforcing_query(query_force_n=4.0, displacement_m=0.012)
    feature = build_hold_feature(task)
    snapshot = PrefixSnapshot(task)
    rows = []
    try:
        for force in forces:
            snapshot.restore()
            apply_grasp_surface_materials(task, friction)
            apply_retention_collision_filters(task)
            rows.append(run_one(task, force, friction, stock, shape_map))
    finally:
        try:
            task.close_env()
        except Exception:
            pass
    return {
        "friction": friction,
        "filter": filt,
        "query_contact_ratio": query.get("contact_ratio"),
        "audit_finger_roles": audit.get("finger_grasp_roles"),
        "feature_sha256": sequence_hash(feature["sequence"]),
        "rows": rows,
    }


def main() -> None:
    out = HERE / "tiny_validation"
    out.mkdir(exist_ok=True)
    payloads = []
    rows = []
    for mu, forces in GRID.items():
        payload = run_mu(mu, forces)
        payloads.append(payload)
        rows.extend(payload["rows"])
        write(out / f"mu{mu:g}.json", payload)
    write(out / "TABLE.json", {"seed": SEED, "rows": rows, "payloads": payloads})
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
