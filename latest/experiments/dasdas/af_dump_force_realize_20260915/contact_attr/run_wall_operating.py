"""Matched-prefix wall-pinch operating-range runs. Official id=1 grasp. Inner unchanged."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

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
from contact_sampler import sample_contacts
from grasp_geom_logger import GraspGeomLogger

SEED = 200014
FORCES = [0.50, 0.75, 1.00, 1.25]


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


def phase_mean(steps: list[dict], names: set[str], key="squeeze") -> float | None:
    rows = [row for row in steps if row.get("phase") in names]
    if not rows:
        return None
    tail = rows[-25:]
    return float(np.mean([float(row[key]) for row in tail]))


def nx_edge(finger: dict | None) -> float | None:
    if not finger:
        return None
    val = finger.get("edge_v")
    if val is None:
        return None
    if abs(float(val)) > 0.2:
        return None
    return float(val)


def summarize_run(out: Path, result: dict, dumped: dict) -> dict:
    steps = []
    path = out / "steps.jsonl"
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    steps.append(json.loads(line))
    flags = dumped.get("flags") or {}
    settle = [row for row in steps if row.get("phase") == "settle"]
    lift = [row for row in steps if row.get("phase") == "lift"]
    wrist = [row for row in steps if row.get("phase") == "wrist-rotation-start"]
    pour = [row for row in steps if row.get("phase") in {"pour-start", "pour", "terminal"}]
    motion = [row for row in steps if row.get("phase") != "settle"]
    last_settle = settle[-1] if settle else {}
    left = last_settle.get("left") or {}
    right = last_settle.get("right") or {}
    uni = flags.get("T4_unilateral_loss") or flags.get("T4_unilateral_flicker")
    uni_loss = flags.get("T4_unilateral_loss")
    bilat = flags.get("T6_bilateral_loss")
    apertures = [row.get("aperture") for row in motion if row.get("aperture") is not None]
    drels = [row.get("drel_from_settle_start_deg") for row in steps if row.get("drel_from_settle_start_deg") is not None]
    left_edges = [nx_edge(row.get("left")) for row in motion]
    right_edges = [nx_edge(row.get("right")) for row in motion]
    left_edges = [x for x in left_edges if x is not None]
    right_edges = [x for x in right_edges if x is not None]
    travels = []
    for row in motion:
        for key in ("left_travel", "right_travel"):
            if row.get(key) is not None:
                travels.append(float(row[key]))
    summary = {
        **result,
        "left_slab_settle": left.get("slab"),
        "right_slab_settle": right.get("slab"),
        "pair_dist_settle": last_settle.get("pair_dist"),
        "same_slab_settle": last_settle.get("same_slab"),
        "settle_squeeze": phase_mean(steps, {"settle"}),
        "lift_squeeze": phase_mean(steps, {"lift"}),
        "wrist_squeeze": phase_mean(steps, {"wrist-rotation-start"}),
        "pour_squeeze_geom": phase_mean(steps, {"pour-start", "pour", "terminal"}),
        "max_relative_rotation_deg": float(np.max(drels)) if drels else None,
        "min_aperture": float(np.min(apertures)) if apertures else None,
        "unilateral_loss": bool(uni_loss),
        "unilateral_loss_t": None if not uni_loss else uni_loss.get("t"),
        "unilateral_flicker_t": None if not uni else uni.get("t"),
        "bilateral_loss": bool(bilat),
        "bilateral_loss_t": None if not bilat else bilat.get("t"),
        "min_left_edge_v": None if not left_edges else float(np.min(left_edges)),
        "min_right_edge_v": None if not right_edges else float(np.min(right_edges)),
        "min_edge_margin": None if not (left_edges or right_edges) else float(np.min(left_edges + right_edges)),
        "max_contact_travel_m": None if not travels else float(np.max(travels)),
        "event_order": dumped.get("event_order"),
    }
    write(out / "metrics.json", summary)
    return summary


def capture_prefix(friction: float, episode_id: int, skip_query: bool = False):
    last_error = None
    for attempt in range(3):
        ep = int(episode_id) + 10 * attempt
        task = make_task(SEED, friction, ep)
        stock = capture_stock_drives(task)
        apply_retention_collision_filters(task)
        apply_grasp_surface_materials(task, friction)
        task.activate_activeforcing_candidate_force()
        try:
            scripted_establish_grasp(task)
        except Exception as exc:
            last_error = exc
            try:
                task.close_env()
            except Exception:
                pass
            continue
        for _ in range(20):
            task.scene.step()
        apply_retention_collision_filters(task)
        apply_grasp_surface_materials(task, friction)
        if not skip_query:
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
                last_error = RuntimeError("query failed")
                try:
                    task.close_env()
                except Exception:
                    pass
                continue
            feature = build_hold_feature(task)
            feature_sha = sequence_hash(feature["sequence"])
            query_cr = query.get("contact_ratio")
        else:
            feature_sha = "no-query"
            query_cr = None
        snapshot = PrefixSnapshot(task)
        return {
            "task": task,
            "stock": stock,
            "snapshot": snapshot,
            "feature_sha": feature_sha,
            "query_contact_ratio": query_cr,
            "episode_id": ep,
        }
    raise last_error or RuntimeError("prefix capture failed")


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
    grouped = {}
    for pair in sample_contacts(task, logger.shape_map, logger.dt):
        grouped[pair["kind"]] = grouped.get(pair["kind"], 0.0) + float(pair["fn"])
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
    result = {
        "force_N": float(force),
        "friction": float(friction),
        "retention_success": int(labels["retention_success"]),
        "contact_ratio": labels.get("contact_ratio"),
        "drop": labels.get("drop"),
        "measured_force_mean_n": labels.get("measured_force_mean_n"),
        "deskbin_z": labels.get("deskbin_z"),
        "realize": realize,
        "force_realized": bool(realize.get("force_realized_before_motion")),
        "settle_finger_inner_n": float(sum(v for k, v in grouped.items() if k.startswith("finger-inner"))),
        "settle_palm_n": float(sum(v for k, v in grouped.items() if k.startswith("robot_nonfinger"))),
        "settle_kinds": grouped,
        "event_order": dumped.get("event_order"),
        "flags": dumped.get("flags"),
    }
    write(out / "result.json", result)
    return summarize_run(out, result, dumped)


def run_context(friction: float, episode_id: int, forces: list[float], out_root: Path, skip_query: bool = False) -> dict:
    captured = capture_prefix(friction, episode_id, skip_query=skip_query)
    task = captured["task"]
    snapshot = captured["snapshot"]
    stock = captured["stock"]
    outcomes = []
    try:
        for force in forces:
            snapshot.restore()
            apply_grasp_surface_materials(task, friction)
            apply_retention_collision_filters(task)
            branch = out_root / f"r{episode_id}_mu{friction:g}_F{force:g}"
            branch.mkdir(parents=True, exist_ok=True)
            row = run_one(task, force, friction, stock, branch)
            row["feature_sha256"] = captured["feature_sha"]
            row["episode_id"] = captured["episode_id"]
            write(branch / "metrics.json", row)
            print(json.dumps({"repeat": episode_id, "mu": friction, "F": force, "ret": row.get("retention_success"), "pour": row.get("pour_squeeze_geom")}, default=str), flush=True)
            outcomes.append(row)
    finally:
        try:
            task.close_env()
        except Exception:
            pass
    payload = {
        "seed": SEED,
        "friction": friction,
        "episode_id": captured["episode_id"],
        "feature_sha256": captured["feature_sha"],
        "query_contact_ratio": captured["query_contact_ratio"],
        "outcomes": outcomes,
    }
    write(out_root / f"r{episode_id}_mu{friction:g}_CONTEXT.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "wall_pinch" / "repeats")
    parser.add_argument("--mu", type=float, default=0.425)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--start-repeat", type=int, default=0)
    parser.add_argument("--no-query", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for rid in range(args.start_repeat, args.start_repeat + args.repeats):
        payload = run_context(float(args.mu), rid, FORCES, args.out, skip_query=bool(args.no_query))
        jobs.append({"repeat": rid, "episode_id": payload["episode_id"], "sha": payload["feature_sha256"]})
        write(args.out / "INDEX.json", {"seed": SEED, "mu": args.mu, "jobs": jobs})
    print(json.dumps({"n_contexts": len(jobs), "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
