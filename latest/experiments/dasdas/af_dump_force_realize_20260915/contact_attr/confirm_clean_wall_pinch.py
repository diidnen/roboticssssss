"""STEP 1: confirm official id=1 is a clean same-wall grasp_nx pinch."""
from __future__ import annotations

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

from patch_grasp_surface import apply_grasp_surface_materials
from run_liftstyle_context import scripted_establish_grasp, write
from run_sweep_context import make_task
from force_realize import capture_stock_drives, realize_commanded_force, uninstall_lift_squeeze
from collision_filter import apply_retention_collision_filters
from contact_sampler import sample_contacts, _finger_state, _relative_pose
from grasp_geom_logger import GraspGeomLogger
from slab_geometry import quat_to_rpy

SEED = 200014
MU = 0.425
SETTLE_F = 1.0
OUT = HERE / "wall_pinch" / "clean_settle"


def kinds(task, shape_map, dt) -> dict:
    grouped: dict[str, float] = {}
    for pair in sample_contacts(task, shape_map, dt):
        grouped[pair["kind"]] = grouped.get(pair["kind"], 0.0) + float(pair["fn"])
    return grouped


def plot_clean(row: dict, slabs: dict, rgb: Path | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(13.2, 5.4), dpi=140)
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)
    nx = slabs.get("grasp_nx") or {}
    lo = np.asarray(nx.get("min") or [0, 0, 0], dtype=float)
    hi = np.asarray(nx.get("max") or [0, 0, 0], dtype=float)
    ax0.add_patch(
        Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=True, alpha=0.15, edgecolor="C0", facecolor="C0", label="grasp_nx")
    )
    ax1.add_patch(
        Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=True, alpha=0.15, edgecolor="C0", facecolor="C0", label="grasp_nx")
    )
    for side, color, marker in (("left", "C2", "o"), ("right", "C1", "s")):
        p = row.get(f"{side}_p_obj")
        n = row.get(f"{side}_n_obj")
        if not p:
            continue
        ax0.scatter([p[0]], [p[2]], c=color, marker=marker, s=70, zorder=5, label=f"{side} {row.get(f'{side}_slab')}")
        ax1.scatter([p[0]], [p[1]], c=color, marker=marker, s=70, zorder=5, label=f"{side} {row.get(f'{side}_slab')}")
        if n:
            ax0.arrow(p[0], p[2], 0.008 * n[0], 0.008 * n[2], color=color, width=0.00025, head_width=0.0025)
            ax1.arrow(p[0], p[1], 0.008 * n[0], 0.008 * n[1], color=color, width=0.00025, head_width=0.0025)
    ax0.set_xlabel("object x (m)")
    ax0.set_ylabel("object z (m)")
    ax0.set_title("grasp_nx face (xz) + contacts/normals")
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=7)
    ax1.set_xlabel("object x (m)")
    ax1.set_ylabel("object y (m)")
    ax1.set_title("thickness vs height")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)
    if rgb and rgb.exists():
        ax2.imshow(plt.imread(str(rgb)))
        ax2.set_title("head camera settle")
        ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "no RGB", ha="center")
        ax2.axis("off")
    fig.suptitle(
        f"official id=1 wall pinch  pair={row.get('pair_dist')}  "
        f"NL={row.get('nl'):.2f} NR={row.get('nr'):.2f}  clean={row.get('clean')}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    task = make_task(SEED, MU, 0)
    stock = capture_stock_drives(task)
    apply_retention_collision_filters(task)
    apply_grasp_surface_materials(task, MU)
    task.activate_activeforcing_candidate_force()
    scripted_establish_grasp(task)
    apply_retention_collision_filters(task)
    apply_grasp_surface_materials(task, MU)
    for _ in range(20):
        task.scene.step()
    logger = GraspGeomLogger(task, OUT, SETTLE_F)
    task._af_diag_phase = "settle"
    realize = realize_commanded_force(task, SETTLE_F, stock)
    fingers, table_fn, table_present = logger._finger_grasp(task)
    left = fingers[logger.left_name]
    right = fingers[logger.right_name]
    grouped = kinds(task, logger.shape_map, logger.dt)
    pair = None
    if left.get("p_world") and right.get("p_world"):
        pair = float(np.linalg.norm(np.asarray(left["p_world"]) - np.asarray(right["p_world"])))
    world = {}
    for link in task.robot.left_entity.get_links():
        name = str(link.get_name())
        if name in {logger.left_name, logger.right_name}:
            world[name] = np.asarray(link.get_pose().p, dtype=float)
    finger_sep = None
    if logger.left_name in world and logger.right_name in world:
        finger_sep = float(np.linalg.norm(world[logger.left_name] - world[logger.right_name]))
    gripper = _finger_state(task)
    pose = task.deskbin.get_pose()
    ee = task.get_arm_pose("left")
    dummy = type("P", (), {})()
    if hasattr(ee, "p"):
        dummy.p, dummy.q = np.asarray(ee.p, dtype=float), np.asarray(ee.q, dtype=float)
    else:
        arr = np.asarray(ee, dtype=float).reshape(-1)
        dummy.p, dummy.q = arr[:3], arr[3:7]
    rel_p, rel_ang, rel_q = _relative_pose(pose, dummy)
    roll, pitch, yaw = quat_to_rpy(rel_q)
    logger.save_frame("settle")
    finger_inner = float(sum(v for k, v in grouped.items() if k.startswith("finger-inner")))
    palm = float(sum(v for k, v in grouped.items() if k.startswith("robot_nonfinger")))
    lip = float(sum(v for k, v in grouped.items() if "lip" in k))
    reasons = []
    if left.get("slab") != "grasp_nx" or right.get("slab") != "grasp_nx":
        reasons.append(f"slabs={left.get('slab')},{right.get('slab')}")
    if not left.get("present") or not right.get("present"):
        reasons.append("missing finger-wall contact")
    if float(left["fn"]) < 0.05 or float(right["fn"]) < 0.05:
        reasons.append(f"forces NL={left['fn']:.3f} NR={right['fn']:.3f}")
    if pair is None:
        reasons.append("no pair")
    elif not (0.002 <= pair <= 0.015):
        reasons.append(f"pair {pair:.4f}m not wall-thickness scale")
    if finger_inner > 0.05:
        reasons.append("finger-inner support")
    if palm > 0.05:
        reasons.append("palm/deskbin")
    if lip > 0.05:
        reasons.append("lip contact")
    row = {
        "seed": SEED,
        "contact_point_id": 1,
        "clean": len(reasons) == 0,
        "reasons": reasons,
        "left_finger": logger.left_name,
        "right_finger": logger.right_name,
        "left_slab": left.get("slab"),
        "right_slab": right.get("slab"),
        "left_p_world": left.get("p_world"),
        "right_p_world": right.get("p_world"),
        "left_p_obj": left.get("p_obj"),
        "right_p_obj": right.get("p_obj"),
        "left_n_world": left.get("n_world"),
        "right_n_world": right.get("n_world"),
        "left_n_obj": left.get("n_obj"),
        "right_n_obj": right.get("n_obj"),
        "left_edge_v": left.get("edge_v"),
        "right_edge_v": right.get("edge_v"),
        "nl": float(left["fn"]),
        "nr": float(right["fn"]),
        "squeeze": 2.0 * min(float(left["fn"]), float(right["fn"])),
        "pair_dist": pair,
        "finger_sep": finger_sep,
        "aperture": gripper["aperture_m"],
        "finger_inner_n": finger_inner,
        "palm_n": palm,
        "lip_n": lip,
        "table_bottom_n": float(table_fn),
        "table_bottom": bool(table_present),
        "kinds": grouped,
        "rel_p": [float(x) for x in rel_p],
        "rel_ang_deg": float(np.degrees(rel_ang)),
        "rel_rpy_deg": [float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))],
        "realize": realize,
        "force_realized": bool(realize.get("force_realized_before_motion")),
        "regime": "CLEAN WALL PINCH" if not reasons else "NOT CLEAN",
    }
    rgb = OUT / "frames" / "settle.png"
    plot_clean(row, logger.slabs, rgb if rgb.exists() else None, OUT / "clean_wall_pinch.png")
    write(OUT / "result.json", row)
    uninstall_lift_squeeze(task)
    task.close_env()
    print(json.dumps({"clean": row["clean"], "regime": row["regime"], "pair": pair, "nl": row["nl"], "nr": row["nr"]}, default=str), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
