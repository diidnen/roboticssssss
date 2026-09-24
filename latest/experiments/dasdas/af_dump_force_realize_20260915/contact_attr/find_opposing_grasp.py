"""Find a true opposing px/nx pinch. Settle only. Original squeeze inner unchanged."""
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
from run_sweep_context import make_task
from force_realize import capture_stock_drives, realize_commanded_force, uninstall_lift_squeeze
from collision_filter import apply_retention_collision_filters
from contact_sampler import _finger_state, _relative_pose, sample_contacts
from debug_grasp import scripted_establish_grasp_debug, write_debug_grasp
from grasp_geom_logger import GraspGeomLogger
from slab_geometry import quat_to_rpy

SEED = 200014
MU = 0.425
SETTLE_F = 1.0
OUT = HERE / "opposing_probe"
OPPOSITE = {"grasp_px", "grasp_nx"}
MIN_PAIR_M = 0.04
FAIL_PAIR_M = 0.02

# 4/5 default = pz wall pinch. Roll 90 + lateral still closed through the cavity.
# Official id=1/2 sit above one X-face wall; centering along TCP local Y is the
# allowed knob that puts the closing axis through the bin (px/nx).
# (id, grasp_dis, tcp_roll_deg, lateral_y_m, center_over_bin)
CANDIDATES = [
    (1, 0.0, 0.0, 0.0, True),
    (1, 0.02, 0.0, 0.0, True),
    (2, 0.0, 0.0, 0.0, True),
]


def unintended(task, shape_map, dt) -> dict:
    pairs = sample_contacts(task, shape_map, dt)
    kinds: dict[str, float] = {}
    for pair in pairs:
        kinds[pair["kind"]] = kinds.get(pair["kind"], 0.0) + float(pair["fn"])
    return {
        "finger_inner_n": float(sum(v for k, v in kinds.items() if k.startswith("finger-inner"))),
        "palm_n": float(sum(v for k, v in kinds.items() if k.startswith("robot_nonfinger"))),
        "table_bottom_n": float(kinds.get("table-bottom", 0.0)),
        "kinds": kinds,
    }


def judge(row: dict) -> tuple[bool, list[str]]:
    reasons = []
    slabs = {row.get("left_slab"), row.get("right_slab")}
    if slabs != OPPOSITE:
        reasons.append(f"slabs={sorted(str(s) for s in slabs)} not grasp_px+grasp_nx")
    pair = row.get("pair_dist")
    if pair is None:
        reasons.append("no bilateral contact pair")
    elif float(pair) < FAIL_PAIR_M:
        reasons.append(f"pair_dist={pair:.4f}m still wall-pinch scale")
    elif float(pair) < MIN_PAIR_M:
        reasons.append(f"pair_dist={pair:.4f}m >5mm but below 4cm opposing target")
    if not row.get("left_present") or not row.get("right_present"):
        reasons.append("missing left or right grasp-slab contact")
    nl, nr = float(row.get("nl") or 0), float(row.get("nr") or 0)
    if nl < 0.05 or nr < 0.05:
        reasons.append(f"forces not both nonzero NL={nl:.3f} NR={nr:.3f}")
    elif max(nl, nr) > 1e-6 and min(nl, nr) / max(nl, nr) < 0.25:
        reasons.append(f"forces very asymmetric NL={nl:.3f} NR={nr:.3f}")
    if float(row.get("finger_inner_n") or 0) > 0.05:
        reasons.append("finger-inner not zero")
    if float(row.get("palm_n") or 0) > 0.05:
        reasons.append("palm/deskbin not zero")
    if not row.get("force_realized"):
        reasons.append("original squeeze inner did not realize F_cmd")
    return (len(reasons) == 0), reasons


def plot_contacts(row: dict, slabs: dict, path: Path, rgb_path: Path | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(12.5, 5.2), dpi=130)
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)
    for name, color in (("grasp_nx", "C0"), ("grasp_px", "C3")):
        slab = slabs.get(name) or {}
        lo = np.asarray(slab.get("min") or [0, 0, 0], dtype=float)
        hi = np.asarray(slab.get("max") or [0, 0, 0], dtype=float)
        ax0.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=False, edgecolor=color, label=name))
        ax1.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False, edgecolor=color, label=name))
    for side, color, marker in (("left", "C2", "o"), ("right", "C1", "s")):
        p = row.get(f"{side}_p_obj")
        n = row.get(f"{side}_n_obj")
        if not p:
            continue
        ax0.scatter([p[0]], [p[2]], c=color, marker=marker, s=60, zorder=5, label=f"{side} {row.get(f'{side}_slab')}")
        ax1.scatter([p[0]], [p[1]], c=color, marker=marker, s=60, zorder=5, label=f"{side} {row.get(f'{side}_slab')}")
        if n:
            ax0.arrow(p[0], p[2], 0.02 * n[0], 0.02 * n[2], color=color, width=0.0004, head_width=0.004)
            ax1.arrow(p[0], p[1], 0.02 * n[0], 0.02 * n[1], color=color, width=0.0004, head_width=0.004)
    ax0.set_xlabel("object x (m)")
    ax0.set_ylabel("object z (m)")
    ax0.set_title("object xz: px/nx + contacts")
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=7, loc="best")
    ax1.set_xlabel("object x (m)")
    ax1.set_ylabel("object y (m)")
    ax1.set_title("object xy: height vs width")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc="best")
    if rgb_path and Path(rgb_path).exists():
        ax2.imshow(plt.imread(str(rgb_path)))
        ax2.set_title("head camera settle")
        ax2.axis("off")
    else:
        ax2.text(0.5, 0.5, "no RGB", ha="center")
        ax2.axis("off")
    fig.suptitle(
        f"id={row.get('left_contact_point_id')} grasp_dis={row.get('left_grasp_dis')} "
        f"pair={row.get('pair_dist')} pass={row.get('pass')}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_candidate(
    cid: int, grasp_dis: float, yaw_deg: float = 0.0, lateral_y_m: float = 0.0, center_over_bin: bool = False
) -> dict:
    tag = f"id{cid}_d{grasp_dis:g}_yaw{yaw_deg:g}_lat{lateral_y_m:g}"
    if center_over_bin:
        tag += "_center"
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "left_contact_point_id": int(cid),
        "left_grasp_dis": float(grasp_dis),
        "tcp_roll_about_local_x_deg": float(yaw_deg),
        "tcp_lateral_y_m": float(lateral_y_m),
        "center_over_bin": bool(center_over_bin),
        "ok": False,
        "pass": False,
        "reasons": [],
    }
    task = None
    try:
        task = make_task(SEED, MU, 0)
        stock = capture_stock_drives(task)
        apply_retention_collision_filters(task)
        apply_grasp_surface_materials(task, MU)
        task.activate_activeforcing_candidate_force()
        scripted_establish_grasp_debug(
            task,
            cid,
            grasp_dis,
            tcp_roll_about_local_x_deg=yaw_deg,
            tcp_lateral_y_m=lateral_y_m,
            center_over_bin=center_over_bin,
        )
        apply_retention_collision_filters(task)
        apply_grasp_surface_materials(task, MU)
        for _ in range(20):
            task.scene.step()
        logger = GraspGeomLogger(task, out_dir, SETTLE_F)
        task._af_diag_phase = "settle"
        realize = realize_commanded_force(task, SETTLE_F, stock)
        fingers, table_fn, table_present = logger._finger_grasp(task)
        left = fingers[logger.left_name]
        right = fingers[logger.right_name]
        unint = unintended(task, logger.shape_map, logger.dt)
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
        if not hasattr(ee, "p"):
            dummy = type("P", (), {})()
            arr = np.asarray(ee, dtype=float).reshape(-1)
            dummy.p, dummy.q = arr[:3], arr[3:7]
            ee = dummy
        rel_p, rel_ang, rel_q = _relative_pose(pose, ee)
        roll, pitch, yaw = quat_to_rpy(rel_q)
        logger.save_frame("settle")
        row.update(
            {
                "ok": True,
                "left_finger": logger.left_name,
                "right_finger": logger.right_name,
                "left_present": bool(left["present"]),
                "right_present": bool(right["present"]),
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
                "nl": float(left["fn"]),
                "nr": float(right["fn"]),
                "squeeze": 2.0 * min(float(left["fn"]), float(right["fn"])),
                "pair_dist": pair,
                "finger_sep": finger_sep,
                "aperture": gripper["aperture_m"],
                "finger_inner_n": unint["finger_inner_n"],
                "palm_n": unint["palm_n"],
                "table_bottom_n": unint["table_bottom_n"],
                "table_bottom": bool(table_present or table_fn > 1e-6),
                "rel_p": [float(x) for x in rel_p],
                "rel_ang_deg": float(np.degrees(rel_ang)),
                "rel_rpy_deg": [float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))],
                "realize": realize,
                "force_realized": bool(realize.get("force_realized_before_motion")),
                "kinds": unint["kinds"],
                "handover": bool((getattr(task, "_af_debug_grasp", None) or {}).get("handover")),
                "finger_world": {k: v.tolist() for k, v in world.items()},
                "object_p": [float(x) for x in pose.p],
                "object_q": [float(x) for x in pose.q],
                "slabs": {k: v for k, v in logger.slabs.items() if not str(k).startswith("_")},
                "debug_grasp": getattr(task, "_af_debug_grasp", None),
            }
        )
        passed, reasons = judge(row)
        row["pass"] = bool(passed)
        row["reasons"] = reasons
        rgb = out_dir / "frames" / "settle.png"
        plot_contacts(row, logger.slabs, out_dir / "opposing_debug.png", rgb if rgb.exists() else None)
        uninstall_lift_squeeze(task)
    except Exception as exc:
        row["ok"] = False
        row["error"] = repr(exc)
        row["traceback"] = traceback.format_exc()
        row["reasons"] = [repr(exc)]
    finally:
        (out_dir / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        if task is not None:
            try:
                task.close_env()
            except Exception:
                pass
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    found = None
    queue = list(CANDIDATES)
    for cid, grasp_dis, yaw_deg, lateral_y_m, center_over_bin in queue:
        print(
            json.dumps(
                {
                    "trying": {
                        "id": cid,
                        "grasp_dis": grasp_dis,
                        "tcp_roll": yaw_deg,
                        "lateral_y": lateral_y_m,
                        "center_over_bin": center_over_bin,
                    }
                }
            ),
            flush=True,
        )
        row = run_candidate(cid, grasp_dis, yaw_deg, lateral_y_m, center_over_bin)
        dbg = row.get("debug_grasp") or {}
        summary = {
            "id": cid,
            "grasp_dis": grasp_dis,
            "yaw_deg": yaw_deg,
            "lateral_y_m": dbg.get("tcp_lateral_y_m", lateral_y_m),
            "center_over_bin": center_over_bin,
            "centering": dbg.get("centering"),
            "ok": row.get("ok"),
            "pass": row.get("pass"),
            "left_slab": row.get("left_slab"),
            "right_slab": row.get("right_slab"),
            "pair_dist": row.get("pair_dist"),
            "finger_sep": row.get("finger_sep"),
            "nl": row.get("nl"),
            "nr": row.get("nr"),
            "squeeze": row.get("squeeze"),
            "force_realized": row.get("force_realized"),
            "reasons": row.get("reasons"),
            "error": row.get("error"),
        }
        print(json.dumps(summary, default=str), flush=True)
        results.append(summary)
        if row.get("pass"):
            found = row
            break
    payload = {"seed": SEED, "settle_F": SETTLE_F, "results": results, "chosen": None}
    if found:
        cfg = {
            "left_contact_point_id": int(found["left_contact_point_id"]),
            "left_grasp_dis": float(found["left_grasp_dis"]),
            "tcp_roll_about_local_x_deg": float(found.get("tcp_roll_about_local_x_deg") or 0.0),
            "tcp_lateral_y_m": float((found.get("debug_grasp") or {}).get("tcp_lateral_y_m") or found.get("tcp_lateral_y_m") or 0.0),
            "center_over_bin": bool((found.get("debug_grasp") or {}).get("center_over_bin")),
            "pair_dist_m": found.get("pair_dist"),
            "left_slab": found.get("left_slab"),
            "right_slab": found.get("right_slab"),
            "nl": found.get("nl"),
            "nr": found.get("nr"),
            "note": "debug opposing pinch; official scripted grasp and assets unchanged",
        }
        write_debug_grasp(cfg)
        payload["chosen"] = cfg
    (OUT / "INDEX.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"chosen": payload["chosen"], "n": len(results)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
