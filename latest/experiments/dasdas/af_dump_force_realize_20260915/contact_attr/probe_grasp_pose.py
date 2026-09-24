"""Settle-only probe of debug grasp poses. Does not change official assets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ISO = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
sys.path[:0] = [str(ISO), str(V4), str(REPO), str(ROOT), str(HERE)]

from envs.utils import ArmTag
from patch_grasp_surface import apply_grasp_surface_materials
from run_sweep_context import make_task
from collision_filter import apply_retention_collision_filters
from contact_sampler import sample_contacts, deskbin_shape_index

SEED = 200014
MU = 0.425


def grasp_with(task, contact_point_id: int, grasp_dis: float) -> None:
    deskbin_pose = task.deskbin.get_pose().p
    grasp_arm = ArmTag("left" if deskbin_pose[0] < 0 else "right")
    place_arm = ArmTag("left")
    if grasp_arm == "right":
        task.move(
            task.grasp_actor(task.deskbin, arm_tag=grasp_arm, pre_grasp_dis=0.08, contact_point_id=3)
        )
        task.move(task.move_by_displacement(grasp_arm, z=0.08, move_axis="arm"))
        task.move(
            task.place_actor(
                task.deskbin,
                target_pose=task.middle_pose,
                arm_tag=grasp_arm,
                pre_dis=0.08,
                dis=0.01,
            )
        )
        task.move(task.move_by_displacement(grasp_arm, z=0.1, move_axis="arm"))
        task.move(
            task.back_to_origin(grasp_arm),
            task.grasp_actor(
                task.deskbin,
                arm_tag=place_arm,
                pre_grasp_dis=0.08,
                grasp_dis=float(grasp_dis),
                contact_point_id=int(contact_point_id),
            ),
        )
    else:
        task.move(
            task.grasp_actor(
                task.deskbin,
                arm_tag=place_arm,
                pre_grasp_dis=0.08,
                grasp_dis=float(grasp_dis),
                contact_point_id=int(contact_point_id),
            )
        )
    task._af_grasp_arm_tag = "left"


def snapshot(task) -> dict:
    dt = float(getattr(task, "physics_timestep", 1.0 / 250.0))
    shape_map = deskbin_shape_index(task)
    pairs = sample_contacts(task, shape_map, dt)
    finger = [p for p in pairs if str(p["kind"]).startswith("finger-grasp")]
    slabs = sorted({p.get("sa") or p.get("sb") for p in finger if (p.get("sa") or p.get("sb"))})
    entity = task.robot.left_entity
    fps = {}
    for link in entity.get_links():
        name = link.get_name()
        if name in {"fl_link7", "fl_link8"}:
            fps[name] = [float(x) for x in link.get_pose().p]
    sep = None
    if "fl_link7" in fps and "fl_link8" in fps:
        import numpy as np

        sep = float(np.linalg.norm(np.asarray(fps["fl_link7"]) - np.asarray(fps["fl_link8"])))
    slabs = sorted({p.get("deskbin_name") or p.get("name") for p in finger})
    kinds = {}
    for p in pairs:
        kinds[p["kind"]] = kinds.get(p["kind"], 0.0) + float(p["fn"])
    return {
        "finger_grasp_pairs": [{"kind": p["kind"], "fn": p["fn"], "sa": p.get("sa"), "sb": p.get("sb")} for p in finger],
        "slabs": slabs,
        "finger_sep": sep,
        "n_finger_grasp": len(finger),
        "kinds": kinds,
        "obj_p": [float(x) for x in task.deskbin.get_pose().p],
    }


def main() -> None:
    probes = [
        {"contact_point_id": 1, "grasp_dis": 0.0},
        {"contact_point_id": 4, "grasp_dis": 0.0},
        {"contact_point_id": 5, "grasp_dis": 0.0},
        {"contact_point_id": 1, "grasp_dis": -0.04},
    ]
    out = []
    for spec in probes:
        task = make_task(SEED, MU, 0)
        try:
            apply_retention_collision_filters(task)
            apply_grasp_surface_materials(task, MU)
            task.activate_activeforcing_candidate_force()
            try:
                grasp_with(task, spec["contact_point_id"], spec["grasp_dis"])
            except Exception as exc:
                out.append({**spec, "ok": False, "error": repr(exc)})
                continue
            for _ in range(20):
                task.scene.step()
            row = snapshot(task)
            row.update(spec)
            row["ok"] = True
            out.append(row)
            print(json.dumps({k: row[k] for k in ("contact_point_id", "grasp_dis", "ok", "slabs", "finger_sep", "n_finger_grasp")}, default=str), flush=True)
        finally:
            try:
                task.close_env()
            except Exception:
                pass
    path = HERE / "geom_runs" / "GRASP_POSE_PROBE.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
