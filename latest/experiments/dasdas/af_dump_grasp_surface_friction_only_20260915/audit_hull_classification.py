"""Classify deskbin convex hulls by runtime finger / inner / table contacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
sys.path.insert(0, str(REPO))
OUT = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915")


def make_task(seed: int, friction):
    from scripts.eval_policy_xpolicylab import class_decorator, load_task_args

    user_args = {
        "task_name": "dump_bin_bigbin",
        "task_config": "demo_clean",
        "policy_name": "expert",
        "ckpt_setting": "grasp-surface-friction-audit",
        "activeforcing_enabled": True,
        "af_dynamic_evaluator": True,
        "af_contact_friction": friction,
        "af_force_limit_n": 12.0,
    }
    task_args, _ = load_task_args(user_args)
    task_args["eval_mode"] = True
    task_args["render_freq"] = 0
    task = class_decorator("dump_bin_bigbin")
    task.setup_demo(now_ep_num=0, seed=int(seed), is_test=True, **task_args)
    return task


def deskbin_shapes(task):
    entity = task.deskbin.actor if hasattr(task.deskbin, "actor") else task.deskbin
    body = entity.find_component_by_type(type(entity.get_components()[0]))
    for component in entity.get_components():
        if hasattr(component, "get_collision_shapes"):
            return list(component.get_collision_shapes())
    raise RuntimeError("no deskbin shapes")


def shape_key(shape):
    verts = np.asarray(shape.get_vertices(), dtype=np.float64)
    return tuple(np.round(verts.mean(0), 6)) + tuple(np.round(verts.min(0), 6)) + tuple(np.round(verts.max(0), 6))


def entity_name(body):
    try:
        ent = body.entity if hasattr(body, "entity") else body
        return ent.get_name()
    except Exception:
        return str(body)


def classify(task):
    shapes = deskbin_shapes(task)
    keys = [shape_key(s) for s in shapes]
    key_to_index = {k: i for i, k in enumerate(keys)}
    geom = []
    cents = []
    for i, shape in enumerate(shapes):
        verts = np.asarray(shape.get_vertices(), dtype=np.float64)
        row = {
            "index": i,
            "centroid": verts.mean(0).tolist(),
            "min": verts.min(0).tolist(),
            "max": verts.max(0).tolist(),
            "n_vertices": int(len(verts)),
        }
        geom.append(row)
        cents.append(verts.mean(0))
    cents = np.asarray(cents)
    aabb_min = np.min([np.asarray(g["min"]) for g in geom], 0)
    aabb_max = np.max([np.asarray(g["max"]) for g in geom], 0)

    def bucket_contact():
        finger = set()
        garbage = set()
        table = set()
        for contact in task.scene.get_contacts():
            names = [entity_name(b) for b in contact.bodies]
            joined = " ".join(names)
            if "063_tabletrashbin" not in joined:
                continue
            idxs = []
            for shape in contact.shapes:
                try:
                    idxs.append(key_to_index[shape_key(shape)])
                except Exception:
                    continue
            if "fl_link7" in joined or "fl_link8" in joined or "fr_link7" in joined or "fr_link8" in joined:
                finger.update(idxs)
            if "garbage" in joined:
                garbage.update(idxs)
            if "table" in joined:
                table.update(idxs)
        return finger, garbage, table

    finger, garbage, table = bucket_contact()
    y = cents[:, 1]
    xz = cents[:, [0, 2]]
    center_xz = 0.5 * (aabb_min[[0, 2]] + aabb_max[[0, 2]])
    r = np.linalg.norm(xz - center_xz, axis=1)
    height = aabb_max[1] - aabb_min[1]
    y_frac = (y - aabb_min[1]) / height
    r_frac = r / (r.max() + 1e-9)
    geom_bottom = set(np.where(y_frac < 0.18)[0].tolist())
    geom_outer = set(np.where((y_frac >= 0.18) & (r_frac >= 0.55))[0].tolist())
    geom_inner = set(np.where((y_frac >= 0.18) & (r_frac < 0.55))[0].tolist())
    return {
        "deskbin_id": int(task.deskbin_id),
        "n_shapes": len(shapes),
        "aabb_min": aabb_min.tolist(),
        "aabb_max": aabb_max.tolist(),
        "contact_finger": sorted(finger),
        "contact_garbage": sorted(garbage),
        "contact_table": sorted(table),
        "overlap_finger_garbage": sorted(finger & garbage),
        "overlap_finger_table": sorted(finger & table),
        "geom_bottom": sorted(geom_bottom),
        "geom_outer": sorted(geom_outer),
        "geom_inner": sorted(geom_inner),
        "hulls": geom,
    }


def scripted_left_grasp(task):
    from envs.utils import ArmTag

    pose = task.deskbin.get_pose().p
    arm = ArmTag("left" if pose[0] < 0 else "right")
    place = ArmTag("left")
    if arm == "right":
        task.move(task.grasp_actor(task.deskbin, arm_tag=arm, pre_grasp_dis=0.08, contact_point_id=3))
        task.move(task.move_by_displacement(arm, z=0.08, move_axis="arm"))
        task.move(task.place_actor(task.deskbin, target_pose=task.middle_pose, arm_tag=arm, pre_dis=0.08, dis=0.01))
        task.move(task.move_by_displacement(arm, z=0.1, move_axis="arm"))
        task.move(task.back_to_origin(arm), task.grasp_actor(task.deskbin, arm_tag=place, pre_grasp_dis=0.08, contact_point_id=1))
    else:
        task.move(task.grasp_actor(task.deskbin, arm_tag=place, pre_grasp_dis=0.08, contact_point_id=1))
    task._af_grasp_arm_tag = "left"


def main() -> None:
    task = make_task(200014, 0.85)
    try:
        before = classify(task)
        scripted_left_grasp(task)
        for _ in range(50):
            task.scene.step()
        after = classify(task)
        payload = {"seed": 200014, "pre_grasp": before, "post_grasp": after}
        (OUT / "PHASE1_HULL_CLASSIFICATION.json").write_text(json.dumps(payload, indent=2) + "\n")
        print("deskbin_id", after["deskbin_id"], "n", after["n_shapes"])
        print("finger", after["contact_finger"])
        print("garbage", after["contact_garbage"])
        print("table", after["contact_table"])
        print("overlap finger∩garbage", after["overlap_finger_garbage"])
        print("overlap finger∩table", after["overlap_finger_table"])
        print("n geom_outer", len(after["geom_outer"]), "inner", len(after["geom_inner"]), "bottom", len(after["geom_bottom"]))
        print("finger in geom_outer", sorted(set(after["contact_finger"]) & set(after["geom_outer"])))
        print("finger not outer", sorted(set(after["contact_finger"]) - set(after["geom_outer"])))
        print("garbage in inner", sorted(set(after["contact_garbage"]) & set(after["geom_inner"])))
        print("garbage not inner", sorted(set(after["contact_garbage"]) - set(after["geom_inner"])))
    finally:
        task.close_env()


if __name__ == "__main__":
    main()
