"""Phase 1 runtime audit: deskbin/finger/table/ball materials and contacts.

Does not change assets. Writes JSON only. cwd must be RoboTwin.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sapien
import sapien.physx as px

REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
sys.path.insert(0, str(REPO))
OUT = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915")


def mat_row(material, extra=None):
    if material is None:
        return {"present": False}
    row = {
        "present": True,
        "id": int(id(material)),
        "static_friction": float(material.static_friction),
        "dynamic_friction": float(material.dynamic_friction),
        "restitution": float(material.restitution),
        "combine_mode_exposed": False,
        "combine_mode_note": "SAPIEN 3 PhysxMaterial has no combine API; PhysX 5 default is AVERAGE",
    }
    if extra:
        row.update(extra)
    return row


def shape_row(shape, index):
    pose = shape.get_local_pose()
    aabb = None
    vertices = None
    kind = type(shape).__name__
    try:
        if hasattr(shape, "vertices"):
            vertices = np.asarray(shape.vertices, dtype=np.float64)
        elif hasattr(shape, "get_vertices"):
            vertices = np.asarray(shape.get_vertices(), dtype=np.float64)
    except Exception:
        vertices = None
    if vertices is not None and vertices.size:
        aabb = {
            "min": vertices.min(0).tolist(),
            "max": vertices.max(0).tolist(),
            "centroid": vertices.mean(0).tolist(),
            "n_vertices": int(len(vertices)),
        }
    return {
        "index": index,
        "type": kind,
        "local_pose_p": np.asarray(pose.p, dtype=float).tolist(),
        "local_pose_q": np.asarray(pose.q, dtype=float).tolist(),
        "aabb_local": aabb,
        "material": mat_row(shape.physical_material if hasattr(shape, "physical_material") else shape.get_physical_material()),
        "collision_groups": list(shape.get_collision_groups()) if hasattr(shape, "get_collision_groups") else None,
    }


def entity_shapes(entity):
    rows = []
    components = []
    if hasattr(entity, "get_components"):
        components.extend(entity.get_components())
    if hasattr(entity, "get_links"):
        components.extend(entity.get_links())
    index = 0
    for component in components:
        getter = getattr(component, "get_collision_shapes", None)
        if not callable(getter):
            continue
        for shape in getter():
            row = shape_row(shape, index)
            row["component"] = type(component).__name__
            row["component_name"] = getattr(component, "name", None) or getattr(entity, "get_name", lambda: None)()
            rows.append(row)
            index += 1
    return rows


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


def contact_pairs(task):
    contacts = task.scene.get_contacts()
    rows = []
    for contact in contacts:
        bodies = []
        for item in (contact.bodies if hasattr(contact, "bodies") else [contact.actor0, contact.actor1]):
            name = None
            try:
                ent = item.entity if hasattr(item, "entity") else item
                name = ent.get_name() if hasattr(ent, "get_name") else str(ent)
            except Exception:
                name = str(item)
            bodies.append(name)
        points = []
        for point in getattr(contact, "points", []):
            impulse = np.asarray(getattr(point, "impulse", [0, 0, 0]), dtype=float)
            pos = np.asarray(getattr(point, "position", [0, 0, 0]), dtype=float)
            points.append({"position": pos.tolist(), "impulse_n": float(np.linalg.norm(impulse))})
        rows.append({"bodies": bodies, "n_points": len(points), "points": points[:8]})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    default = px.get_default_material()
    engine_default = mat_row(default, extra={"source": "physx.get_default_material"})

    report = {
        "seed": 200014,
        "sapien": sapien.__version__,
        "physx_engine_default_material": engine_default,
        "physx_combine_rule": {
            "api_exposed": False,
            "assumed": "AVERAGE",
            "formula": "mu_eff = 0.5 * (mu_a + mu_b)",
            "source": "PhysX 5 PxMaterial default PxCombineMode::eAVERAGE; SAPIEN 3.0.0b1 does not expose setters",
        },
        "before_af_friction": None,
        "after_af_friction_0.850": None,
        "post_grasp_contacts": None,
        "deskbin_id": None,
    }

    task = make_task(200014, None)
    try:
        scene_default = mat_row(task.scene.default_physical_material, extra={"source": "scene.default_physical_material"})
        deskbin = task.deskbin.actor if hasattr(task.deskbin, "actor") else task.deskbin
        table = None
        actors = []
        for entity in task.scene.get_all_actors():
            name = entity.get_name()
            actors.append({"name": name, "n_shapes": len(entity_shapes(entity))})
            if "table" in name.lower() or name in {"table", "table_static"}:
                table = entity
        articulations = []
        finger_shapes = []
        for art in task.scene.get_all_articulations():
            art_row = {"name": art.get_name(), "links": []}
            for link in art.get_links():
                shapes = []
                for i, shape in enumerate(link.get_collision_shapes()):
                    row = shape_row(shape, i)
                    row["link"] = link.name
                    shapes.append(row)
                    if "finger" in link.name.lower() or "gripper" in link.name.lower() or "pad" in link.name.lower() or "left_link7" in link.name.lower() or "right_link7" in link.name.lower():
                        finger_shapes.append(row)
                art_row["links"].append({"name": link.name, "n_shapes": len(shapes), "shapes": shapes})
            articulations.append(art_row)

        balls = []
        for sphere in task.sphere_lst:
            balls.append({"name": sphere.get_name(), "mass": 0.0001, "shapes": entity_shapes(sphere)})

        report["before_af_friction"] = {
            "scene_default_physical_material": scene_default,
            "deskbin_name": deskbin.get_name(),
            "deskbin_id_choice": int(task.deskbin_id),
            "deskbin_shapes": entity_shapes(deskbin),
            "table_name": None if table is None else table.get_name(),
            "table_shapes": [] if table is None else entity_shapes(table),
            "ground_note": "scene.add_ground uses engine default unless overridden",
            "balls": balls,
            "actors": actors,
            "robot_links_with_collision": articulations,
            "finger_like_shapes": finger_shapes,
        }
        report["deskbin_id"] = int(task.deskbin_id)
    finally:
        task.close_env()

    task = make_task(200014, 0.85)
    try:
        deskbin = task.deskbin.actor if hasattr(task.deskbin, "actor") else task.deskbin
        report["after_af_friction_0.850"] = {
            "af_contact_friction": float(task.af_contact_friction),
            "deskbin_shapes": entity_shapes(deskbin),
            "balls": [{"name": s.get_name(), "shapes": entity_shapes(s)} for s in task.sphere_lst],
        }
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
        task.scene.step()
        contacts = contact_pairs(task)
        finger_contacts = []
        ball_contacts = []
        table_contacts = []
        for row in contacts:
            names = " ".join(str(x).lower() for x in row["bodies"])
            if "063_tabletrashbin" in names or "tabletrash" in names:
                if "link" in names or "finger" in names or "gripper" in names or "aloha" in names or "arm" in names:
                    finger_contacts.append(row)
                if "garbage" in names:
                    ball_contacts.append(row)
                if "table" in names or "ground" in names:
                    table_contacts.append(row)
        report["post_grasp_contacts"] = {
            "n_contacts": len(contacts),
            "finger_or_robot_vs_deskbin": finger_contacts[:20],
            "garbage_vs_deskbin": ball_contacts[:20],
            "table_vs_deskbin": table_contacts[:20],
            "all_body_name_pairs": [c["bodies"] for c in contacts[:80]],
        }
    finally:
        task.close_env()

    (OUT / "PHASE1_RUNTIME_AUDIT.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("wrote", OUT / "PHASE1_RUNTIME_AUDIT.json")
    shapes = report["before_af_friction"]["deskbin_shapes"]
    print("deskbin_id", report["deskbin_id"], "n_shapes_before", len(shapes))
    mus = sorted({s["material"]["static_friction"] for s in shapes if s["material"]["present"]})
    print("deskbin mus before", mus)
    after = report["after_af_friction_0.850"]["deskbin_shapes"]
    mus2 = sorted({s["material"]["static_friction"] for s in after if s["material"]["present"]})
    print("deskbin mus after 0.85 set", mus2)
    print("finger-like", len(report["before_af_friction"]["finger_like_shapes"]))
    if report["before_af_friction"]["finger_like_shapes"]:
        print("finger mu sample", report["before_af_friction"]["finger_like_shapes"][0]["material"])


if __name__ == "__main__":
    main()
