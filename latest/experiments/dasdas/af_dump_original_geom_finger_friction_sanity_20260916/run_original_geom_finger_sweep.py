#!/usr/bin/env python3
"""Original deskbin collider + finger-μ-only retention sanity.

Does not import split-collision patches. Does not recapture 32×20.
Does not overwrite official 18/19 or Fig.B.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path("/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin")
V4 = Path("/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915")
ROOT = Path("/media/volume/dasdas/exouser/af_dump_force_realize_20260915")
GATE_PATH = Path("/media/volume/dasdas/exouser/af_dump_pregrasp_gate_audit_20260916/GATE_DEFINITION.json")
SPLIT_HULL = Path("/media/volume/dasdas/exouser/af_dump_025n_retention_diag_20260916/MASS_MU_HULL.json")
ISO16 = Path("/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260916/ASSET_AUDIT.json")

sys.path[:0] = [str(HERE), str(V4), str(ROOT), str(REPO)]

from envs.utils import ArmTag
from run_liftstyle_context import (
    GRASP_FORCE_N,
    QUERY_FORCE_N,
    PrefixSnapshot,
    scripted_establish_grasp,
    write,
)
from force_realize import (
    capture_stock_drives,
    realize_commanded_force,
    restore_stock_drives,
    uninstall_lift_squeeze,
)

GATE = json.loads(GATE_PATH.read_text())
CUT = GATE["cutoffs"]
OFFICIAL_FORCES = [round(0.25 * i, 2) for i in range(1, 21)]
OFFICIAL_MU_EFF = {"low": 0.3625, "mid": 0.4375, "high": 0.575}
NEAR_ZERO_FINGER = 0.001
FINGER_TOKENS = ("fl_link7", "fl_link8", "fr_link7", "fr_link8")
DESKBIN = "063_tabletrashbin"
SPLIT_NAMES = {"grasp_px", "grasp_nx", "grasp_pz", "grasp_nz", "inner", "bottom"}
COMBINE_NOTE = (
    "SAPIEN 3 PhysxMaterial has no combine-mode API. PhysX 5 default is eAVERAGE. "
    "μ_eff = 0.5*(μ_finger + μ_deskbin_fixed)."
)
QUERY_F = 4.0
QUERY_DIS = 0.012


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def dump_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n")


def shape_key(shape) -> tuple:
    vertices = np.asarray(shape.get_vertices(), dtype=float)
    return (int(len(vertices)),) + tuple(np.round(vertices.mean(0), 5))


def mat_of(shape) -> dict:
    material = shape.get_physical_material()
    return {
        "static_friction": float(material.static_friction),
        "dynamic_friction": float(material.dynamic_friction),
        "restitution": float(material.restitution),
        "material_id": int(id(material)),
    }


def collision_shapes(entity) -> list:
    components = []
    if hasattr(entity, "get_components"):
        components.extend(entity.get_components())
    if hasattr(entity, "get_links"):
        components.extend(entity.get_links())
    shapes = []
    for component in components:
        getter = getattr(component, "get_collision_shapes", None)
        if callable(getter):
            shapes.extend(list(getter()))
    return shapes


def deskbin_shapes(task) -> list:
    entity = task.deskbin.actor if hasattr(task.deskbin, "actor") else task.deskbin
    shapes = collision_shapes(entity)
    if not shapes:
        raise RuntimeError("official deskbin has no collision shapes")
    return shapes


def describe_shape(shape, index: int) -> dict:
    vertices = np.asarray(shape.get_vertices(), dtype=float)
    pose = shape.get_local_pose()
    aabb_min = vertices.min(0)
    aabb_max = vertices.max(0)
    return {
        "index": int(index),
        "type": type(shape).__name__,
        "n_vertices": int(len(vertices)),
        "centroid_local": vertices.mean(0).tolist(),
        "aabb_min": aabb_min.tolist(),
        "aabb_max": aabb_max.tolist(),
        "aabb_extent": (aabb_max - aabb_min).tolist(),
        "local_pose_p": np.asarray(pose.p, dtype=float).tolist(),
        "local_pose_q": np.asarray(pose.q, dtype=float).tolist(),
        "material": mat_of(shape),
        "name_attr": str(getattr(shape, "name", "") or ""),
    }


def assert_official_geometry(task) -> dict:
    actor = task.deskbin
    names = list(getattr(actor, "_af_collision_names", None) or [])
    roles = list(getattr(actor, "_af_collision_roles", None) or [])
    if names or roles:
        raise RuntimeError(f"split collision metadata present names={names} roles={roles}")
    shapes = deskbin_shapes(task)
    rows = [describe_shape(shape, i) for i, shape in enumerate(shapes)]
    leaked = [row for row in rows if row["name_attr"] in SPLIT_NAMES]
    if leaked:
        raise RuntimeError(f"split hull names on official collider: {leaked}")
    deskbin_id = int(getattr(task, "deskbin_id", -1))
    collision_file = f"assets/objects/063_tabletrashbin/collision/base{deskbin_id}.glb"
    visual_file = f"assets/objects/063_tabletrashbin/visual/base{deskbin_id}.glb"
    mass = None
    entity = actor.actor if hasattr(actor, "actor") else actor
    try:
        import sapien

        body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if body is not None:
            mass = float(body.mass)
    except Exception:
        mass = None
    return {
        "official": True,
        "split_metadata_absent": True,
        "deskbin_id": deskbin_id,
        "collision_file": collision_file,
        "visual_file": visual_file,
        "n_shapes": len(rows),
        "shapes": rows,
        "mass_kg": mass,
        "loader": "create_actor(..., convex=True) -> add_multiple_convex_collisions_from_file",
    }


def collect_link_materials(entity, tokens=None) -> list[dict]:
    rows = []
    links = entity.get_links() if hasattr(entity, "get_links") else []
    for link in links:
        name = str(link.get_name())
        if tokens is not None and name not in tokens:
            continue
        getter = getattr(link, "get_collision_shapes", None)
        if not callable(getter):
            continue
        for shape in getter():
            row = mat_of(shape)
            row["link"] = name
            rows.append(row)
    return rows


def actor_materials(obj, name: str) -> list[dict]:
    entity = obj.actor if hasattr(obj, "actor") else obj
    rows = []
    for shape in collision_shapes(entity):
        row = mat_of(shape)
        row["actor"] = name
        rows.append(row)
    return rows


def unique_mu(rows: list[dict]) -> list[float]:
    vals = sorted({round(float(r["static_friction"]), 6) for r in rows})
    return vals


def mean_mu(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return float(np.mean([r["static_friction"] for r in rows]))


def make_task(seed: int, episode_id: int):
    from scripts.eval_policy_xpolicylab import class_decorator, load_task_args
    import envs.dump_bin_bigbin as dump_mod
    import inspect

    src = inspect.getsource(dump_mod.create_actor)
    if "grasp_px" in src or "convex_panels" in src or "install_patches" in src:
        raise RuntimeError("split create_actor patch is installed; abort")

    user_args = {
        "task_name": "dump_bin_bigbin",
        "task_config": "demo_clean",
        "policy_name": "expert",
        "ckpt_setting": "scripted-original-geom-finger-friction",
        "activeforcing_enabled": True,
        "af_dynamic_evaluator": True,
        "af_force_limit_n": float(GRASP_FORCE_N),
    }
    task_args, _ = load_task_args(user_args)
    task_args["eval_mode"] = True
    task_args["render_freq"] = 0
    if task_args.get("af_contact_friction") not in (None, False):
        raise RuntimeError(f"af_contact_friction leaked into task_args: {task_args.get('af_contact_friction')!r}")
    task = class_decorator("dump_bin_bigbin")
    task.setup_demo(now_ep_num=episode_id, seed=int(seed), is_test=True, **task_args)
    if getattr(task, "af_contact_friction", None) not in (None, False):
        raise RuntimeError("deskbin af_contact_friction was set; this experiment must leave it unset")
    return task


def freeze_deskbin_material(task, mu: float) -> int:
    return int(task.set_actor_contact_friction(task.deskbin, float(mu)))


def set_finger_friction(task, mu: float) -> dict:
    material = task.scene.create_physical_material(float(mu), float(mu), 0.0)
    painted = []
    for entity in (task.robot.left_entity, task.robot.right_entity):
        for link in entity.get_links():
            name = str(link.get_name())
            if name not in FINGER_TOKENS:
                continue
            for shape in link.get_collision_shapes():
                shape.set_physical_material(material)
                painted.append({"link": name, **mat_of(shape)})
    if len(painted) < 4:
        raise RuntimeError(f"expected >=4 finger shapes, got {len(painted)}")
    mus = unique_mu(painted)
    if mus != [round(float(mu), 6)] and abs(mus[0] - float(mu)) > 1e-5:
        raise RuntimeError(f"finger μ not applied: {mus}")
    return {"n_shapes": len(painted), "mu": float(mu), "shapes": painted}


def materials_audit(task) -> dict:
    desk = [describe_shape(s, i) | {"material": mat_of(s)} for i, s in enumerate(deskbin_shapes(task))]
    fingers = collect_link_materials(task.robot.left_entity, FINGER_TOKENS) + collect_link_materials(
        task.robot.right_entity, FINGER_TOKENS
    )
    table_rows = actor_materials(task.table, "table") if getattr(task, "table", None) is not None else []
    balls = []
    for sphere in task.sphere_lst:
        balls.extend(actor_materials(sphere, "garbage"))
    desk_mu = unique_mu([row["material"] for row in desk])
    finger_mu = unique_mu(fingers)
    table_mu = unique_mu(table_rows)
    ball_mu = unique_mu(balls)
    dmu = desk_mu[0] if len(desk_mu) == 1 else None
    fmu = finger_mu[0] if len(finger_mu) == 1 else None
    return {
        "deskbin_mu_values": desk_mu,
        "finger_mu_values": finger_mu,
        "table_mu_values": table_mu,
        "ball_mu_values": ball_mu,
        "deskbin_mu": dmu,
        "finger_mu": fmu,
        "table_mu": table_mu[0] if len(table_mu) == 1 else mean_mu(table_rows),
        "ball_mu": ball_mu[0] if len(ball_mu) == 1 else mean_mu(balls),
        "mu_eff_average": None if dmu is None or fmu is None else 0.5 * (dmu + fmu),
        "n_deskbin_shapes": len(desk),
        "n_finger_shapes": len(fingers),
        "n_table_shapes": len(table_rows),
        "n_ball_shapes": len(balls),
        "deskbin_shapes_mu": [{"index": row["index"], **row["material"]} for row in desk],
        "finger_shapes": fingers,
    }


def body_name(body) -> str:
    try:
        return str(body.entity.name)
    except Exception:
        return "?"


def axis_bucket(normal) -> str:
    n = np.asarray(normal, dtype=float)
    n = n / (float(np.linalg.norm(n)) + 1e-12)
    idx = int(np.argmax(np.abs(n)))
    sign = "+" if n[idx] >= 0 else "-"
    return sign + "xyz"[idx]


def shape_map(task) -> dict:
    mapping = {}
    for index, shape in enumerate(deskbin_shapes(task)):
        mapping[shape_key(shape)] = {"index": index, "type": type(shape).__name__, "material": mat_of(shape)}
    return mapping


def contact_geometry(task) -> dict:
    dt = float(getattr(task, "physics_timestep", 1.0 / 250.0))
    mapping = shape_map(task)
    left_names = {joint.child_link.get_name() for joint, _, _ in task.robot.left_gripper}
    right_names = {joint.child_link.get_name() for joint, _, _ in task.robot.right_gripper}
    fingers = left_names | right_names | set(FINGER_TOKENS)
    per_finger = {
        name: {"shape_ids": [], "normals": [], "fn": 0.0, "axes": [], "mu_pairs": []}
        for name in sorted(fingers)
    }
    ball_mu = []
    table_mu = []
    finger_mu_eff = []
    details = []
    for contact in task.scene.get_contacts():
        name0 = body_name(contact.bodies[0])
        name1 = body_name(contact.bodies[1])
        if DESKBIN not in (name0, name1):
            continue
        try:
            shapes = list(contact.shapes)
        except Exception:
            shapes = []
        metas = []
        for shape in shapes[:2]:
            try:
                metas.append(mapping.get(shape_key(shape)))
            except Exception:
                metas.append(None)
        other = name1 if name0 == DESKBIN else name0
        desk_meta = metas[0] if name0 == DESKBIN else metas[1] if len(metas) > 1 else None
        other_shape = shapes[1] if name0 == DESKBIN and len(shapes) > 1 else shapes[0] if name1 == DESKBIN and shapes else None
        desk_mu = None if desk_meta is None else float(desk_meta["material"]["static_friction"])
        other_mu = None
        if other_shape is not None:
            try:
                other_mu = float(other_shape.get_physical_material().static_friction)
            except Exception:
                other_mu = None
        force = 0.0
        normals = []
        for point in contact.points:
            impulse = np.asarray(point.impulse, dtype=float)
            normal = np.asarray(point.normal, dtype=float)
            nrm = float(np.linalg.norm(normal)) + 1e-12
            normal = normal / nrm
            fn = abs(float(np.dot(impulse, normal))) / max(dt, 1e-9)
            force += fn
            normals.append(normal.tolist())
        if force < 1e-6:
            continue
        mu_eff = None if desk_mu is None or other_mu is None else 0.5 * (desk_mu + other_mu)
        shape_id = None if desk_meta is None else int(desk_meta["index"])
        mean_n = np.mean(normals, axis=0).tolist() if normals else [0.0, 0.0, 1.0]
        rec = {
            "other": other,
            "deskbin_shape_id": shape_id,
            "fn": float(force),
            "n_world": [float(x) for x in mean_n],
            "axis": axis_bucket(mean_n),
            "deskbin_mu": desk_mu,
            "other_mu": other_mu,
            "mu_eff": mu_eff,
        }
        details.append(rec)
        if other in fingers:
            bucket = per_finger.setdefault(
                other, {"shape_ids": [], "normals": [], "fn": 0.0, "axes": [], "mu_pairs": []}
            )
            if shape_id is not None:
                bucket["shape_ids"].append(shape_id)
            bucket["normals"].append([float(x) for x in mean_n])
            bucket["fn"] += float(force)
            bucket["axes"].append(axis_bucket(mean_n))
            bucket["mu_pairs"].append({"deskbin_mu": desk_mu, "finger_mu": other_mu, "mu_eff": mu_eff})
            if mu_eff is not None:
                finger_mu_eff.append(mu_eff)
        elif other == "table":
            if mu_eff is not None:
                table_mu.append({"deskbin_mu": desk_mu, "table_mu": other_mu, "mu_eff": mu_eff})
        elif "garbage" in other or other.startswith("sphere"):
            if mu_eff is not None:
                ball_mu.append({"deskbin_mu": desk_mu, "ball_mu": other_mu, "mu_eff": mu_eff})
    contact_forces = task.get_actor_gripper_contact_forces(task.deskbin, "left")
    per = contact_forces.get("per_finger_normal_force_n") or {}
    names = list(per)
    nl = float(per.get(names[0], 0.0)) if names else 0.0
    nr = float(per.get(names[1], 0.0)) if len(names) > 1 else 0.0
    joints = task.robot.left_gripper
    entity = task.robot.left_entity
    active = list(entity.get_active_joints())
    positions = []
    worlds = []
    for joint, _, _ in joints:
        positions.append(abs(float(entity.qpos[active.index(joint)])))
        link = joint.child_link
        pose = None
        for attr in ("get_entity_pose", "get_pose", "pose"):
            val = getattr(link, attr, None)
            try:
                pose = val() if callable(val) else val
                if pose is not None:
                    break
            except Exception:
                continue
        if pose is not None and hasattr(pose, "p"):
            worlds.append(np.asarray(pose.p, dtype=float))
    aperture = float(np.mean(positions)) if positions else float(task.robot.get_left_gripper_val())
    pair = float(np.linalg.norm(worlds[0] - worlds[1])) if len(worlds) == 2 else None
    pose = task.deskbin.get_pose()
    summary_fingers = {}
    wrapping = False
    for name, bucket in per_finger.items():
        ids = sorted(set(bucket["shape_ids"]))
        axes = sorted(set(bucket["axes"]))
        multi = len(ids) >= 2 and len(axes) >= 2
        if bucket["fn"] > 1e-6 and multi:
            wrapping = True
        summary_fingers[name] = {
            "present": bool(bucket["fn"] > 1e-6),
            "fn": float(bucket["fn"]),
            "n_shapes": len(ids),
            "shape_ids": ids,
            "axes": axes,
            "normals": bucket["normals"],
            "multi_hull_wedging": bool(multi and bucket["fn"] > 1e-6),
            "mu_eff_values": sorted(
                {round(float(x["mu_eff"]), 6) for x in bucket["mu_pairs"] if x.get("mu_eff") is not None}
            ),
        }
    contacting = [row for row in summary_fingers.values() if row["present"]]
    return {
        "left_present": bool(nl > 1e-6 or any(summary_fingers.get(n, {}).get("present") for n in left_names)),
        "right_present": bool(nr > 1e-6 or any(summary_fingers.get(n, {}).get("present") for n in right_names)),
        "nl": nl,
        "nr": nr,
        "squeeze": 2.0 * min(nl, nr) if names else 2.0 * float(contact_forces.get("single_finger_normal_force_n") or 0.0),
        "bilateral": bool(contact_forces.get("bilateral_contact")),
        "aperture": aperture,
        "pair_dist": pair,
        "obj_xyz": np.asarray(pose.p, dtype=float).tolist(),
        "obj_q": np.asarray(pose.q, dtype=float).tolist(),
        "obj_z": float(pose.p[2]),
        "per_finger": summary_fingers,
        "n_contacting_fingers": len(contacting),
        "multi_hull_wedging_any_finger": wrapping,
        "split_like_nx_pz": any(
            {"-x", "+z"} <= set(row["axes"]) or {"+x", "+z"} <= set(row["axes"]) for row in contacting
        ),
        "split_like_nx_nz_pz": any(len(set(row["axes"]) & {"-x", "+x", "-z", "+z"}) >= 3 for row in contacting),
        "finger_deskbin_mu_eff": sorted({round(x, 6) for x in finger_mu_eff}),
        "ball_deskbin_mu_eff": sorted({round(float(x["mu_eff"]), 6) for x in ball_mu}),
        "table_deskbin_mu_eff": sorted({round(float(x["mu_eff"]), 6) for x in table_mu}),
        "ball_contacts": ball_mu[:8],
        "table_contacts": table_mu[:4],
        "n_deskbin_contacts": len(details),
        "contact_shape_ids": sorted({d["deskbin_shape_id"] for d in details if d["deskbin_shape_id"] is not None}),
        "details": details[:40],
    }


def pack_state(task) -> dict:
    geom = contact_geometry(task)
    return {
        "left_present": geom["left_present"],
        "right_present": geom["right_present"],
        "nl": geom["nl"],
        "nr": geom["nr"],
        "squeeze": geom["squeeze"],
        "pair_dist": geom["pair_dist"],
        "bilateral": geom["bilateral"],
        "aperture": geom["aperture"],
        "obj_z": geom["obj_z"],
        "contact_shape_ids": geom["contact_shape_ids"],
        "per_finger": {k: {"shape_ids": v["shape_ids"], "axes": v["axes"], "fn": v["fn"], "n_shapes": v["n_shapes"]} for k, v in geom["per_finger"].items() if v["present"]},
        "multi_hull_wedging_any_finger": geom["multi_hull_wedging_any_finger"],
        "split_like_nx_pz": geom["split_like_nx_pz"],
        "split_like_nx_nz_pz": geom["split_like_nx_nz_pz"],
        "finger_deskbin_mu_eff": geom["finger_deskbin_mu_eff"],
        "ball_deskbin_mu_eff": geom["ball_deskbin_mu_eff"],
        "table_deskbin_mu_eff": geom["table_deskbin_mu_eff"],
    }


def classify_pre(window: list[dict]) -> dict:
    if not window:
        return {"gate": "INVALID", "reason": "empty PRE window", "window_cr": 0.0}
    bits = [bool(row["bilateral"] and row["left_present"] and row["right_present"]) for row in window]
    cr = float(np.mean(bits))
    end = window[-1]
    ap = float(end["aperture"] or 0.0)
    sq = float(end["squeeze"] or 0.0)
    endpoint = bool(end["bilateral"] and end["left_present"] and end["right_present"])
    if ap >= CUT["aperture_miss_m"] or (cr <= 1e-9 and sq <= CUT["squeeze_alive_n"] and not endpoint):
        return {
            "gate": "INVALID",
            "reason": f"open/miss ap={ap*1000:.1f} mm squeeze={sq:.3f} CR={cr:.2f}",
            "window_cr": cr,
        }
    if cr >= CUT["window_bilateral_valid_min"] and endpoint and sq > CUT["squeeze_alive_n"] and ap < CUT["aperture_miss_m"]:
        return {
            "gate": "VALID",
            "reason": "PRE bilateral pinch on official collider (nx+nx N/A)",
            "window_cr": cr,
        }
    if cr >= CUT["window_bilateral_borderline_min"] and ap < CUT["aperture_miss_m"] and (not endpoint or sq <= CUT["squeeze_alive_n"]):
        return {"gate": "BORDERLINE", "reason": "high-CR PRE with endpoint flicker", "window_cr": cr}
    return {
        "gate": "INVALID",
        "reason": f"weak PRE CR={cr:.2f} bilateral={endpoint} squeeze={sq:.3f}",
        "window_cr": cr,
    }


def ball_band(task):
    zs = []
    n = 0
    for sphere in task.sphere_lst:
        z = float(sphere.get_pose().p[2])
        zs.append(z)
        if 0.13 <= z <= 0.25:
            n += 1
    return n, zs


def labels_from_metrics(task, metrics: dict) -> dict:
    official = bool(task.check_success()) and bool(task.plan_success)
    contact_ratio = float(metrics.get("contact_ratio") or 0.0)
    irrecoverable = bool(metrics.get("irrecoverable_failure"))
    n_balls, ball_z = ball_band(task)
    pose = np.asarray(task.deskbin.get_pose().p, dtype=float)
    retention = bool(contact_ratio >= 0.8 and not irrecoverable)
    trace = list(getattr(task, "_af_force_trace", []) or [])
    left_n = np.asarray([float(s["left_force_n"]) for s in trace], dtype=float) if trace else np.zeros(0)
    right_n = np.asarray([float(s["right_force_n"]) for s in trace], dtype=float) if trace else np.zeros(0)
    squeeze = np.asarray([2.0 * min(a, b) for a, b in zip(left_n, right_n)], dtype=float) if trace else np.zeros(0)
    bilateral = np.asarray(
        [bool(s["left_bilateral"] or s["right_bilateral"]) for s in trace], dtype=bool
    ) if trace else np.zeros(0, dtype=bool)
    return {
        "official_full_task_success": int(official),
        "retention_success": int(retention),
        "contact_ratio": contact_ratio,
        "persistent_contact": bool(contact_ratio >= 0.8),
        "irrecoverable_failure": irrecoverable,
        "drop": int(irrecoverable or contact_ratio < 0.5),
        "n_balls_in_official_band": int(n_balls),
        "ball_z": ball_z,
        "deskbin_xyz": pose.tolist(),
        "deskbin_z": float(pose[2]),
        "measured_force_mean_n": metrics.get("measured_force_mean_n"),
        "measured_bilateral_squeeze_mean_n": float(np.mean(squeeze)) if len(squeeze) else 0.0,
        "measured_left_mean_n": float(np.mean(left_n)) if len(left_n) else 0.0,
        "measured_right_mean_n": float(np.mean(right_n)) if len(right_n) else 0.0,
        "bilateral_fraction": float(np.mean(bilateral)) if len(bilateral) else 0.0,
        "samples": metrics.get("samples"),
    }


def remainder_phased(task) -> None:
    place = ArmTag("left")
    task.move(task.move_by_displacement(arm_tag=place, z=0.08, move_axis="arm"))
    task.move(task.pour_actions)
    task.move(task.pour_actions)
    task.move(task.pour_actions)
    task.delay(6)


def remainder_with_geom(task, want_geom: bool) -> dict:
    samples = {}
    place = ArmTag("left")
    task.move(task.move_by_displacement(arm_tag=place, z=0.08, move_axis="arm"))
    if want_geom:
        samples["after_lift"] = contact_geometry(task)
    task.move(task.pour_actions)
    task.move(task.pour_actions)
    task.move(task.pour_actions)
    task.delay(6)
    if want_geom:
        samples["after_delay"] = contact_geometry(task)
    return samples


def compute_finger_levels(deskbin_mu: float) -> dict:
    levels = {}
    for name, target in OFFICIAL_MU_EFF.items():
        finger = 2.0 * float(target) - float(deskbin_mu)
        if finger < 0:
            raise RuntimeError(
                f"cannot match μ_eff={target} with fixed deskbin μ={deskbin_mu}: finger would be {finger}"
            )
        levels[name] = {
            "mu_finger": float(finger),
            "mu_deskbin_fixed": float(deskbin_mu),
            "mu_eff": 0.5 * (finger + float(deskbin_mu)),
            "mu_eff_target": float(target),
        }
    levels["near_zero_diagnostic"] = {
        "mu_finger": float(NEAR_ZERO_FINGER),
        "mu_deskbin_fixed": float(deskbin_mu),
        "mu_eff": 0.5 * (NEAR_ZERO_FINGER + float(deskbin_mu)),
        "mu_eff_target": None,
        "paper_data": False,
    }
    return levels


def split_before_payload() -> dict:
    split = {}
    if SPLIT_HULL.exists():
        hull = json.loads(SPLIT_HULL.read_text())
        split["n_shapes"] = hull.get("hulls", {}).get("n_shapes")
        split["grasp_names"] = hull.get("hulls", {}).get("grasp_names")
        split["shapes"] = list((hull.get("hulls", {}).get("shapes") or {}).keys())
        split["source"] = str(SPLIT_HULL)
    if ISO16.exists():
        audit = json.loads(ISO16.read_text())
        split["iso16_n_shapes"] = audit.get("n_deskbin_shapes_after_split")
        split["iso16_roles"] = audit.get("roles")
    return split


def probe(seed: int, episode_id: int) -> dict:
    task = make_task(seed, episode_id)
    try:
        geom = assert_official_geometry(task)
        before = split_before_payload()
        measured = materials_audit(task)
        desk_mu_vals = measured["deskbin_mu_values"]
        if len(desk_mu_vals) != 1:
            raise RuntimeError(f"official deskbin μ not unique before freeze: {desk_mu_vals}")
        desk_mu = float(desk_mu_vals[0])
        n_painted = freeze_deskbin_material(task, desk_mu)
        after_freeze = materials_audit(task)
        if after_freeze["deskbin_mu_values"] != [round(desk_mu, 6)] and abs(after_freeze["deskbin_mu"] - desk_mu) > 1e-5:
            raise RuntimeError("deskbin freeze failed")
        default_finger = measured["finger_mu"]
        levels = compute_finger_levels(desk_mu)
        set_finger_friction(task, levels["mid"]["mu_finger"])
        after_finger = materials_audit(task)
        if abs(after_finger["finger_mu"] - levels["mid"]["mu_finger"]) > 1e-5:
            raise RuntimeError("finger μ paint failed")
        if abs(after_finger["deskbin_mu"] - desk_mu) > 1e-5:
            raise RuntimeError("painting fingers changed deskbin μ")
        if abs(after_finger["table_mu"] - measured["table_mu"]) > 1e-5:
            raise RuntimeError("painting fingers changed table μ")
        if abs(after_finger["ball_mu"] - measured["ball_mu"]) > 1e-5:
            raise RuntimeError("painting fingers changed ball μ")
        payload = {
            "seed": seed,
            "episode_id": episode_id,
            "combine_note": COMBINE_NOTE,
            "geometry_before_split": before,
            "geometry_after_official": geom,
            "n_shapes_before_split": before.get("n_shapes") or before.get("iso16_n_shapes"),
            "n_shapes_after_official": geom["n_shapes"],
            "geometry_restored": bool(geom["official"] and geom["split_metadata_absent"] and geom["n_shapes"] > 0),
            "split_names_absent": True,
            "default_deskbin_mu_runtime": desk_mu,
            "default_finger_mu_runtime": default_finger,
            "default_table_mu_runtime": measured["table_mu"],
            "default_ball_mu_runtime": measured["ball_mu"],
            "freeze_painted_shapes": n_painted,
            "finger_levels": levels,
            "isolation_spotcheck_after_mid_finger_paint": {
                "finger_mu": after_finger["finger_mu"],
                "deskbin_mu": after_finger["deskbin_mu"],
                "table_mu": after_finger["table_mu"],
                "ball_mu": after_finger["ball_mu"],
                "mu_eff": after_finger["mu_eff_average"],
            },
            "af_contact_friction": getattr(task, "af_contact_friction", None),
        }
        dump_json(HERE / "GEOMETRY_BEFORE_AFTER.json", payload)
        dump_json(HERE / "MU_PLAN.json", {"deskbin_mu_fixed": desk_mu, "levels": levels, "combine": COMBINE_NOTE})
        print(json.dumps({"probe": True, "n_shapes": geom["n_shapes"], "deskbin_mu": desk_mu, "levels": {k: v["mu_finger"] for k, v in levels.items()}}, indent=2), flush=True)
        return payload
    finally:
        try:
            task.close_env()
        except Exception:
            pass


def prepare_condition(task, desk_mu: float, finger_mu: float) -> dict:
    freeze_deskbin_material(task, desk_mu)
    fingers = set_finger_friction(task, finger_mu)
    audit = materials_audit(task)
    if abs(audit["deskbin_mu"] - desk_mu) > 1e-5:
        raise RuntimeError("deskbin μ drifted")
    if abs(audit["finger_mu"] - finger_mu) > 1e-5:
        raise RuntimeError("finger μ drifted")
    geom = assert_official_geometry(task)
    return {"fingers": fingers, "audit": audit, "geometry": {"n_shapes": geom["n_shapes"], "deskbin_id": geom["deskbin_id"]}}


def capture_prefix(seed: int, episode_id: int, desk_mu: float, finger_mu: float, label: str):
    task = make_task(seed, episode_id)
    try:
        stock = capture_stock_drives(task)
        prepared = prepare_condition(task, desk_mu, finger_mu)
        task.activate_activeforcing_candidate_force()
        scripted_establish_grasp(task)
        window = []
        for _ in range(20):
            task.scene.step()
            window.append(pack_state(task))
        gate = classify_pre(window)
        pre_geom = contact_geometry(task)
        print(
            json.dumps(
                {
                    "event": "PRE",
                    "label": label,
                    "episode_id": episode_id,
                    "gate": gate["gate"],
                    "cr": gate["window_cr"],
                    "squeeze": window[-1]["squeeze"] if window else None,
                    "aperture_mm": None if not window else 1000.0 * float(window[-1]["aperture"] or 0.0),
                    "finger_mu": finger_mu,
                    "deskbin_mu": desk_mu,
                    "mu_eff": 0.5 * (finger_mu + desk_mu),
                    "n_shapes": prepared["geometry"]["n_shapes"],
                    "contact_shape_ids": window[-1]["contact_shape_ids"] if window else [],
                }
            ),
            flush=True,
        )
        pre_query = PrefixSnapshot(task)
        query = None
        query_error = None
        for _ in range(3):
            try:
                query = task.run_activeforcing_query(query_force_n=QUERY_F, displacement_m=QUERY_DIS)
                query_error = None
                break
            except RuntimeError as exc:
                query_error = repr(exc)
                if "planning failed" not in query_error:
                    raise
                pre_query.restore()
                prepare_condition(task, desk_mu, finger_mu)
        if query is None:
            query = {"contact_ratio": 0.0, "query_failed": True, "error": query_error}
        snapshot = PrefixSnapshot(task)
        post_query = pack_state(task)
        post_query_geom = contact_geometry(task)
        return {
            "task": task,
            "stock": stock,
            "snapshot": snapshot,
            "episode_id": episode_id,
            "deskbin_id": prepared["geometry"]["deskbin_id"],
            "n_shapes": prepared["geometry"]["n_shapes"],
            "audit": prepared["audit"],
            "pre_gate": gate,
            "pre_end": window[-1] if window else {},
            "pre_window": window,
            "pre_geom": pre_geom,
            "query": query,
            "post_query": post_query,
            "post_query_geom": post_query_geom,
        }
    except Exception:
        try:
            task.close_env()
        except Exception:
            pass
        raise


def find_and_capture(seed: int, desk_mu: float, finger_mu: float, label: str, prefer: int, force_episode: int | None):
    last = None
    episodes = [int(force_episode)] if force_episode is not None else [prefer] + [i for i in range(10) if i != prefer]
    for episode_id in episodes:
        try:
            captured = capture_prefix(seed, episode_id, desk_mu, finger_mu, label)
            if captured["pre_gate"]["gate"] == "VALID":
                return captured
            last = captured
            try:
                captured["task"].close_env()
            except Exception:
                pass
        except Exception as exc:
            print(json.dumps({"event": "capture_error", "label": label, "episode_id": episode_id, "error": repr(exc)}), flush=True)
            if type(exc).__name__ == "UnStableError" or "UnStableError" in repr(exc):
                raise
            last = exc
    if isinstance(last, dict) and last.get("pre_gate", {}).get("gate") in {"VALID", "BORDERLINE"}:
        return last
    if isinstance(last, dict):
        return last
    raise last or RuntimeError(f"no prefix for {label}")


def run_one(task, force: float, stock, desk_mu: float, finger_mu: float, geom_f025: bool) -> dict:
    prepare_condition(task, desk_mu, finger_mu)
    task.plan_success = True
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    realize = realize_commanded_force(task, force, stock)
    post_realize = contact_geometry(task) if geom_f025 else pack_state(task)
    task._af_force_trace = []
    task._af_force_trace_step = 0
    task._af_no_contact_samples = 0
    task._af_ever_lifted = False
    task.activate_activeforcing_candidate_force()
    restore_stock_drives(stock)
    geom_samples = remainder_with_geom(task, geom_f025) if geom_f025 else {}
    if not geom_f025:
        remainder_phased(task)
    uninstall_lift_squeeze(task)
    metrics = task.compute_activeforcing_dynamic_metrics()
    labels = labels_from_metrics(task, metrics)
    end = pack_state(task)
    end_geom = contact_geometry(task)
    return {
        "force_N": float(force),
        "retention_success": int(labels["retention_success"]),
        "official_full_task_success": int(labels["official_full_task_success"]),
        "contact_ratio": labels.get("contact_ratio"),
        "drop": labels.get("drop"),
        "irrecoverable_failure": labels.get("irrecoverable_failure"),
        "measured_force_mean_n": labels.get("measured_force_mean_n"),
        "measured_bilateral_squeeze_mean_n": labels.get("measured_bilateral_squeeze_mean_n"),
        "measured_left_mean_n": labels.get("measured_left_mean_n"),
        "measured_right_mean_n": labels.get("measured_right_mean_n"),
        "settle_squeeze_mean_n": realize.get("pre_motion_squeeze_mean_n"),
        "force_realized": bool(realize.get("force_realized_before_motion")),
        "n_balls_in_official_band": labels.get("n_balls_in_official_band"),
        "deskbin_z": labels.get("deskbin_z"),
        "end_aperture_m": end.get("aperture"),
        "end_pair_dist": end.get("pair_dist"),
        "end_squeeze": end.get("squeeze"),
        "end_nl": end.get("nl"),
        "end_nr": end.get("nr"),
        "end_contact_shape_ids": end.get("contact_shape_ids"),
        "end_multi_hull_wedging": end.get("multi_hull_wedging_any_finger"),
        "end_split_like_nx_pz": end.get("split_like_nx_pz"),
        "end_split_like_nx_nz_pz": end.get("split_like_nx_nz_pz"),
        "finger_deskbin_mu_eff_end": end.get("finger_deskbin_mu_eff"),
        "ball_deskbin_mu_eff_end": end.get("ball_deskbin_mu_eff"),
        "table_deskbin_mu_eff_end": end.get("table_deskbin_mu_eff"),
        "realize": realize,
        "post_realize_geom": post_realize if geom_f025 else None,
        "lift_delay_geom": geom_samples,
        "end_geom": end_geom if geom_f025 else None,
        "end_per_finger": end.get("per_finger"),
    }


def fmin_first(forces, bits):
    for force, ok in zip(forces, bits):
        if int(ok):
            return float(force)
    return None


def fmin_stable(forces, bits):
    seen = 0
    for val in bits:
        if int(val) < seen:
            return None
        seen = max(seen, int(val))
    return fmin_first(forces, bits)


def run_level(seed: int, name: str, spec: dict, forces: list[float], prefer_episode: int, force_episode: int | None) -> dict:
    desk_mu = float(spec["mu_deskbin_fixed"])
    finger_mu = float(spec["mu_finger"])
    mu_eff = float(spec["mu_eff"])
    captured = find_and_capture(seed, desk_mu, finger_mu, name, prefer_episode, force_episode)
    task = captured["task"]
    outcomes = []
    try:
        for force in forces:
            captured["snapshot"].restore()
            prepare_condition(task, desk_mu, finger_mu)
            row = run_one(task, force, captured["stock"], desk_mu, finger_mu, geom_f025=abs(float(force) - 0.25) < 1e-9)
            outcomes.append(row)
            print(
                json.dumps(
                    {
                        "event": "F",
                        "label": name,
                        "F": force,
                        "ret": row["retention_success"],
                        "task": row["official_full_task_success"],
                        "CR": row["contact_ratio"],
                        "squeeze": row["settle_squeeze_mean_n"],
                        "mu_eff_contact": row["finger_deskbin_mu_eff_end"],
                        "wedging": row["end_multi_hull_wedging"],
                    }
                ),
                flush=True,
            )
        bits_r = [o["retention_success"] for o in outcomes]
        bits_t = [o["official_full_task_success"] for o in outcomes]
        context = {
            "label": name,
            "paper_data": spec.get("paper_data", True),
            "seed": seed,
            "episode_id": captured["episode_id"],
            "deskbin_id": captured["deskbin_id"],
            "n_shapes": captured["n_shapes"],
            "mu_finger": finger_mu,
            "mu_deskbin_fixed": desk_mu,
            "mu_eff": mu_eff,
            "audit": captured["audit"],
            "pre_gate": captured["pre_gate"],
            "pre_end": captured["pre_end"],
            "pre_geom": captured["pre_geom"],
            "query": {
                "contact_ratio": (captured["query"] or {}).get("contact_ratio"),
                "query_failed": bool((captured["query"] or {}).get("query_failed")),
                "error": (captured["query"] or {}).get("error"),
            },
            "post_query": captured["post_query"],
            "post_query_geom": captured["post_query_geom"],
            "outcomes": outcomes,
            "F_min_retention_first": fmin_first(forces, bits_r),
            "F_min_retention_stable": fmin_stable(forces, bits_r),
            "F_min_fulltask_first": fmin_first(forces, bits_t),
            "n_retention": int(sum(bits_r)),
            "n_fulltask": int(sum(bits_t)),
            "retain_at_0.25": None if not outcomes else int(outcomes[0]["retention_success"]),
        }
        dump_json(HERE / "sweeps" / f"{name}_CONTEXT.json", context)
        return context
    finally:
        try:
            task.close_env()
        except Exception:
            pass


def classify_verdict(contexts: list[dict]) -> dict:
    paper = [c for c in contexts if c.get("paper_data", True) and c["label"] in {"low", "mid", "high"}]
    by = {c["label"]: c for c in paper}
    fmin = {k: by[k]["F_min_retention_first"] if k in by else None for k in ("low", "mid", "high")}
    retain025 = {k: by[k]["retain_at_0.25"] if k in by else None for k in ("low", "mid", "high")}
    wedging = []
    for c in paper:
        o025 = next((o for o in c["outcomes"] if abs(o["force_N"] - 0.25) < 1e-9), None)
        geom = (o025 or {}).get("lift_delay_geom") or {}
        lift = geom.get("after_lift") or {}
        wedging.append(
            {
                "label": c["label"],
                "pre": bool((c.get("pre_geom") or {}).get("multi_hull_wedging_any_finger")),
                "post_query": bool((c.get("post_query_geom") or {}).get("multi_hull_wedging_any_finger")),
                "after_lift": bool(lift.get("multi_hull_wedging_any_finger")),
                "end": bool((o025 or {}).get("end_multi_hull_wedging")),
                "split_like_after_lift": bool(lift.get("split_like_nx_pz") or lift.get("split_like_nx_nz_pz")),
            }
        )
    nums = [fmin[k] for k in ("low", "mid", "high")]
    ordered = all(x is not None for x in nums) and nums[0] > nums[1] > nums[2]
    weak_order = all(x is not None for x in nums) and nums[0] >= nums[1] >= nums[2] and nums[0] > nums[2]
    all_retain_025 = all(v == 1 for v in retain025.values())
    low_fails_low_f = (by.get("low") or {}).get("retain_at_0.25") == 0
    wrap_gone = all(not w["after_lift"] and not w["split_like_after_lift"] for w in wedging) if wedging else False
    wrap_present = any(w["after_lift"] or w["split_like_after_lift"] or w["end"] for w in wedging)
    if (ordered or (weak_order and low_fails_low_f)) and wrap_gone:
        verdict = "PASS"
        why = "original geometry: low μ_eff needs larger F; high μ_eff retains at smaller F; split-hull wrapping gone"
    elif all_retain_025 and (all(x == 0.25 for x in nums if x is not None) or len({x for x in nums if x is not None}) <= 1):
        verdict = "FAIL"
        why = "original geometry: F=0.25 still retains across μ_eff; retention threshold does not move with finger friction"
    else:
        verdict = "MIXED"
        why = "friction trend partially recovered and/or original multi-convex / 10 g mass still matter"
    if wrap_present and verdict == "PASS":
        verdict = "MIXED"
        why = "μ_eff ordering appeared but multi-hull wedging is still present on official collider"
    return {
        "verdict": verdict,
        "why": why,
        "F_min_retention": fmin,
        "retain_at_0.25": retain025,
        "ordered_strict": ordered,
        "ordered_weak": weak_order,
        "all_retain_at_0.25": all_retain_025,
        "wedging": wedging,
        "wrap_gone_after_lift": wrap_gone,
    }


def plot_curves(contexts: list[dict], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        dump_json(out.with_suffix(".plot_error.json"), {"error": repr(exc)})
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    colors = {"low": "#b42318", "mid": "#b54708", "high": "#027a48", "near_zero_diagnostic": "#667085"}
    for ctx in contexts:
        forces = [o["force_N"] for o in ctx["outcomes"]]
        ret = [o["retention_success"] for o in ctx["outcomes"]]
        task = [o["official_full_task_success"] for o in ctx["outcomes"]]
        label = f"{ctx['label']} μ_eff={ctx['mu_eff']:.4f}"
        color = colors.get(ctx["label"], "#344054")
        ls = "--" if ctx["label"] == "near_zero_diagnostic" else "-"
        axes[0].step(forces, ret, where="mid", color=color, linestyle=ls, label=label)
        axes[1].step(forces, task, where="mid", color=color, linestyle=ls, label=label)
    axes[0].set_title("Retention vs commanded F")
    axes[1].set_title("Full-task success vs commanded F")
    for ax in axes:
        ax.set_xlabel("F_cmd (N)")
        ax.set_ylabel("success")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Official deskbin geometry, finger-μ only, seed 200014")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--seed", type=int, default=200014)
    parser.add_argument("--prefer-episode", type=int, default=1)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--skip-near-zero", action="store_true")
    args = parser.parse_args()
    probe_payload = probe(args.seed, args.prefer_episode)
    if args.probe_only:
        return 0
    levels = probe_payload["finger_levels"]
    order = ["mid", "low", "high"]
    if not args.skip_near_zero:
        order.append("near_zero_diagnostic")
    contexts = []
    shared_episode = args.episode
    for name in order:
        ctx = run_level(args.seed, name, levels[name], OFFICIAL_FORCES, args.prefer_episode, shared_episode)
        contexts.append(ctx)
        if shared_episode is None and ctx["pre_gate"]["gate"] == "VALID":
            shared_episode = ctx["episode_id"]
            print(json.dumps({"event": "lock_episode", "episode_id": shared_episode, "from": name}), flush=True)
    verdict = classify_verdict(contexts)
    plot_curves(contexts, HERE / "retention_and_task_vs_F.png")
    summary = {
        "seed": args.seed,
        "shared_episode_after_first_valid": shared_episode,
        "deskbin_mu_fixed": probe_payload["default_deskbin_mu_runtime"],
        "n_shapes_official": probe_payload["n_shapes_after_official"],
        "n_shapes_split": probe_payload["n_shapes_before_split"],
        "levels": {k: levels[k] for k in order},
        "verdict": verdict,
        "contexts": [
            {
                "label": c["label"],
                "episode_id": c["episode_id"],
                "pre_gate": c["pre_gate"],
                "mu_finger": c["mu_finger"],
                "mu_eff": c["mu_eff"],
                "F_min_retention_first": c["F_min_retention_first"],
                "F_min_retention_stable": c["F_min_retention_stable"],
                "n_retention": c["n_retention"],
                "n_fulltask": c["n_fulltask"],
                "retain_at_0.25": c["retain_at_0.25"],
                "query_cr": c["query"]["contact_ratio"],
            }
            for c in contexts
        ],
    }
    dump_json(HERE / "SUMMARY.json", summary)
    dump_json(HERE / "VERDICT.json", verdict)
    print(json.dumps({"done": True, "verdict": verdict["verdict"], "why": verdict["why"], "F_min": verdict["F_min_retention"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
