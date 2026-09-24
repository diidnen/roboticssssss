"""Patch dump deskbin collision to grasp-surface-only variable friction.

Does not modify RoboTwin assets or official experiment directories.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPLIT_DIR = HERE / "collision_splits"
CONVEX_DIR = SPLIT_DIR / "convex_panels"
FIXED_MU = 0.3
FIXED_RESTITUTION = 0.1
ROLES = ("grasp", "inner", "bottom")
FINGER_TOKENS = ("fl_link7", "fl_link8", "fr_link7", "fr_link8")


def deskbin_shapes(actor):
    entity = actor.actor if hasattr(actor, "actor") else actor
    roles = list(getattr(actor, "_af_collision_roles", None) or [])
    names = list(getattr(actor, "_af_collision_names", None) or [])
    for component in entity.get_components():
        getter = getattr(component, "get_collision_shapes", None)
        if callable(getter):
            shapes = list(getter())
            if shapes:
                return entity, shapes, roles, names
    raise RuntimeError("deskbin has no collision shapes")


def _shape_key(shape) -> tuple:
    vertices = np.asarray(shape.get_vertices(), dtype=float)
    return (int(len(vertices)),) + tuple(np.round(vertices.mean(0), 5))


def apply_grasp_surface_materials(task, grasp_mu: float) -> dict:
    entity, shapes, roles, names = deskbin_shapes(task.deskbin)
    if len(roles) != len(shapes):
        raise RuntimeError(f"role/shape mismatch {roles!r} vs {len(shapes)} shapes")
    grasp_mat = task.scene.create_physical_material(float(grasp_mu), float(grasp_mu), 0.0)
    fixed_mat = task.scene.create_physical_material(FIXED_MU, FIXED_MU, FIXED_RESTITUTION)
    assigned = []
    for shape, role, name in zip(shapes, roles, names or [""] * len(shapes)):
        material = grasp_mat if role == "grasp" else fixed_mat
        shape.set_physical_material(material)
        assigned.append(
            {
                "name": name,
                "role": role,
                "static_friction": float(shape.get_physical_material().static_friction),
                "dynamic_friction": float(shape.get_physical_material().dynamic_friction),
                "restitution": float(shape.get_physical_material().restitution),
            }
        )
    task._af_grasp_surface_mu = float(grasp_mu)
    task._af_fixed_surface_mu = FIXED_MU
    return {"n_shapes": len(shapes), "assigned": assigned}


def contact_material_audit(task) -> dict:
    entity, shapes, roles, names = deskbin_shapes(task.deskbin)
    keys = [_shape_key(shape) for shape in shapes]

    def shape_meta(shape):
        try:
            key = _shape_key(shape)
        except Exception:
            return None
        for index, candidate in enumerate(keys):
            if candidate == key:
                material = shape.get_physical_material()
                return {
                    "role": roles[index],
                    "name": names[index] if index < len(names) else "",
                    "static_friction": float(material.static_friction),
                    "dynamic_friction": float(material.dynamic_friction),
                    "restitution": float(material.restitution),
                }
        return None

    buckets = {"finger_grasp_roles": [], "garbage_roles": [], "table_roles": []}
    details = {"finger": [], "garbage": [], "table": []}
    for contact in task.scene.get_contacts():
        body_names = []
        for body in contact.bodies:
            try:
                body_names.append(body.entity.name)
            except Exception:
                body_names.append("?")
        joined = " ".join(body_names)
        if "063_tabletrashbin" not in joined:
            continue
        metas = [shape_meta(shape) for shape in contact.shapes]
        roles_here = [meta["role"] for meta in metas if meta]
        if any(token in joined for token in FINGER_TOKENS):
            buckets["finger_grasp_roles"].extend(roles_here)
            details["finger"].append({"bodies": body_names, "shapes": [m for m in metas if m]})
        if "garbage" in joined:
            buckets["garbage_roles"].extend(roles_here)
            details["garbage"].append({"bodies": body_names, "shapes": [m for m in metas if m]})
        if any(name == "table" for name in body_names):
            buckets["table_roles"].extend(roles_here)
            details["table"].append({"bodies": body_names, "shapes": [m for m in metas if m]})
    summary = {}
    for key, values in buckets.items():
        summary[key] = {
            "counts": {role: int(values.count(role)) for role in ROLES},
            "n": len(values),
        }
    summary["details"] = details
    return summary


def load_pieces(split_dir: Path, model_id: int) -> list[dict]:
    path = Path(split_dir) / "convex_panels" / f"base{int(model_id)}_pieces.json"
    if not path.exists():
        path = CONVEX_DIR / f"base{int(model_id)}_pieces.json"
    payload = json.loads(path.read_text())
    pieces = list(payload["pieces"])
    if not pieces:
        raise RuntimeError(f"no convex panels for deskbin id {model_id}")
    return pieces


def install_patches(split_dir: Path | None = None) -> None:
    split_dir = Path(split_dir or SPLIT_DIR)
    import envs.utils as utils_pkg
    import envs.dump_bin_bigbin as dump_mod
    from envs.utils.create_actor import Actor, get_glb_or_obj_file, preprocess
    from pathlib import Path as _Path

    original = dump_mod.create_actor

    def create_actor(scene, pose, modelname, scale=(1, 1, 1), convex=False, is_static=False, model_id=0):
        if modelname != "063_tabletrashbin":
            return original(scene, pose, modelname, scale=scale, convex=convex, is_static=is_static, model_id=model_id)
        scene, pose = preprocess(scene, pose)
        modeldir = _Path("assets/objects") / modelname
        json_file = modeldir / f"model_data{model_id}.json"
        model_data = json.loads(json_file.read_text())
        scale = model_data["scale"]
        visual = get_glb_or_obj_file(modeldir / "visual", model_id)
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("static" if is_static else "dynamic")
        fixed = scene.create_physical_material(FIXED_MU, FIXED_MU, FIXED_RESTITUTION)
        pieces = load_pieces(split_dir, int(model_id))
        roles = []
        names = []
        for piece in pieces:
            collision_type = piece.get("collision_type", "convex")
            if collision_type == "nonconvex":
                builder.add_nonconvex_collision_from_file(filename=piece["path"], scale=scale, material=fixed)
            else:
                builder.add_convex_collision_from_file(filename=piece["path"], scale=scale, material=fixed)
            roles.append(piece["role"])
            names.append(piece["name"])
        builder.add_visual_from_file(filename=str(visual), scale=scale)
        mesh = builder.build(name=modelname)
        mesh.set_name(modelname)
        mesh.set_pose(pose)
        actor = Actor(mesh, model_data)
        actor._af_collision_roles = list(roles)
        actor._af_collision_names = list(names)
        return actor

    dump_mod.create_actor = create_actor
    utils_pkg.create_actor = create_actor

    original_configure = dump_mod.dump_bin_bigbin.configure_activeforcing_physics

    def configure_activeforcing_physics(self):
        self._af_force_intervention_started = False
        self._af_force_trace_step = 0
        self._af_force_trace = []
        self._af_no_contact_samples = 0
        self._af_ever_lifted = False
        self._af_query_context = None
        if self.af_object_mass_kg is not None:
            mass_kg = float(self.af_object_mass_kg)
            if not np.isfinite(mass_kg) or mass_kg <= 0:
                raise ValueError("af_object_mass_kg must be finite and positive")
            self.deskbin.set_mass(mass_kg)
        if self.af_contact_friction is not None:
            apply_grasp_surface_materials(self, self.af_contact_friction)

    dump_mod.dump_bin_bigbin.configure_activeforcing_physics = configure_activeforcing_physics
    return original_configure, original
