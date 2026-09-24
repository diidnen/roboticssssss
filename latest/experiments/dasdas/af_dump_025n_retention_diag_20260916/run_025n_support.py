"""Helpers for the 0.25 N retention diagnostic. No official experiment writes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from patch_grasp_surface import deskbin_shapes
from slab_geometry import infer_mesh_scale, pose_transform_points, thin_axis_for_name


def json_dump(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def downsample(rows: list[dict], dt: float) -> list[dict]:
    if not rows:
        return []
    out = []
    next_t = rows[0]["t"]
    last = rows[0]
    for row in rows:
        last = row
        if row["t"] + 1e-9 >= next_t:
            out.append(
                {
                    "t": round(float(row["t"]), 4),
                    "phase": row["phase"],
                    "s": float(row["s"]),
                    "nl": float(row["nl"]),
                    "nr": float(row["nr"]),
                    "N_min": float(row["N_min"]),
                    "aperture_mm": None if row["aperture"] is None else 1000.0 * float(row["aperture"]),
                    "pair_mm": None if row["pair_dist"] is None else 1000.0 * float(row["pair_dist"]),
                    "z": float(row["obj_z"]),
                    "bilateral": int(bool(row["bilateral"])),
                    "nx_nx": int(bool(row["nx_nx"])),
                }
            )
            next_t += dt
    if out[-1]["t"] != last["t"]:
        out.append(
            {
                "t": round(float(last["t"]), 4),
                "phase": last["phase"],
                "s": float(last["s"]),
                "nl": float(last["nl"]),
                "nr": float(last["nr"]),
                "N_min": float(last["N_min"]),
                "aperture_mm": None if last["aperture"] is None else 1000.0 * float(last["aperture"]),
                "pair_mm": None if last["pair_dist"] is None else 1000.0 * float(last["pair_dist"]),
                "z": float(last["obj_z"]),
                "bilateral": int(bool(last["bilateral"])),
                "nx_nx": int(bool(last["nx_nx"])),
            }
        )
    return out


def _local_verts(shape) -> np.ndarray:
    raw = np.asarray(shape.get_vertices(), dtype=float)
    scale = infer_mesh_scale(raw)
    local = raw * scale
    getter = getattr(shape, "get_local_pose", None)
    if callable(getter):
        local = pose_transform_points(getter(), local)
    return local, scale


def hull_audit(task) -> dict:
    entity, shapes, roles, names = deskbin_shapes(task.deskbin)
    out = {}
    for shape, role, name in zip(shapes, roles, names):
        verts, scale = _local_verts(shape)
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
        size = hi - lo
        thin = thin_axis_for_name(str(name)) if role == "grasp" else int(np.argmin(size))
        out[str(name)] = {
            "role": role,
            "n_verts": int(len(verts)),
            "scale_applied": float(scale),
            "min": [float(x) for x in lo],
            "max": [float(x) for x in hi],
            "size": [float(x) for x in size],
            "thin_axis": int(thin),
            "thickness_m": float(size[thin]),
            "inplane_m": [float(size[i]) for i in range(3) if i != thin],
        }
    # pairwise AABB overlap for grasp slabs
    grasp = {k: v for k, v in out.items() if v["role"] == "grasp"}
    return {"shapes": out, "n_shapes": len(out), "grasp_names": list(grasp)}


def aabb_overlap(slabs: dict) -> dict:
    names = [k for k in slabs if not str(k).startswith("_")]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            lo_a = np.asarray(slabs[a]["min"], dtype=float)
            hi_a = np.asarray(slabs[a]["max"], dtype=float)
            lo_b = np.asarray(slabs[b]["min"], dtype=float)
            hi_b = np.asarray(slabs[b]["max"], dtype=float)
            inter = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
            if np.all(inter > 0):
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "overlap_extents_m": [float(x) for x in inter],
                        "overlap_volume_aabb": float(np.prod(inter)),
                    }
                )
    return {"any_overlap": bool(pairs), "pairs": pairs}


def material_rows(task) -> list[dict]:
    entity, shapes, roles, names = deskbin_shapes(task.deskbin)
    rows = []
    for shape, role, name in zip(shapes, roles, names):
        mat = shape.get_physical_material()
        rows.append(
            {
                "name": name,
                "role": role,
                "static_friction": float(mat.static_friction),
                "dynamic_friction": float(mat.dynamic_friction),
            }
        )
    return rows
