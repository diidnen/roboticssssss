"""Grasp-slab local frames and edge distance. Debug only; official meshes unchanged."""
from __future__ import annotations

import numpy as np

from patch_grasp_surface import deskbin_shapes


def quat_to_rpy(quat_wxyz) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr, cosr))
    sinp = 2.0 * (w * y - z * x)
    pitch = float(np.arcsin(np.clip(sinp, -1.0, 1.0)))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny, cosy))
    return roll, pitch, yaw


def quat_geodesic_deg(q_a, q_b) -> float:
    a = np.asarray(q_a, dtype=float)
    b = np.asarray(q_b, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.abs(np.clip(np.dot(a, b), -1.0, 1.0)))
    return float(np.degrees(2.0 * np.arccos(dot)))


def invert_quat_wxyz(quat):
    w, x, y, z = [float(v) for v in quat]
    return np.array([w, -x, -y, -z], dtype=float)


def mul_quat(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def rotate_inverse(quat_wxyz, vec):
    w, x, y, z = [float(v) for v in quat_wxyz]
    qvec = np.array([x, y, z], dtype=float)
    v = np.asarray(vec, dtype=float)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def world_to_object(obj_pose, world_p):
    return rotate_inverse(obj_pose.q, np.asarray(world_p, dtype=float) - np.asarray(obj_pose.p, dtype=float))


def thin_axis_for_name(name: str) -> int:
    if name in {"grasp_px", "grasp_nx"}:
        return 0
    if name in {"grasp_pz", "grasp_nz"}:
        return 2
    return 0


def rotate_vector(quat_wxyz, vec):
    w, x, y, z = [float(v) for v in quat_wxyz]
    qvec = np.array([x, y, z], dtype=float)
    v = np.asarray(vec, dtype=float)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def pose_transform_points(pose, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    rotated = np.stack([rotate_vector(pose.q, row) for row in pts], axis=0)
    return rotated + np.asarray(pose.p, dtype=float).reshape(1, 3)


def infer_mesh_scale(verts: np.ndarray) -> float:
    extent = float(np.max(np.ptp(verts, axis=0)))
    if extent > 0.5:
        return 0.08
    return 1.0


def build_slab_catalog(task) -> dict:
    entity, shapes, roles, names = deskbin_shapes(task.deskbin)
    obj = task.deskbin.get_pose()
    catalog = {}
    for shape, role, name in zip(shapes, roles, names):
        if role != "grasp":
            continue
        raw = np.asarray(shape.get_vertices(), dtype=float)
        scale = infer_mesh_scale(raw)
        local_pts = raw * scale
        getter = getattr(shape, "get_local_pose", None)
        if callable(getter):
            local_pts = pose_transform_points(getter(), local_pts)
        lo = local_pts.min(axis=0)
        hi = local_pts.max(axis=0)
        catalog[str(name)] = {
            "min": lo.tolist(),
            "max": hi.tolist(),
            "center": (0.5 * (lo + hi)).tolist(),
            "size": (hi - lo).tolist(),
            "thin_axis": thin_axis_for_name(str(name)),
            "vertex_scale_applied": scale,
            "n_verts": int(len(raw)),
        }
    if not catalog:
        raise RuntimeError("no grasp slabs in deskbin catalog")
    catalog["_object_pose_at_build"] = {
        "p": [float(x) for x in obj.p],
        "q": [float(x) for x in obj.q],
    }
    return catalog


def inplane_axes(thin_axis: int) -> tuple[int, int]:
    return tuple(axis for axis in (0, 1, 2) if axis != int(thin_axis))  # type: ignore[return-value]


def local_uv_and_edge(p_obj: np.ndarray, slab: dict) -> dict:
    lo = np.asarray(slab["min"], dtype=float)
    hi = np.asarray(slab["max"], dtype=float)
    thin = int(slab["thin_axis"])
    u_axis, v_axis = inplane_axes(thin)
    u = float(p_obj[u_axis])
    v = float(p_obj[v_axis])
    dist_u = min(u - float(lo[u_axis]), float(hi[u_axis]) - u)
    dist_v = min(v - float(lo[v_axis]), float(hi[v_axis]) - v)
    return {
        "u": u,
        "v": v,
        "u_axis": int(u_axis),
        "v_axis": int(v_axis),
        "thin": float(p_obj[thin]),
        "edge_u": float(dist_u),
        "edge_v": float(dist_v),
        "edge": float(min(dist_u, dist_v)),
        "u_lo": float(lo[u_axis]),
        "u_hi": float(hi[u_axis]),
        "v_lo": float(lo[v_axis]),
        "v_hi": float(hi[v_axis]),
    }
