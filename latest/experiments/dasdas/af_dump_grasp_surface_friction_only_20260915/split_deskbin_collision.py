"""Split 063_tabletrashbin collision into grasp slabs / inner / bottom.

Original GLBs are never overwritten. Split meshes are written under the
experiment directory.

Runtime collision:
- grasp: thin outer convex slabs at the upper walls (finger contact)
- inner: nonconvex triangle mesh (trash-ball cavity)
- bottom: nonconvex triangle mesh (table contact)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

SRC = Path(
    "/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/"
    "RoboTwin/assets/objects/063_tabletrashbin/collision"
)
MODEL_IDS = (0, 3, 7, 8, 9, 10)
SIDES = ("px", "nx", "pz", "nz")
SLAB_THICKNESS = 0.05


def _as_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def wall_side(centers: np.ndarray, origin: np.ndarray) -> np.ndarray:
    delta = centers - origin
    x_dom = np.abs(delta[:, 0]) >= np.abs(delta[:, 2])
    sides = np.empty(len(centers), dtype=object)
    sides[x_dom & (delta[:, 0] >= 0)] = "px"
    sides[x_dom & (delta[:, 0] < 0)] = "nx"
    sides[~x_dom & (delta[:, 2] >= 0)] = "pz"
    sides[~x_dom & (delta[:, 2] < 0)] = "nz"
    return sides


def face_masks(mesh: trimesh.Trimesh) -> dict[str, np.ndarray]:
    centers = mesh.triangles_center
    normals = mesh.face_normals
    bounds = mesh.bounds
    origin = 0.5 * (bounds[0] + bounds[1])
    height = float(bounds[1, 1] - bounds[0, 1])
    y_frac = (centers[:, 1] - bounds[0, 1]) / max(height, 1e-9)
    radial = np.stack(
        [centers[:, 0] - origin[0], np.zeros(len(centers)), centers[:, 2] - origin[2]],
        axis=1,
    )
    radial_n = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9)
    outward = (normals * radial_n).sum(axis=1)
    ny = normals[:, 1]
    inner_floor = (ny > 0.4) & (y_frac < 0.25)
    outer_bottom = (ny < -0.4) & (y_frac < 0.20)
    upper = y_frac > 0.70
    grasp = (~inner_floor) & (~outer_bottom) & (outward >= 0) & upper
    inner = ((~outer_bottom) & (outward < 0) & (y_frac < 0.88)) | inner_floor
    bottom = outer_bottom
    leftover = (~grasp) & (~inner) & (~bottom)
    inner = inner | leftover
    sides = wall_side(centers, origin)
    masks: dict[str, np.ndarray] = {f"grasp_{side}": grasp & (sides == side) for side in SIDES}
    masks["inner"] = inner
    masks["bottom"] = bottom
    return masks


def submesh(mesh: trimesh.Trimesh, mask: np.ndarray) -> trimesh.Trimesh:
    part = mesh.submesh([np.where(mask)[0]], append=True)
    part.remove_unreferenced_vertices()
    return part


def thin_outer_hull(part: trimesh.Trimesh, side: str, thickness: float = SLAB_THICKNESS) -> trimesh.Trimesh:
    vertices = np.asarray(part.vertices, dtype=float)
    if side == "px":
        outer = vertices[vertices[:, 0] >= vertices[:, 0].max() - 0.02].copy()
        inset = outer.copy()
        inset[:, 0] -= thickness
    elif side == "nx":
        outer = vertices[vertices[:, 0] <= vertices[:, 0].min() + 0.02].copy()
        inset = outer.copy()
        inset[:, 0] += thickness
    elif side == "pz":
        outer = vertices[vertices[:, 2] >= vertices[:, 2].max() - 0.02].copy()
        inset = outer.copy()
        inset[:, 2] -= thickness
    else:
        outer = vertices[vertices[:, 2] <= vertices[:, 2].min() + 0.02].copy()
        inset = outer.copy()
        inset[:, 2] += thickness
    if len(outer) < 3:
        outer = vertices
        inset = vertices
    cloud = np.vstack([outer, inset])
    hull = trimesh.convex.convex_hull(cloud)
    return hull


def export_splits(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    convex_dir = out_dir / "convex_panels"
    convex_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(SRC),
        "runtime": "thin_grasp_convex_plus_inner_bottom_trimesh",
        "slab_thickness_mesh_units": SLAB_THICKNESS,
        "models": {},
    }
    for model_id in MODEL_IDS:
        source = SRC / f"base{model_id}.glb"
        mesh = _as_mesh(source)
        origin = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
        masks = face_masks(mesh)
        files = {}
        counts = {name: int(np.sum(mask)) for name, mask in masks.items()}
        pieces = []
        for side in SIDES:
            name = f"grasp_{side}"
            mask = masks[name]
            if not np.any(mask):
                continue
            part = submesh(mesh, mask)
            raw_path = out_dir / f"base{model_id}_{name}.obj"
            part.export(raw_path)
            hull = thin_outer_hull(part, side)
            hull_path = convex_dir / f"base{model_id}_{name}.obj"
            hull.export(hull_path)
            contains = bool(hull.contains([origin])[0])
            files[name] = {"triangles": str(raw_path), "convex": str(hull_path)}
            pieces.append(
                {
                    "name": name,
                    "role": "grasp",
                    "collision_type": "convex",
                    "n_faces": int(np.sum(mask)),
                    "path": str(hull_path),
                    "hull_volume": float(hull.volume),
                    "contains_origin": contains,
                }
            )
        for name, role in (("inner", "inner"), ("bottom", "bottom")):
            mask = masks[name]
            if not np.any(mask):
                raise RuntimeError(f"{source} missing {name} faces")
            part = submesh(mesh, mask)
            path = out_dir / f"base{model_id}_{name}.obj"
            part.export(path)
            files[name] = {"triangles": str(path)}
            pieces.append(
                {
                    "name": name,
                    "role": role,
                    "collision_type": "nonconvex",
                    "n_faces": int(np.sum(mask)),
                    "path": str(path),
                }
            )
        piece_path = convex_dir / f"base{model_id}_pieces.json"
        piece_path.write_text(json.dumps({"model_id": model_id, "pieces": pieces}, indent=2) + "\n")
        manifest["models"][str(model_id)] = {
            "source": str(source),
            "n_faces": int(len(mesh.faces)),
            "counts": counts,
            "files": files,
            "pieces_json": str(piece_path),
            "n_runtime_shapes": len(pieces),
        }
    return manifest


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    manifest = export_splits(here / "collision_splits")
    (here / "COLLISION_SPLIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: {"n": v["n_runtime_shapes"], "counts": v["counts"]} for k, v in manifest["models"].items()}, indent=2))
