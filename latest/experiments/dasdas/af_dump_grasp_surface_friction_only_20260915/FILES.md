# Files touched

No RoboTwin asset overwrite. No official 18/19 / Fig.B / checkpoint overwrite.

## New (this experiment only)

- `patch_grasp_surface.py` — monkeypatch `dump_bin_bigbin.create_actor` and `configure_activeforcing_physics`
- `split_deskbin_collision.py` — exports split meshes under `collision_splits/`
- `run_sweep_context.py` / `run_sweep_queue.py` / `summarize_sweep.py`
- `collision_splits/` triangle + thin convex grasp slabs
- `original_collision_backup/` SHA256 copies of original GLBs

## Unchanged

- `RoboTwin/envs/dump_bin_bigbin.py`
- `RoboTwin/assets/objects/063_tabletrashbin/**`
- `/media/volume/dasdas/exouser/af_dump_formal_liftclone_eu_fmax5_20260915`
- `/media/volume/dasdas/exouser/af_dump_liftstyle_feas_v4_relabel_20260915`
