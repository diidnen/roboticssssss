# Grasp-surface-only friction isolation — 2026-09-16

Official 18/19, Fig.B, checkpoints, query, and 32×20 were not overwritten. No retrain.

## 1. Files

Not modified:
- RoboTwin `assets/objects/063_tabletrashbin` visual/collision GLBs
- official 18/19, Fig.B, v4 labels, query, controller

Reused (runtime patch only):
- `af_dump_grasp_surface_friction_only_20260915/patch_grasp_surface.py`
- `.../collision_splits/convex_panels/` (4 grasp slabs + inner + bottom)
- `contact_attr/collision_filter.py` — **new this round vs 20260915 sweep**

New:
- `af_dump_grasp_surface_friction_only_20260916/run_isolation_sweep.py`
- this directory's logs

## 2. Collision / material before vs after

Before (official `dump_bin_bigbin`):
- `collision/base{id}.glb` via `add_multiple_convex_collisions_from_file`
- `configure_activeforcing_physics` calls `set_actor_contact_friction(deskbin, μ)` so **every** hull gets the experiment μ

After (runtime `create_actor` patch, visual GLB unchanged):
- grasp_px, grasp_nx, grasp_pz, grasp_nz: convex, **variable μ**
- inner: nonconvex, **fixed 0.3**
- bottom: nonconvex, **fixed 0.3**
- collision filter: fingers ignore inner/bottom; `fl_link6` ignores the whole bin

## 3–4. Variable vs fixed

Variable: outer grasp slabs only.

Fixed: inner, bottom, table (~0.34–0.50), trash balls (0.50), fingers (0.30 measured on fl_link7/8).

## 5. PhysX combine

SAPIEN 3 `PhysxMaterial` has no combine-mode API. PhysX 5 default is **eAVERAGE**.

## 6. Object-side μ and μ_eff (finger 0.30 × AVERAGE)

| band | object-side grasp μ | μ_eff average | min | multiply |
|---|---|---|---|---|
| low | 0.425 | 0.3625 | 0.30 | 0.1275 |
| mid | 0.575 | 0.4375 | 0.30 | 0.1725 |
| high | 0.85 | 0.575 | 0.30 | 0.255 |

Official 3-level dump set is **0.425 / 0.575 / 0.85**, not 0.30 / 0.575 / 0.85.

## Runtime contact audit (PRE, after filter)

Finger–bin: **grasp 4, inner 0, bottom 0** (20260915 leak is gone).

Balls: **inner 5, grasp 0**, inner μ stays 0.3 at both 0.425 and 0.85.

## 7–11. Seed 200014 force sweep (official 0.25–5.0 N)

PRE_VALID required. Episode 0 is INVALID at μ=0.425 (open 45 mm). Episode 1 is INVALID at μ=0.85. **No episode is PRE_VALID at all three μ**, so μ is still entangled with geometry.

| μ | episode | retention bits | F_min_ret first | F_min_ret stable | full-task | mean balls |
|---|---|---|---|---|---|---|
| 0.425 | 1 | 11011100001111111101 | 0.25 | none | 0/20 | 0.15 |
| 0.575 | 0 | 20×1 | 0.25 | 0.25 | 20/20 | 5.0 |
| 0.575 | 1 | 11111110111111111111 | 0.25 | none | 0/20 | 0.10 |
| 0.85 | 0 | 20×1 | 0.25 | 0.25 | 0/20 | 0.0 |

Coulomb order `F_min(low) > F_min(mid) > F_min(high)`: **no**. All first-crossings are 0.25 N.

Same episode_id=0: mid μ dumps all 5 balls at every F; high μ dumps 0 balls while retaining the bin. Inner μ is fixed, so dump-out still follows experimental μ through grasp/pour kinematics.

## 12. Seed 200003

`UnStableError: garbage` at env init (same as 20260915). Split inner collision does not spawn this seed. Recorded as geometry-dominated spawn failure, not used to change criteria.

## 13–14. Verdict

**FAIL** on the only question that matters:

> After changing only finger–deskbin grasp-interface friction, does retention recover “low μ needs larger F”?

No. When a wall pinch exists, retention already appears at 0.25 N at every μ. Isolation of inner contacts **works** (fingers hit grasp only; balls hit inner@0.3), but that is not enough for a Coulomb retention threshold.

**Do not recapture 32×20** on this setup. Next would have to change something that actually moves F_min (not query, not another unlabeled 32×20).
