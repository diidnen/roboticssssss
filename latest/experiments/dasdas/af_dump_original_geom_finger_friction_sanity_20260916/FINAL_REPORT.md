# Original deskbin geometry + finger-friction-only sanity

Independent directory. Not official 18/19. Not Fig.B. Not a 32×20 recapture.

**Verdict: MIXED**

Restoring the official collider and varying only finger friction does recover a friction effect that split-hull isolation hid: F=0.25 drops for every μ_eff, and low μ_eff is much harder to retain than mid. It does **not** recover a clean Coulomb ladder `F_min(low) > F_min(mid) > F_min(high)`. High μ_eff is less stable than mid.

---

## 1. Official geometry restored

| | Split isolation (disallowed) | This run |
|---|---|---|
| Loader | runtime `install_patches` convex panels | official `create_actor(..., convex=True)` |
| File | generated `grasp_px/nx/pz/nz` + inner/bottom | `assets/objects/063_tabletrashbin/collision/base10.glb` |
| Visual | unchanged | `visual/base10.glb` (unchanged) |
| n_shapes | 6 named hulls | **45** convex pieces |
| Split metadata `_af_collision_names` | present | **absent** |
| Mass | 0.010 kg | 0.010 kg |
| `af_contact_friction` | painted deskbin | **unset** (deskbin not used as μ knob) |

Evidence: `GEOMETRY_BEFORE_AFTER.json`. Seed 200014 episode 1, `deskbin_id=10` (same episode as the previous PRE_VALID split diagnostic).

## 2. Deskbin shape list

45 `PhysxCollisionShapeConvexMesh` pieces from `collision/base10.glb`. No `grasp_px/nx/pz/nz`. Full AABB/centroid list in `GEOMETRY_BEFORE_AFTER.json` → `geometry_after_official.shapes`.

## 3. Fixed deskbin μ

Runtime-read official/default (GLB / PhysX default, **not invented**):

- deskbin μ = **0.30** on all 45 shapes (frozen every run)
- table μ ≈ **0.34**
- ball μ = **0.50**
- default finger μ before overwrite = **0.30**

Official 3-level experiments had painted the **deskbin** to 0.425/0.575/0.85 against finger 0.30. Here the deskbin stays at the asset default 0.30.

## 4–5. Finger μ and μ_eff (eAVERAGE)

`μ_eff = 0.5 * (μ_finger + 0.30)` chosen to match official 3-level averages:

| level | μ_finger | μ_deskbin | μ_eff | matches official |
|---|---|---|---|---|
| low | 0.425 | 0.30 | **0.3625** | 0.5*(0.30+0.425) |
| mid | 0.575 | 0.30 | **0.4375** | 0.5*(0.30+0.575) |
| high | 0.85 | 0.30 | **0.575** | 0.5*(0.30+0.85) |
| near-zero (diagnostic) | 0.001 | 0.30 | **0.1505** | not paper |

Left and right fingers (`fl_link7/8`, `fr_link7/8`) share the same μ. Gripper geometry unchanged.

## Isolation (runtime contacts)

Across all four conditions, measured contact μ_eff:

- finger ↔ deskbin: 0.3625 / 0.4375 / 0.575 / 0.1505 (tracks finger μ)
- ball ↔ deskbin: **0.40** = 0.5*(0.50+0.30), unchanged
- table ↔ deskbin: **0.40**, unchanged

## 6. Seed 200014 force sweep

Protocol: official grasp → 20-step PRE → query 4 N / 12 mm → snapshot fork F → original squeeze inner → scripted remainder.

All four μ levels: episode **1**, PRE **VALID** (bilateral pinch; nx+nx N/A), query CR 1.0 (near-zero 0.95).

Official 20-point F grid 0.25 … 5.0 N.

## 7. Retention vs F

| F | low 0.3625 | mid 0.4375 | high 0.575 | near-zero 0.1505* |
|---|---|---|---|---|
| 0.25 | 0 | 0 | 0 | 0 |
| 0.50 | 0 | 0 | 0 | 0 |
| 0.75 | 0 | **1** | **1** | 1 |
| 1.00–1.50 | 0 | 1 | 0 | 0 |
| 1.75 | 0 | 0 | 0 | 0 |
| 2.00–2.25 | 0 | 1 | 0 | 0 |
| 2.50 | **1** | 1 | 0 | 1 |
| 2.75–3.50 | 0 | 1 | 0 | 1 (2.75–3.50) |
| 3.75–4.25 | 0 | 1 | 0 / 1@4.0 / 0 | mixed |
| 4.50–5.00 | 0 / 1@4.75 / 0 | 1 | 1 / 0 / 1 | mixed |

\*near-zero is diagnostic, not paper.

n_retention / 20: low **2**, mid **17**, high **4**, near-zero **8**.

## 8. Full-task vs F (auxiliary)

Almost none. Mid 0/20, low 0/20, high 2/20 (4.5 N and 5.0 N), near-zero 8/20 (when it retained). Dump success is not the Coulomb readout.

## 9. F_min_retention (first crossing)

| μ_eff | F_min first | F_min stable | n_ret |
|---|---|---|---|
| low 0.3625 | **2.5 N** | none (non-monotone) | 2 |
| mid 0.4375 | **0.75 N** | none (fails 1.75) | 17 |
| high 0.575 | **0.75 N** | none | 4 |

Wanted: F_min(low) > F_min(mid) > F_min(high).
Got: 2.5 > 0.75 = 0.75. Low is clearly harder. High is not easier than mid.

## 10. F=0.25 N

**No μ_eff retains.** Including high and near-zero.

Actual settle squeeze at 0.25 N is ~0.00–0.04 N (inner deadzone / failed realize), not leftover query preload. After lift there is still a weak ±x pinch (~0.17–0.20 N per finger), then the bin is gone by delay.

This is the opposite of split-collision, where F=0.25 retained at every grasp μ including near-zero.

## 11. Original-geometry multi-hull wedging at F=0.25

PRE / query / lift: **opposing ±x pinch**, not split `nx+pz` / `nx+nz+pz`.

Typical PRE (all μ):

- `fl_link7`: 1–2 hulls, normals ≈ (−1, 0, 0)
- `fl_link8`: 1 hull, normals ≈ (+1, 0, 0)
- pair ≈ 57.5 mm, aperture ≈ 4.3 mm

A finger may touch two convex pieces, but those pieces share the **same** axis. Classifier `multi_hull_wedging` (2+ hulls AND 2+ axis buckets) is false at PRE, query, and F=0.25 lift.

Extra shape IDs in the global contact list are mostly balls/table, not finger wrapping.

## 12. Split vs original contact

| | Split (previous, disallowed) | Original (this run) |
|---|---|---|
| Hulls | 4 wall slabs + inner + bottom | 45 official convex pieces |
| F=0.25 fingers | both on `nx+pz`, later `nx+nz+pz` | `−x` vs `+x` pinch |
| F=0.25 retain | yes, all μ including μ≈0 | **no**, all μ |
| Corner wrap | yes | **no** at 0.25 lift |

Split isolation introduced a geometry confound. It cannot be used to claim the official dump grasp is friction-insensitive.

## 13. PASS / MIXED / FAIL

**MIXED**

- Not FAIL: 0.25 N is no longer universal retain; low μ_eff slips far more than mid.
- Not PASS: high μ_eff does not retain at smaller F more stably than mid; F_min is not strictly ordered; low/high curves are sparse; 10 g + realize floor + 45-hull contacts remain.
- Near-zero diagnostic still retains 8/20, often with end-of-rollout wedging flags — leftover form-closure on the official multi-convex, not a Coulomb zero.

## 14. Recapture?

**Do not recapture 32×20. Do not overwrite 18/19 or Fig.B.**

This is enough to retire split-collision as a friction readout, and to show that official geometry **can** drop at 0.25 N when finger μ is the only knob. It is not enough to claim a paper-ready Coulomb F_min(μ). If a follow-up happens, it should be more PRE_VALID seeds of **this** original-geom / finger-μ protocol, still sanity-scale.
