# Phase 1 audit — dump deskbin materials (no code/asset changes yet)

Date: 2026-09-15. Seed used only for inspection: 200014. Official 18/19 untouched.

## 1. Asset location

- Mesh/collision: `/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/RoboTwin/assets/objects/063_tabletrashbin/`
  - `collision/base{0,1,...,10}.glb` (physics)
  - `visual/base{id}.glb` (render only)
  - `model_data{id}.json` (scale 0.08, grasp contact frames)
- Task constructor: `RoboTwin/envs/dump_bin_bigbin.py` `load_actors()` → `create_actor(..., modelname="063_tabletrashbin", convex=True)`
- Loader: `RoboTwin/envs/utils/create_actor.py` `create_actor()` uses `add_multiple_convex_collisions_from_file` on the collision GLB. Not USD, not URDF.

`deskbin_id` is random in `{0,3,7,8,9,10}`. Seed 200014 loaded **id 10**.

## 2. Collision shapes (runtime, id 10)

45 separate `PhysxCollisionShapeConvexMesh` pieces. One shared PhysX material object. No per-face material index. Not a single mesh at runtime; a convex decomposition of one GLB.

## 3. Shared material?

Yes, before AF override: all 45 shapes use `physx.get_default_material()` = **static 0.3 / dynamic 0.3 / restitution 0.1**.

`set_actor_contact_friction(deskbin, μ)` then writes **the same μ onto every one of the 45 shapes**.

## 4. Current friction table (runtime)

| Body | shapes | static | dynamic | restitution | source |
|---|---|---|---|---|---|
| deskbin (before AF) | 45 convex | 0.3 | 0.3 | 0.1 | engine default (create_actor material=None) |
| deskbin (after AF μ=0.85) | 45 convex | 0.85 | 0.85 | 0.0 | `set_actor_contact_friction` |
| Aloha fingers `fl_link7/8` (left grasp) | 1 each | 0.3 | 0.3 | 0.1 | engine default |
| trash balls (5) | 1 sphere each | 0.5 | 0.5 | 0.0 | `scene.default_physical_material` |
| table | 5 boxes | 4×0.3 + 1×0.5 | same | 0.1 / 0.0 | mixed |
| ground | 1 | engine default | | | `add_ground` |

Scene also creates `default_physical_material` 0.5/0.5/0.0, but deskbin/fingers **do not use it**.

## 5. PhysX combine mode

SAPIEN 3.0.0b1 `PhysxMaterial` has **no combine-mode API**. PhysX 5 default is `PxCombineMode::eAVERAGE`.

Assumed: `μ_eff = 0.5 * (μ_a + μ_b)`.

## 6. Effective finger–deskbin μ

`μ_finger = 0.3` (fixed). Official dump object-side bins are **0.425 / 0.575 / 0.85** (not 0.30).

| object-side grasp μ | μ_eff (AVERAGE) |
|---|---|
| 0.425 | 0.3625 |
| 0.575 | 0.4375 |
| 0.85 | 0.575 |

v4 relabel also used 0.30: μ_eff = 0.30.

## Contact confound (runtime, after scripted left grasp)

Finger contacts and garbage contacts share hulls `{8, 42}`. Table contacts almost the whole convex set because the bin is still on the table during grasp. Existing convex pieces **cannot** isolate grasp vs inner vs bottom.

Collision GLB **does** have distinct inner vs outer triangles (~5 mm wall at scale 0.08). Phase 2 splits those faces, keeps the original GLB, and assigns materials on the new collision parts only.
