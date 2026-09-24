# Phase 2 collision split (original assets untouched)

Original GLBs remain in `RoboTwin/assets/objects/063_tabletrashbin/collision/`.
SHA256 copies: `original_collision_backup/`.

## Before (official / v4)

- Loader: `create_actor(..., convex=True)` → 45 `PhysxCollisionShapeConvexMesh` from one GLB.
- One shared PhysX material. `set_actor_contact_friction` writes the experiment μ onto **all** 45 hulls.
- Finger, balls, and table therefore share the same object-side μ.

## After (this directory only, monkeypatch)

Runtime shapes for each `deskbin_id`:

1. `grasp_{px,nx,pz,nz}` — thin outer convex slabs on the **upper** walls (finger interface). Variable μ.
2. `inner` — nonconvex triangle mesh of cavity / inner walls / leftover non-grasp faces. Fixed μ=0.3, restitution=0.1.
3. `bottom` — nonconvex triangle mesh of low downward faces. Fixed μ=0.3, restitution=0.1.

Visual mesh unchanged. Mass still 0.01 via Actor default. Official force-selection / P4 / EU / maxF=5 untouched.

Seed 200014 smoke (μ_object=0.85):

- fingers: bilateral contact, mostly `grasp_*` at 0.85; occasional inner-lip contact at 0.3
- 5 garbage spheres: **inner only** at 0.3
- table: bottom + inner leftover (fixed 0.3); some residual grasp-table pairs while the bin is still on the table

SAPIEN Entity rejects extra attributes; roles are stored on the `Actor` wrapper.
