# Grasp-surface-only friction isolation — decision record

Date: 2026-09-15. Seed 200014 frozen. Official 18/19 / Fig.B / checkpoints / v4 labels not overwritten.
Directory: `/media/volume/dasdas/exouser/af_dump_grasp_surface_friction_only_20260915`

## Gate

**MIXED.** Do **not** start 32×20 recapture.

Retention first-crossing on seed 200014 has the Coulomb order
`F_first(low=0.425)=1.25 > F_first(mid=0.575)=1.00 > F_first(high=0.85)=0.75`,
but raw retention-vs-F is not monotonic because commanded F is often not realized
(`measured squeeze << F`, contact_ratio ~0.1). Official dump success is a separate
downstream process: mid-μ dumps when held; high-μ holds an empty-ish bin (0 balls in
the official dustbin band).

## Material isolation (runtime)

Fingers contact `grasp_*` slabs at the experiment μ. Five trash balls contact `inner`
at fixed 0.3 (seed 200014 smoke). Bottom/table stay 0.3. Finger material 0.3 unchanged.

## 200003

Not executed: `UnStableError` (garbage) under the split collision. Not used to change
any criterion. Original 45-hull layout could spawn this seed; the split cannot.
