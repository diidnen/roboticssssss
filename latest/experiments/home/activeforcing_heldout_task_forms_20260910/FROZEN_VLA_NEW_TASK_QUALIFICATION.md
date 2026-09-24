# Frozen VLA new-task qualification

**No new task passed admission.** These are native-reset Fixed-5 competence screens, not AF tests and not supplied-grasp downstream tests. Each task was stopped after its second valid failure under the rule frozen before VLA calls. Unrun contexts are not counted as failures. The empirical outcome rate is based only on attempted contexts; the 5/6 rule is an operational admission criterion, not a precise population-reliability estimate.

| Task | Candidate form | Success / attempted | Unrun | Best possible out of six | Established grasps | Decision |
|---|---|---:|---:|---:|---:|---|
| libero_10:5 — Book to back compartment | FORM_C_CANDIDATE | 0/2 | 4 | 4/6 | 0 | Not admitted |
| libero_goal:9 — Wine bottle to rack | FORM_C_CANDIDATE | 0/2 | 4 | 4/6 | 0 | Not admitted |
| libero_spatial:4 — Bowl from open drawer to plate | FORM_D_CANDIDATE | 0/2 | 4 | 4/6 | 0 | Not admitted |

The executed cases used development root 5100 at object-side friction 0.30 and 0.50. Root 6100 and the other friction conditions remain unrun after futility stopping. No fresh formal roots were exposed. The frozen VLA checkpoint digest was checked by the dedicated server and every inference receipt. All arm targets came from online inference at the existing ten-step cadence. Fixed-5 used the existing six-body sensor correction, force servo, and VLA-controlled opening rule. No feasibility or belief model was called.

The observed failure type is available per context in `FROZEN_VLA_QUALIFICATION_RESULTS.csv` and raw `qualification/*/QUALIFICATION_RESULT.json`. In this screen, failure to acquire a grasp within 200 VLA steps is separately labeled `VLA_PREGRASP_FAILURE`; it is not a force-related drop, release failure, geometric insertion failure, or evidence of failed feasibility transfer. There were no observed supplied-grasp downstream trials. The initial-state requirement is stronger than the method's assumed established-grasp handoff, so these negative screens cannot rule out competent downstream execution if a validated grasp is supplied.

Four native scenes passed config/observation preflight. The original extraction reset ignored the task's `joint_pos_range`, visibly leaving the intended drawer closed. Before any extraction VLA outcome, `extraction_reset_compatibility.py` restored the source-specified top joint default to -0.1523311883211136; a separate GPU preflight verified the named top joint and corrected frame. No threshold, force control, model, object root pose, or terminal criterion changed. The original queue was paused before extraction and its already completed book/wine outcomes retained.

Book/caddy and wine/rack native position/contact predicates do not certify compartment placement or target orientation. Even a positive native screen would need a stronger task-specific evaluator before formal admission. Drawer pulling cannot use the current rigid-object contact alias for an articulated handle. Factory insertion/gear/threading definitions exist but have a different controller/observation contract and no verified current-checkpoint qualification. Plate pushing and multi-grasp composites are out of scope.

Preflight teardown aborts occurred after complete scene/observation receipts; these were not task outcomes. The first sandbox preflight had no CUDA access and was repeated with GPU access. These infrastructure records remain in `preflight/` and `preflight_gpu/`. They do not enter the Fixed-5 denominator.

Sources: `QUALIFICATION_PROTOCOL.md`, `QUALIFICATION_EXECUTION_ADDENDUM.md`, `qualification/EXECUTION_FREEZE.json`, `qualification/EXTRACTION_COMPATIBILITY_FREEZE.json`, per-context RPC receipts, and `evidence/QUALIFICATION_QUEUE_COMPATIBILITY_PAUSE.json`.
