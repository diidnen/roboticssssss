# Unseen-task interface audit

No architecture, checkpoint, normalizer, task code, or runtime selector has been changed. Fixed-5 screening never imports or invokes feasibility or the belief model.

The four training task identities coincide one-to-one with four objects. Therefore the learned four-way code cannot be identified from training as purely a task-form variable or purely an object variable. Current code explicitly indexes a task ID, so describing it as an object embedding would be an unsupported reinterpretation.

| Candidate interface | Dimensionally possible? | Scientific status |
|---|---|---|
| New numeric task ID through existing builder | No | Fails tuple lookup before inference |
| Reuse the code associated with the same training object on a new semantic task | Yes | Could be a predeclared category-C compatibility convention; assumes object-associated prior is appropriate and must be stated |
| All-zero code for an unseen object/task | Yes | Untrained categorical input; extrapolation, not a guaranteed neutral representation |
| Average the four task codes or predictions | Yes | New inference convention; must be frozen before results and changes the decision procedure |
| Delete task columns/weights | Changes model function | Not an exact constant-column migration like phase removal; not automatically allowed |
| Train a new code or feasibility model using new-form labels | Possible | Violates the primary zero-shot held-out-form test; becomes broader task coverage |

No candidate in the currently identified same-object set establishes a genuinely different maintained-grasp form: cream-cheese-to-bowl is conservatively same-form, and the multi-object or drawer-composite tasks require another grasp/hand mode. New-form rigid-object candidates change both object and task form. Their current unresolved code convention is a limitation even if VLA competence succeeds.

Other independent compatibility gates:

- `current_runtime_core.validate_call` permits only 0,1,5,6 and instantiates `libero_object`.
- Instruction tables and geometry manifests contain those same four tasks.
- `continuous_belief.rows` rejects task IDs outside the qualified four even though task ID is not a numeric belief feature.
- The probe tangent is derived using object-to-basket direction; an extraction/rotation task needs an explicitly frozen compatible query definition without using new-task outcomes.
- The geometry initializer assumes the original floor support and task-specific collision exports.
- The current state/snapshot/contact alias uses a `RigidObject`. Drawer handles belong to an articulation; cabinet-base force is not handle-grasp force.
- The original online terminal evaluator enforces basket containment, release and support; it is not a generic complete-task evaluator.

The VLA-only screen binds the existing six-body force sensors to each candidate rigid object and supplies the native language instruction. It leaves the authoritative source files intact. Its numeric IDs in provenance are routing identifiers only; no task one-hot is supplied because no feasibility model runs. Positive screen outcomes would still require a common P4B post-query handoff and adequate task evaluator before any formal AF branch.
