# 1. Executive conclusion

**Held-out task-form transfer remains NOT_TESTED.** The authoritative model does contain an explicit four-way task one-hot. New-form scene definitions exist, but no screened task passed the frozen admission rule. Therefore the final task set is empty and no AF/Fixed formal branches were run. This is an upstream qualification limitation, not a demonstrated failure of feasibility transfer.

The native-reset screen adds a requirement the original method avoids by assuming an established grasp. Its negative outcomes do not answer whether the frozen VLA would succeed from a supplied common grasp. Do not present this report as a completed downstream task-form generalization experiment.

# 2. Current model-input audit

**Explicit task code: YES.** The phase-free `(8,64)` tensor contains six Cartesian command channels, four task-one-hot channels, candidate force, friction, duplicated 13D state and duplicated masks. Object velocity and lateral speed come from the simulator snapshot. Wrist orientation/rotation, target geometry and future VLA replans are not direct feasibility inputs. The three frozen models and pooled TRAIN normalizers are unchanged. `c_task` is paper notation, not a generic implemented semantic embedding. All 64 scalar channels and normalization values are listed in [CURRENT_FEASIBILITY_FEATURES.csv](CURRENT_FEASIBILITY_FEATURES.csv); the full audit is [CURRENT_FEASIBILITY_INPUT_AUDIT.md](CURRENT_FEASIBILITY_INPUT_AUDIT.md).

# 3. What generalization was already present

Root-held-out main evaluation: 96 contexts / 384 branches / eight roots. Exact-held-out in-support friction: 48 contexts / 144 branches / two roots. Both use training task IDs 0,1,5,6 and their same objects. Archived August 30 task LOTO exists, but is a different model/data/runtime with failed generalization gates within the same form. It does not establish current-model task or task-form transfer. See [EXISTING_GENERALIZATION_AUDIT.md](EXISTING_GENERALIZATION_AUDIT.md).

# 4. Available task inventory

The search ledger contains 842 inventory entries: 46 native Tabero config records, five other native task semantics, six identifier-only tasks, and 785 additional archived BDDL definitions/variants. These are availability tiers, not 842 runnable supported tasks. The configured Tabero evaluation list contains nine libero_object tasks. Local assembled HDF5s cover those nine objects plus bowl-to-stove, with 50 demonstrations each; these are not proof of current-checkpoint competence on new forms. Twenty-three Gym environment registrations include controller/sensor variants, which do not increase task-form diversity. See [ALL_AVAILABLE_TASKS.csv](ALL_AVAILABLE_TASKS.csv) and `evidence/task_search_scope.json` for roots, exclusions and search limits.

# 5. Task-form taxonomy

Native definitions provide lift/transport/place (A), insertion/constrained assembly (C), and pulling (D). Reorientation-heavy maintained grasp (B) is only a candidate interpretation of threading/rack tasks; no qualified B instance is established. Plate pushing is not grasped dragging (E). Knob manipulation (F) has unverified hand-control compatibility. Multi-object and close-after-place composites are excluded. Distances/rotations not measured from downstream execution remain unknown. New objects are not new forms. See [TASK_FORM_TAXONOMY.md](TASK_FORM_TAXONOMY.md) and [TASK_FORM_DIVERSITY_MATRIX.csv](TASK_FORM_DIVERSITY_MATRIX.csv).

# 6. VLA qualification

| Task | Candidate form | Success / attempted | Unrun | Best possible out of six | Established grasps | Decision |
|---|---|---:|---:|---:|---:|---|
| libero_10:5 — Book to back compartment | FORM_C_CANDIDATE | 0/2 | 4 | 4/6 | 0 | Not admitted |
| libero_goal:9 — Wine bottle to rack | FORM_C_CANDIDATE | 0/2 | 4 | 4/6 | 0 | Not admitted |
| libero_spatial:4 — Bowl from open drawer to plate | FORM_D_CANDIDATE | 0/2 | 4 | 4/6 | 0 | Not admitted |

All attempted screens used Fixed-5 and development roots, no AF. The rule was frozen at >=5/6; early rejection after two valid failures leaves four contexts unrun. No denominator of 0/6 is fabricated. Observed pregrasp failures do not identify downstream force sufficiency. See [FROZEN_VLA_NEW_TASK_QUALIFICATION.md](FROZEN_VLA_NEW_TASK_QUALIFICATION.md). The extraction reset compatibility fix was validated and frozen before its first VLA outcome.

# 7. Frozen held-out task-form set

`FINAL_HELDOUT_TASK_FORM_SET.json` contains **zero admitted tasks** and all qualification decisions. Its exact SHA256 is in `FINAL_HELDOUT_TASK_FORM_SET_SHA256.txt`. No task was selected using AF performance. No fresh formal roots were allocated or executed.

# 8. Training-overlap audit

All four original identities appear in current feasibility TRAIN (431 rows, 48 contexts; roots 5100,5101,5102,5106). Candidate book, wine bottle, bowl, drawer-handle and factory objects are absent as manipulated training objects. Their potential results would confound task form with object/native-mass differences and belong to category D. No clean category C task with a training-overlap object and a genuinely new single-grasp form was established. No new labels were added to training; no model was retrained. The explicit task-ID compatibility issue remains unresolved rather than silently removed or aliased.

# 9. Formal protocol

The conditional protocol is recorded in [FORMAL_PROTOCOL.md](FORMAL_PROTOCOL.md): three frozen in-support friction values × two fresh roots × Fixed-3/4/5/AF per qualified task, with common grasp/query/post-query state, identical online VLA cadence and evaluator. No mass intervention. It was **not executed** because admission failed. There is no selected new-task encoding, common P4B handoff, or finalized new-form evaluator to claim as validated.

# 10. Per-task-form results

| Form / evidence role | Tasks | Contexts | AF | Fixed-3 | Fixed-4 | Fixed-5 |
|---|---:|---:|---:|---:|---:|---:|
| Original A, archived main evaluation | 4 | 96 | 69/96 | 37/96 | 65/96 | 71/96 |
| New C / D candidates, formal experiment | 0 admitted | 0 | N/A | N/A | N/A | N/A |

Only the original-family row is an AF result. Candidate-screen failures cannot populate new-form AF success rates. For B, C, D, E and F, AF performance remains N/A, with zero formal contexts. No form can be said to transfer successfully or fail feasibility transfer from this run.

# 11. Force-requirement adaptation

NOT_TESTED on new forms: no matched Fixed-3/4/5 outcomes exist, so no requirement class or Spearman correlation is computed. The frozen analysis retains unresolved 000 and non-monotonic patterns separately; it does not force a lift-based threshold model onto contact-rich tasks.

# 12. AF vs Fixed-5 / Fixed-4

NOT_TESTED on new forms. The archived original-family AF result is 69/96 versus Fixed-5 71/96 and Fixed-4 65/96. Archived mean non-release measured squeeze is 3.519 N for AF versus 4.689 N for Fixed-5; these numbers must not be reused as transfer results. No new-form paired differences or confidence intervals can be estimated.

# 13. AF vs GT decision diagnostic

NOT_TESTED: new-task feasibility inference was deliberately not called during qualification. AF–GT force MAE is N/A. The formal diagnostic would replace inferred friction with true friction while retaining the same model/context/search; it would remain decision-only and explicitly privileged.

# 14. Generalized failure analysis

The screen failures occurred before an established grasp. They are VLA pregrasp failures under the specified start-state/budget, not AF grasp-retention failures. AF failure counts are zero only because there are zero AF trials; corresponding rates are N/A. No causal conclusion about task contact, release, physical belief or feasibility transfer follows. [GENERALIZED_FAILURE_TAXONOMY.md](GENERALIZED_FAILURE_TAXONOMY.md) defines retention, task contact, VLA execution, terminal geometry, release and other categories without assuming a lift phase.

# 15. Visual task diversity

`HELDOUT_TASK_FORM_VISUAL_AUDIT/` preserves native scene frames, the corrected extraction reset, source hashes and qualification sequences. `TASK_FORM_OVERVIEW_CONTACT_SHEET.png` is explicitly a **candidate scene audit**, not evidence of successful behaviors or an admitted final benchmark. There is no representative successful final-form rollout to show because the final set is empty. The original closed-drawer reset is retained as diagnostic evidence, not a valid extraction scene.

# 16. Supported / mixed / unsupported claims

| Claim | Status |
|---|---|
| CURRENT_MODEL_HAS_EXPLICIT_TASK_ID | SUPPORTED |
| ROOT_GENERALIZATION_ALREADY_EXISTS | SUPPORTED |
| HELDOUT_FRICTION_GENERALIZATION_ALREADY_EXISTS | SUPPORTED |
| HELDOUT_TASK_GENERALIZATION_PREVIOUSLY_EXISTS | MIXED |
| HELDOUT_TASK_FORM_GENERALIZATION_PREVIOUSLY_EXISTS | NOT_TESTED |
| NEW_TASK_FORMS_FOUND | SUPPORTED |
| FROZEN_VLA_COMPETENT_ON_NEW_FORMS | NOT_SUPPORTED |
| FEASIBILITY_ZERO_SHOT_TRANSFER_TO_NEW_FORMS | NOT_TESTED |
| FORCE_ADAPTATION_TRANSFER_TO_NEW_FORMS | NOT_TESTED |
| RELIABILITY_FORCE_TRADEOFF_TRANSFER_TO_NEW_FORMS | NOT_TESTED |
| GT_DECISION_FIDELITY_ON_NEW_FORMS | NOT_TESTED |
| GENERALIZATION_BEYOND_LIFT_TRANSPORT_PLACE | NOT_TESTED |
| GENERALIZATION_TO_REGRASP_TASKS | NOT_TESTED |
| UNIVERSAL_TASK_GENERALIZATION | NOT_SUPPORTED |

The claim audit distinguishes existence of historical LOTO work from positive current-model transfer. `NOT_TESTED` must not be rewritten as either success or failure of AF on new forms.

# 17. Main-paper recommendation

**Not main-paper-worthy as evidence of task-form generalization.** Correct the method/input description now: the implementation has an explicit task one-hot and privileged simulator velocity. Define full-task feasibility conceptually as terminal success of the complete downstream frozen-VLA behavior under the maintained grasp; label lift/transport/place/release as the original instantiation. Retain the supported root/friction claims and disclose the observed reliability–force tradeoff within that family.

Answers to the requested questions: Did we truly evaluate AF beyond lift/transport/place? **No.** Are the candidate forms absent from feasibility training? **Yes, as candidate/registered forms.** Did the frozen VLA already know them? **Not established; none passed the native-reset screen.** Did AF adapt force without retraining? **No new-form AF inference or branches were run.** Which forms transferred or failed? **Unknown; none reached the formal test.** Why? **Qualification and runtime/evaluator/handoff gates, including pregrasp failures and categorical unseen-task encoding.** Main-paper-worthy? **No for a transfer claim; useful as a scope/input audit.**

To answer the central scientific question, still required are source-backed supplied-grasp snapshots for genuinely different maintained-grasp tasks, task-specific terminal validation, Fixed-5 qualification from the common post-query handoff, a justified frozen unseen-task encoding, and then one matched AF/Fixed formal test on fresh roots. Do not lower this screen's threshold or rerun its negatives until positive. Such a supplied-grasp study must be prospectively versioned as a different qualification protocol and its distinction disclosed.
