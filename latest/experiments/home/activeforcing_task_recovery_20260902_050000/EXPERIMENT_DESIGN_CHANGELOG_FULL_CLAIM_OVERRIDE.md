# EXPERIMENT_DESIGN_CHANGELOG_FULL_CLAIM_OVERRIDE

Status: **ACTIVEFORCING_FULL_CLAIM_EXPERIMENT_DESIGN_READY**

This records the change from the recovered friction-core / optional-extension design to the full ActiveForcing claim-closure design.

| Previous assumption | Full-claim override | Design consequence |
|---|---|---|
| Mass was optional or future work | **Mass is a required final experimental axis** | Added E8a mass identifiability, mass-sensitive qualification, identification baselines, full-task adaptation, and fresh E2E. |
| Friction-only physics was sufficient | Joint physics is (z=(\text{friction},\text{mass})) | Added E8b 3×3 factorial, cross-confusion, joint force-choice, and downstream SR. |
| The old Joint neural architecture could represent joint physics | The old Joint architecture remains rejected | Joint evidence uses explicit physical belief plus point/posterior marginalization. |
| Failed DEV gates could drop re-query or continuous planning | Failed gates require TRAIN/DEV diagnosis and repair | E6/E7 are required rows with repair paths and `COMPLETE_NEGATIVE` handling. |
| Expected utility alone defined the final decision | Final runtime claim is minimum reliable force | Added `rho_sel`, Minimum-Reliable selector, fallback, and matched Expected-Utility ablation. |
| Point physical estimate was sufficient | Calibrated physical belief is required | Added point vs posterior-aware marginalization with identical Direct/checkpoint/candidates/budget. |
| NoQuery vs Active Query was sufficient | Separate information from state/timing effects | Added NoQuery-Prior, Sham/Duration-Matched, Query-Ignored, and Active Query controls. |
| Continuous planning could be abandoned after interpolation failure | Continuous planning must be completed or negatively concluded | Added five planner arms, matched K, sample budget, exact off-grid execution, and interface calibration. |
| Continuous force strata were enough for final Direct | Exact candidate semantics and fresh E2E are required | Existing 720 stays archive-compatible OOF only; E5 requires fresh reset-to-end data. |
| Post-lift candidates stood in for delayed failure | A final task must show model-independent delayed failure | Added long-horizon qualification, checkpoints, and cause-specific failure logging. |
| Four short basket tasks represented the benchmark | Taxonomy must cover complementary regimes | Added A basic, B delayed horizon, C mass-sensitive, D joint friction×mass, E breadth. |
| Shared transfer could be claimed from four short tasks | Transfer must span complementary regimes | E4 now freezes task qualification and shared vs task-specific transfer. |
| LocalLift diagnostic could support supervision claim | Current LocalLift labels are degenerate | E3 is blocked on the archive and requires new stage-complete labels. |
| Empirical frontier could inform test-time selection | Runtime and evaluation thresholds are distinct | `rho_sel` is TRAIN/DEV-frozen; `rho_env` is evaluation-only. |
| A third-party reactive method could be represented by a surrogate | External baseline must be faithful or omitted | Added FORTE-Reactive fidelity audit; no surrogate may be called a reproduction. |
| Existing 720 branches could cover the final matrix | Reuse is limited to scientifically valid OOF | Added targeted collection for long horizon, second query, off-grid, mass, joint, and fresh E2E. |
| Claim prose could precede evidence | Every claim needs an exact evidence mapping | Rebuilt claim matrix with experiment, task, baseline, metric, locked test, table/figure. |

## Regenerated files

- `FINAL_E0_E8_EXPERIMENT_MATRIX.md` and `.csv`
- `FINAL_CLAIM_TO_EVIDENCE_MATRIX.md` and `.csv`
- `FINAL_TASK_TAXONOMY.md`
- `FINAL_EXPERIMENT_ROLLOUT_BUDGET.md`
- `PAPER_EXPERIMENT_SECTION_BLUEPRINT.md`
- `PAPER_TABLE_FIGURE_PLAN.md`
- This changelog

## Evidence boundary retained

The 720-row friction archive remains protected and is not relabeled. Current mass P4-B evidence supports query observability on development/heldout roots, not downstream mass adaptation. Current continuous and fresh-E2E reports remain incomplete/failed at their documented gates. The regenerated design closes the requested scope procedurally while preserving those evidence boundaries.
