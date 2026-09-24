# FINAL_CLAIM_TO_EVIDENCE_MATRIX

Status: **ACTIVEFORCING_FULL_CLAIM_EXPERIMENT_DESIGN_READY**. “Locked test” names the required evidence location; it does not imply the experiment is complete today.

| Claim | Experiment | Task | Baseline / matched contrast | Metric | Locked test | Table / figure |
|---|---|---|---|---|---|---|
| Frozen π0 and unchanged nominal motion | E0 invariance audit | All promoted tasks | Same frozen π0 and nominal trajectory | Action/trajectory hashes; reset parity; phase parity | Fresh root-heldout audit before TEST | Table 1; Fig. 1 |
| Low-level controller is unchanged; adaptation is setpoint-only | E0 interface audit | All promoted tasks | Controller-only replay | Controller checksum; action diff; force audit | Locked controller checksum | Appendix A |
| Active physical interaction contains hidden-physics information | E1 causal query controls | T0/T1/T5/T6 | Active vs NoQuery/Sham/Query-Ignored | Friction MAE/rank; information gain; downstream SR/force | Fresh matched query-control TEST | Table 2; Fig. 2 |
| Query benefit is not state change or time | E1 causal query controls | Core and promoted tasks | Active vs duration-matched Sham and Query-Ignored | State delta; belief improvement; paired decisions | Same roots and initial states | Table 2; Fig. 2 |
| Physical identification is calibrated | E2 identification | Core plus mass/joint tasks | Vision; Physical; Vision+Physical; SysID; GT | MAE; ECE; Brier; NLL; coverage | Root-heldout locked TEST | Table 3; Fig. 3 |
| Physical belief changes downstream decisions | E2 + E5 ablation | E5 contexts | Point vs posterior, same Direct/checkpoint/candidates/budget | Choice accuracy; selected force; SR; under/excess | Fresh locked E5 TEST | Table 4; Fig. 4 |
| Full-Task supervision captures downstream failures | E3 | Qualified stage-complete task | Full-Task vs Local-Lift | Full/stage SR; downstream failure; agreement | Non-degenerate locked TEST | Table 5; Fig. 5 |
| Shared feasibility transfers across tasks | E4 | Core plus complementary tasks | Shared vs task-specific Direct | SR; frontier error; calibration; transfer gap | Task/root-heldout TEST | Table 6; Fig. 6 |
| ActiveForcing works from fresh reset to task end | E5 | T0/T1/T5/T6 plus promoted | Frozen π0 and matched baselines | Full SR; force; under/excess; fallback; failure stage | Fresh locked TEST | Table 7; Fig. 7 |
| Minimum-Reliable selector is distinct from Expected-Utility | E5 | Same contexts | Expected-Utility vs Minimum-Reliable | Full SR; mean force; under-force; excess; fallback | Same locked TEST contexts | Table 8; Fig. 8 |
| Decision-aware uncertainty reduces unnecessary re-query | E6 | Qualified long task | One; Always Two; Raw Gate; Decision-Aware; Oracle | SR; query count; precision; enrichment; under-force; force; latency | Fresh locked TEST | Table 9; Fig. 9 |
| Long task has model-independent delayed failure | E4 + E6 | Frozen long-horizon task | π0-independent low/high screening | Lift success; later stage failure; cause labels | Locked qualification and TEST | Table 10; Fig. 10 |
| Posterior-aware continuous planning improves decisions | E7 | Qualified off-grid task | Grid; Dense; Uniform; Stratified; Proposal/Posterior | SR; under/excess; regret; force; latency; K/sample sensitivity | Exact off-grid locked TEST | Table 11; Fig. 11 |
| Continuous differences exceed certified controller resolution | E0 + E7 | Interface and qualified task | Commanded vs measured force | Monotonicity; tracking; resolution; dead-zone; saturation; range | Frozen calibration before TEST | Appendix A; Fig. 12 |
| Mass is identifiable from a physical query | E2 + E8a | Mass-qualified rigid task | Vision; Physical; Vision+Physical; SysID; GT | Mass MAE; band/rank; cross-confusion | Fresh mass-query locked TEST | Table 12; Fig. 13 |
| Mass changes full-task force choice | E8a | Mass-sensitive dynamic task | Mass-aware vs nominal/friction-only | Low-force failure; high-force success; SR; force; under/excess | Fresh mass locked TEST | Table 13; Fig. 14 |
| Joint friction×mass adaptation is identifiable and useful | E8b | Joint-meaningful task | Point joint vs posterior joint and one-axis ablations | Friction MAE; mass MAE; cross-confusion; choice accuracy; SR | Fresh 3×3 locked TEST | Table 14; Fig. 15 |
| Shared physical identification transfers through feasibility | E4 + E8b | Core plus mass/joint task | Shared vs task-specific Direct | Held-out SR; calibration; joint regret | Task/root-heldout TEST | Table 15 |
| A faithful external reactive baseline is evaluated or omitted honestly | BASELINE audit | Same locked contexts | FORTE-Reactive only after fidelity pass; Tabero-Neutral internal | Fidelity; SR; force; under/excess; latency | Locked TEST after audit pass | Table 16; Appendix B |

No Abstract or Contributions statement may claim an item above unless its locked-test artifact exists and the corresponding table/figure is populated. A repaired false hypothesis is reported as `COMPLETE_NEGATIVE`, with the claim removed.
