# Final pooled predictive-verifier report

In the truly shared four-task setting, fully root-heldout **ActiveForcing-Direct-Pooled reaches macro SR 0.951**. The strict pooled Predictive Verifier reaches 0.850 with coverage 0.884, under-force 0.035, and mean force 3.912 N, versus Direct under-force 0.035 and mean force 4.033 N. The frozen development classification is **PREDICTIVE_VERIFIER_POOLED_DEVELOPMENT_FAIL**. Therefore the main method at this evidence stage is **ActiveForcing-Direct**. No untouched TEST was read.

## Direct answers

1. **Shared pooled Direct SR:** macro 0.951; pooled and per-task values are in `POOLED_DIRECT_RESULTS.csv` and `POOLED_SEARCH_POLICY_ABLATION.csv`.
2. **Is the verifier better?** Strict verifier delta is -0.102 macro SR. Promotion gate: **FAIL**.
3. **Multiple-task consistency:** tasks with positive SR gain are []; the rule requires at least two and all three seed directions to agree.
4. **Why did old 87.5% → 93.8% occur?** The old viewed population had 3 force-rescuable cases among 24 context episodes, Direct SR 0.875, and Fixed-Max SR 0.958. It was a population with a different near-frontier/failure composition, and Max fallback supplied part of the observed improvement; it was not matched fully OOF evidence.
5. **Does new CV contain force-rescuable failures?** It contains 5 among 144 context-repeat episodes; Fmax is not always successful in 2. Thus only the force-rescuable subset is in principle recoverable by upward force search.
6. **Smarter than +1?** One-step macro SR is 0.986; verifier is 0.850. Selective-verifier evidence requires verifier > one-step, which is not present.
7. **PhysicsOnly value:** macro verifier AUROC current=0.828, PhysicsOnly=0.830; control SR current=0.850, PhysicsOnly=0.824; trajectory standardized MAE current=0.233, PhysicsOnly=0.233. Interpret it as physical trajectory signal only if matched control behavior remains close; otherwise current WM is outcome-shaped.
8. **World-model role:** **Neither is required in the promoted main method; Joint remains an unadjudicated training auxiliary/diagnostic**. This experiment rejects the runtime verifier but does not independently establish or refute Joint's training-time auxiliary value. Joint training changes the representation during training; Predictive Verifier is a separate runtime Direct→trajectory→binary check→upward-search mechanism. They are not one “world-model gain.”
9. **Main method:** **ActiveForcing-Direct** pending any separately frozen untouched pooled TEST. A development PASS would still not be a paper claim.

## Per-task results and evidence limits

All requested tables report task0/task1/task5/task6, macro average, and pooled aggregate. task1 retains the material 140/180 reconstructed-label caveat. The model uses authoritative GT physics and no visual/Probe input, so these results are not final Active Probe E2E evidence. Receding-H8 was not run: the one-shot promotion gate failed. Root-scaling and its untouched TEST were neither modified nor read.
