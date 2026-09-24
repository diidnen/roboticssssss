# Done and freeze

These items have enough evidence to stop spending GPU on the same question. “Freeze” does not mean every associated headline claim is paper-final; it means rerunning the same experiment design is not justified.

| Line | Final disposition | Why no more GPU |
|---|---|---|
| task6 failure diagnosis | COMPLETE_NEGATIVE for method promotion | Direct boundary overconfidence is primary; Utility scale is secondary; estimator/pi0 not primary |
| global Fmax / force-scale diagnosis | COMPLETE_NEGATIVE; METHOD_PROMOTION=NO | global 6/8 N scale rescues only 1/5 task6 failures; absolute cost adds task1 collateral |
| current Joint auxiliary architecture | COMPLETE_NEGATIVE | original Joint and ranking-only DecisionAligned do not beat Direct; auxiliary objective remains harmful |
| WM residual as default controller | COMPLETE_NEGATIVE | Current-WM residual fails the preregistered gate; Direct Utility is retained |
| Probe-conditioned WM | COMPLETE_NEGATIVE for promotion | controlled corruption robustness does not become real Probe-error gain |
| hard/predictive verifier family | COMPLETE_NEGATIVE / ABLATION_ONLY | false-negative veto and no-valid-force behavior; not eligible as final controller |
| sample-efficiency 60-rollout question | COMPLETE_NEGATIVE | `60_ROLLOUTS_INSUFFICIENT_AND_WM_DOES_NOT_HELP`; no new simulator data needed |
| E4 frozen transfer DEV design | COMPLETE_NEGATIVE_DEV_ONLY | all 108 shards complete; conclusion is substantial task-specific data required |
| Utility causal/scale ablations | COMPLETE as DEV diagnostics | selector geometry and point/no-probe/GT diagnostics already computed; no method promotion |
| old Point-vs-Posterior archive ablation | DIAGNOSTIC_ONLY and frozen | existing posterior/sigma semantics are not final calibrated posterior evidence; rerunning old selector is useless |
| old 720 LocalLift label audit | COMPLETE_NEGATIVE for that dataset | 720/720 LocalLift labels positive; cannot train the intended contrast |
| Direct versus old Joint architecture | COMPLETE_NEGATIVE | Direct remains the selected feasibility model; old Joint is not joint friction x mass |

Do not confuse this list with fully paper-ready E0-E8 completion. The core unfinished lanes remain E3, E5, E6, E7, Mass, Joint physics, Boundary, and faithful baselines.
