# Deprecated experiments — do not resume

| Experiment/artifact family | Status | Reusable part | Forbidden use |
|---|---|---|---|
| `activeforcing_final_closure_20260902_034923` and other minimum-passing closures | DO_NOT_RESUME | raw tuples/provenance where hashes match | final controller or paper headline |
| `FORTE_e6e7/activeforcing_e6e7_closure_20260902_052500` hard-rho E6/E7 | LEGACY_DIAGNOSTIC_ONLY | belief checkpoints, raw predictions after QA | rho-selected force, rho decision disagreement, final E6/E7 claim |
| copied `activeforcing_full_claim_closure_*/E6_E7` snapshots | DO_NOT_RUN_AS_NEW | documentation only | treating copies as independent experiments |
| old E7 `gnp_style_continuous_20260830_125107` selector/generator result | SUPERSEDED | raw continuous data/checkpoints | legacy negative as final E7; nearest-grid replay |
| E5 legacy smokes around `041520` and `043100` | DIAGNOSTIC_ONLY | engineering adapter evidence | final V2 Utility E5 aggregate |
| legacy P5-S0-D fresh Q2F E2E | HISTORICAL_ONLY | historical context | current Direct all-task E5 evidence |
| old Q2F threshold / minimum-force families | DIAGNOSTIC_ONLY | raw response data | final selector |
| old neural `Joint`/`continuous_probe_joint` | REJECTED_FOR_E8B | friction-only raw data | calling it joint friction x mass evidence |
| old 720 LocalLift classifier plan | DO_NOT_RESUME | delayed-failure audit | training a non-degenerate LocalLift classifier |
| old root-scaling task0 TEST roots 5174-5183 | OBSERVED_NOT_SEALED | descriptive evidence | V2 locked TEST |
| old task0/864-style force-critical challenge line | RETROSPECTIVE_ONLY | frozen stress diagnostics | prospective final challenge confirmation |
| WM residual/default, hard verifier, predictive verifier | COMPLETE_NEGATIVE | ablation tables | re-entry into default controller without new preregistration |
| global-scale Utility promotion | COMPLETE_NEGATIVE | causal diagnosis | `ACTIVEFORCING_GLOBAL_SCALE_UTILITY` method promotion |

Global rule: any artifact containing `MIN_RELIABLE_RHO`, `rho_sel`, `minimum_passing`, or probability-threshold force selection must be labeled `LEGACY_DIAGNOSTIC_ONLY`. It may not populate final-method aggregates.
