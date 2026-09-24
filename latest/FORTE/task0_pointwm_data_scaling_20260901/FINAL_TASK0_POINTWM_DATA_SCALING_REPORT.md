# Final Task0 Point-WM Data-Scaling Report

## Technical summary

**Classification: `WORLD_MODEL_FRONTIER_SIGNAL_WITHOUT_CONTROLLER_GAIN`.** This is a frozen **GT-PHYSICS SCALING DIAGNOSTIC** on the previously evaluated **FROZEN HELD-OUT TASK0 TEST**, not an untouched TEST result and not a deployable ActiveForcing controller claim.
Across 6→15→30→50 independent TRAIN roots, the three-seed mean Point-WM−Direct SR gains are S6 +0.00 pp, S15 -2.00 pp, S30 -8.67 pp, S50 -6.00 pp.
The diagnostic Spearman correlations with root count are ρ=-0.800 for ΔSR, ρ=+0.200 for Δutility, and ρ=+1.000 for Δfrontier (n=4 scale points; descriptive only).

## Direct answers

1. **Independent roots:** S6=6, S15=15, S30=30, S50=50, exactly nested by the frozen manifest.
2. **Real TRAIN branches:** S6=180, S15=270, S30=420, S50=620. Counts are not 60/150/300/500 because S6 has 18 friction contexts and later roots add one context each; every context has 10 branches.
3. **TEST:** 10 independent roots, 90 force cells, 450 real branches. Each controller selects one archived force/root and reads five repeats, so evaluation is exactly 50 controller episodes/method/seed. 0/10 roots are `NO_RELIABLE_FORCE` under the >=4/5 empirical rule and are excluded only from frontier/under-force denominators, never from SR or utility.
4. **Direct SR learning curve (three-seed mean ± population std):** S6 89.33% ± 0.94 pp; S15 90.00% ± 0.00 pp; S30 90.00% ± 0.00 pp; S50 96.67% ± 4.71 pp.
5. **Point-WM SR learning curve:** S6 89.33% ± 0.94 pp; S15 88.00% ± 1.63 pp; S30 81.33% ± 4.99 pp; S50 90.67% ± 5.25 pp.
6. **WM−Direct at each scale:** S6 ΔSR=+0.00 pp, Δutility=-0.0150, Δunder-force=+0.00 pp, Δfrontier=-0.1417 N; S15 ΔSR=-2.00 pp, Δutility=-0.0180, Δunder-force=+0.00 pp, Δfrontier=+0.0250 N; S30 ΔSR=-8.67 pp, Δutility=-0.0663, Δunder-force=+6.67 pp, Δfrontier=+0.1833 N; S50 ΔSR=-6.00 pp, Δutility=-0.0123, Δunder-force=+3.33 pp, Δfrontier=+0.2917 N. Positive Δfrontier means Point-WM is closer.
7. **Does WM gain increase with roots?** No systematic monotonic increase is supported by the four-point diagnostic. Spearman is descriptive because n=4; log(root) slopes are ΔSR -0.03616, Δutility -0.00779, and Δfrontier +0.20568.
8. **Replication of the pooled 60/120 pattern:** The larger-root Task0 curve does not reproduce a clean negative-to-positive late-emergence pattern. The pooled 60/120 evidence and this Task0 root-scaling diagnostic differ in task composition, root/context structure, and evaluation population, so this is replication of direction only, not a pooled-effect estimate.
9. **Does S50 Point-WM exceed Direct?** No on three-seed mean (Δ=-6.00 pp); prediction-level ensemble Δ=-12.00 pp.
10. **Single-seed stability or ensemble-only?** S50 Point-WM beats Direct in 0/3 single seeds. The ensemble is always reported separately and never substituted for the seed mean.
11. **Under-force:** at S50 Δ=+3.33 pp (Point-WM−Direct) over empirical-frontier-supported episodes.
12. **Utility:** at S50 Δ=-0.0123 realized utility.
13. **Closeness to empirical F*0.8:** at S50 Δfrontier=+0.2917 N; positive means Point-WM is closer. No Fmax oracle is substituted for unsupported roots.
14. **H8 trajectory prediction vs scale:** S6 MAE=0.061150; S15 MAE=0.056411; S30 MAE=0.049608; S50 MAE=0.050426.
15. **IE prediction vs scale:** S6 error=0.034015; S15 error=0.032916; S30 error=0.032749; S50 error=0.032057.
16. **Physics versus decision interface:** physics prediction improved (H8 MAE 0.061150→0.050426; IE error 0.034015→0.032057) and S50 frontier MAE improved by +0.2917 N, but SR and utility did not. Across the three S50 seeds there were 0 rescue episodes and 9 collateral-damage episodes out of 150 paired repeat-level evaluations. The evidence therefore locates the failure at the frozen physics-to-decision interface rather than at absence of learned physics signal.
17. **Old auxiliary signal vs current controller:** old Visual Joint frontier MAE was lower than old Direct at all four scales (0.277/0.207/0.245/0.253 N versus 0.297/0.240/0.307/0.312 N). That establishes an old auxiliary boundary signal, not current Point-WM controller gain. The current Direct/Point-WM frontier and controller results are listed below and must be interpreted separately.
18. **Final classification:** `WORLD_MODEL_FRONTIER_SIGNAL_WITHOUT_CONTROLLER_GAIN`.

## Seed stability

| Scale | Seed | Direct SR | Point-WM SR | ΔSR |
|---|---:|---:|---:|---:|
| S6 | 0 | 90.00% | 88.00% | -2.00 pp |
| S6 | 1 | 88.00% | 90.00% | +2.00 pp |
| S6 | 2 | 90.00% | 90.00% | +0.00 pp |
| S15 | 0 | 90.00% | 90.00% | +0.00 pp |
| S15 | 1 | 90.00% | 88.00% | -2.00 pp |
| S15 | 2 | 90.00% | 86.00% | -4.00 pp |
| S30 | 0 | 90.00% | 80.00% | -10.00 pp |
| S30 | 1 | 90.00% | 88.00% | -2.00 pp |
| S30 | 2 | 90.00% | 76.00% | -14.00 pp |
| S50 | 0 | 100.00% | 98.00% | -2.00 pp |
| S50 | 1 | 90.00% | 86.00% | -4.00 pp |
| S50 | 2 | 100.00% | 88.00% | -12.00 pp |

## Controller and prediction metric curves

| Scale | Method | SR mean | Under-force | Mean force | Utility | Frontier MAE | NLL | Brier | H8 MAE | IE error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S6 | Direct | 89.33% | 10.00% | 4.0417 N | 0.0690 | 0.4417 N | 0.2483 | 0.0758 | NA | NA |
| S6 | Point-WM | 89.33% | 10.00% | 4.0667 N | 0.0540 | 0.5833 N | 0.8013 | 0.1435 | 0.061150 | 0.034015 |
| S15 | Direct | 90.00% | 10.00% | 4.1417 N | 0.0567 | 0.5417 N | 0.2470 | 0.0800 | NA | NA |
| S15 | Point-WM | 88.00% | 10.00% | 4.1167 N | 0.0387 | 0.5167 N | 0.6987 | 0.1170 | 0.056411 | 0.032916 |
| S30 | Direct | 90.00% | 10.00% | 4.1250 N | 0.0600 | 0.5250 N | 0.2160 | 0.0689 | NA | NA |
| S30 | Point-WM | 81.33% | 16.67% | 3.9083 N | -0.0063 | 0.3417 N | 0.3331 | 0.0650 | 0.049608 | 0.032749 |
| S50 | Direct | 96.67% | 3.33% | 4.1583 N | 0.1300 | 0.5250 N | 0.2038 | 0.0685 | NA | NA |
| S50 | Point-WM | 90.67% | 6.67% | 3.8500 N | 0.1177 | 0.2333 N | 0.3949 | 0.0680 | 0.050426 | 0.032057 |

## Per-root and mechanism evidence

`TASK0_POINTWM_PER_ROOT_SCALING.csv` contains all 10 roots for every scale and each single seed plus the prediction-level ensemble. `TASK0_POINTWM_MECHANISM_BY_SCALE.csv` counts Rescue, Economize, Collateral Damage, Over-force, and unchanged/other episodes over the exact 50 paired repeat-level evaluations.

## Old auxiliary signal versus current Point-WM controller

| Roots | Old Direct frontier MAE | Old Visual Joint | Current Direct | Current Point-WM |
|---:|---:|---:|---:|---:|
| 6 | 0.297 | 0.277 | 0.442 | 0.583 |
| 15 | 0.240 | 0.207 | 0.542 | 0.517 |
| 30 | 0.307 | 0.245 | 0.525 | 0.342 |
| 50 | 0.312 | 0.253 | 0.525 | 0.233 |

The old frontier advantage and current controller result answer different questions. Old Joint asked whether auxiliary visual/physics supervision improved an estimated predicted boundary. Current Point-WM asks whether a PhysicsOnly H8 model, consumed through root-OOF residual expected utility, improves archived-force SR, under-force, utility, and just-enough force selection. The old predicted-threshold frontier MAE and current selected-force MAE to empirical F*0.8 are not numerically interchangeable; only within-protocol advantage direction should be compared.

## Methodology and frozen semantics

Direct and Point-WM receive identical x, μ_GT, and archived candidate force. Point-WM alone learns the frozen H8×13 physical consequence with smooth-L1 trajectory loss plus λIE=1 adjacent-force supervision; no outcome BCE enters WM. Residual inputs are grouped-root OOF base predictions only. Final Direct and WM train on all S_N TRAIN roots and infer once on the frozen TEST. The controller searches only the nine archived TEST forces and maximizes the frozen expected-utility score.

Point-WM NLL/Brier/AUROC use the utility-implied probability `(score+1)/(R_success+1)` as a diagnostic only. The actual Point-WM controller uses the unmodified Direct utility plus residual score. This conversion does not tune or change selection.

## Secondary existing-trace Probe diagnostic

Single-seed mean Probe-PointWM SR is S6 93.33%; S15 84.67%; S30 84.67%; S50 84.00%; prediction-level ensemble SR is S6 100.00%; S15 88.00%; S30 88.00%; S50 88.00%. This secondary curve also does not show a late-emerging controller benefit.

## Limitations and robustness

- Only four scale points and ten TEST roots are available; trend correlations and log slopes are descriptive, not significance claims.
- Five repeats within a root-force cell and friction-conditioned contexts within the first six root families are not independent roots.
- Empirical F*0.8 is defined only when an archived force has >=4/5 successes. Unsupported roots remain in SR/utility but not frontier/under-force denominators.
- Standard deviations are population standard deviations across three canonical seed metrics, not confidence intervals.
- TEST was previously evaluated; no tuning or selection is performed here, but the result is a frozen diagnostic rather than a fresh confirmatory test.
- Secondary existing-trace Probe-PointWM was run and is reported separately in `TASK0_PROBE_POINTWM_DATA_SCALING_SECONDARY.csv`; it does not enter the primary classification.

## Recommended next step

Treat the selected classification as a diagnosis of the frozen pipeline. Do not tune on this TEST. A future confirmatory claim would require a newly preregistered, independently collected TEST population after any architecture or decision-interface change is frozen.

## Further questions

If controller gain does not track H8/IE improvement, the next question is whether the fixed summary52 residual discards decision-relevant trajectory structure. That question must be studied on TRAIN/DEV or a newly preregistered evaluation set, not by adapting to this frozen TEST.
