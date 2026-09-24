# Mass offline-to-E2E gap report

This comparison uses only the corrected fresh E2E rows; the old first-candidate runner is excluded.

| Arm | Offline formal TEST SR | Corrected fresh E2E SR | Gap |
|---|---:|---:|---:|
| Fixed-Max | 0.667 | 0.167 | -0.500 |
| NoQuery-Prior | 0.667 | 0.000 | -0.667 |
| ActiveForcing-Mass | 0.917 | 0.667 | -0.250 |
| GT-Mass | 0.833 | 0.333 | -0.500 |

## Gap decomposition

- **Upstream/query reach:** corrected Active and GT query reach/valid are 6/6, so the corrected gap is not caused by query invocation failure. Native π0 and Fixed-Max remain weak/failed, showing that the frozen execution path is not uniformly reliable from these fresh roots.
- **State/identifier shift:** fresh mass estimates are +0.3277--+0.3878 kg above GT in all six contexts, versus offline identifier MAE 0.0229 kg. This changes force decisions and is a real secondary distribution-shift/calibration issue.
- **Decision gap:** Active and GT select different forces in all six contexts. The Active-vs-GT success delta is concentrated in two MID contexts: 4.0 N succeeds while 1.5 N fails at transport. The frozen GT score ranks 1.5 N highest for MID despite 2.5 N having nearly the same predicted probability; this is a selector/utility boundary effect.
- **Execution/downstream gap:** both HIGH contexts fail for Active 4.0 N, GT 2.5 N, and Fixed-Max 4.0 N, with transport/placement failures. The Direct model predicts high success probabilities for these forces, so offline-to-fresh state/execution mismatch or Direct calibration under fresh states remains after fixing scoring.

## Final interpretation

The 0.917-to-0.667 Active gap is mixed rather than a single failure. The corrected Active-vs-GT difference is explained by force selection in 2/6 contexts; the remaining Active 2/6 failures are HIGH downstream frozen-π0/state cases not rescued by the observed candidate support. The data do not justify claiming that every possible force would fail, because no new 3--8 N sweep was run under the no-requery constraint.
