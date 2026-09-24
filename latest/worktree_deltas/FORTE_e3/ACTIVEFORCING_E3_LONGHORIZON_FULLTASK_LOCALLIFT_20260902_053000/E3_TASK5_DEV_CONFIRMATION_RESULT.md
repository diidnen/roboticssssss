# E3 task5 fixed DEV confirmation result

## Verdict

`TASK5_DEV_FAIL_NO_ROBUST_REGIME_TRANSFER`

The long-horizon `libero_10/task5` TRAIN support gate passed on root7400, but its three fixed force anchors did not transfer to held-out DEV roots. The preregistered plan contains six cells—three anchors on each of roots 7500 and 7501—and all six completed with valid query states and hashes. Root7500 produced `LOCAL_FAILURE` at 4, 6, and 7 N; root7501 produced two `LOCAL_FAILURE` cells and one `LOCAL_SUCCESS_DOWNSTREAM_FAILURE`, but no `FULL_TASK_SUCCESS`. Thus the frozen gate fails on all six confirmatory cells.

## Case evidence

| Root | Force | Query | Root hash | Pick | Lift | FullTask | Regime |
|---:|---:|---:|---|---:|---:|---:|---|
| 7500 | 4 N | 1 | `a7e8dec7...f51a9` | 1 | 0 | 0 | LOCAL_FAILURE |
| 7500 | 6 N | 1 | `a7e8dec7...f51a9` | 1 | 0 | 0 | LOCAL_FAILURE |
| 7500 | 7 N | 1 | `a7e8dec7...f51a9` | 1 | 0 | 0 | LOCAL_FAILURE |
| 7501 | 4 N | 1 | `220bd2ff...0f2b65` | 1 | 0 | 0 | LOCAL_FAILURE |
| 7501 | 6 N | 1 | `220bd2ff...0f2b65` | 1 | 0 | 0 | LOCAL_FAILURE |
| 7501 | 7 N | 1 | `220bd2ff...0f2b65` | 1 | 1 | 0 | LOCAL_SUCCESS_DOWNSTREAM_FAILURE |

The analyzer validates all six preregistered rows with no schema/telemetry errors: stable same-root hashes, distinct hashes between roots, and all six query states valid. No cell is excluded from the confirmatory table. Root7500 alone is sufficient to show gate failure, while root7501 adds the paired LocalLift-positive/FullTask-negative case without producing a FullTask positive.

## Mechanism

At root7500/F7 the predicted force slot remained 7 N, but measured mean/peak contact force was 0/0 N because the trajectory never reached a lift-capable grasp. The same pattern holds at F4/F6. Thus the DEV failure is primarily frozen-pi0 nominal grasp/lift robustness across reset roots, not evidence that Expected Utility selected too little force, not evidence for a force-range boundary, and not a controller-tracking claim during established contact.

The one TRAIN FullTask success at 7 N remains valid label-support evidence, but it is not robust enough to support a held-out FullTask-vs-LocalLift comparison. Descriptively, the six available DEV rows contain one LocalLift-positive/FullTask-negative branch and zero FullTask positives. Thus the FullTask target is still degenerate, and no classifier-generalization claim is scientifically supported.

## Scientific scope and next action

- No adaptive force insertion, Fmax inference, Utility evaluation, hard-rho selector, TEST outcome, or checkpoint change occurred.
- The authoritative Utility hash remains `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10` with Expected Utility selection.
- Task5 few-demo onboarding is not started: no valid local long-horizon demos were found in the prior audit, and the frozen reserve-task order is the next authorized path.
- Next candidate: nominal-only robust 8 N qualification of `libero_10/task2`, per the already frozen reserve plan, after a fresh resource gate. This qualification makes no Utility/Fmax claim.

Machine-readable incremental gate and cell table are in the pilot root as `E3_TASK5_DEV_CONFIRMATION_GATE.json` and `E3_TASK5_DEV_FIXED_ANCHOR_TABLE.csv`.
