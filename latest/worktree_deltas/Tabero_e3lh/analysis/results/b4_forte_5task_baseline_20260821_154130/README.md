# B4 FORTE-Inspired Reactive Baseline

Status: `B4_POST_SLIP_REACTION_TOO_LATE_MULTI_TASK`.

`METHOD_CHANGE = NONE`. This is not an official FORTE reproduction. B4 tests the FORTE-style principle `start low -> detect GT slip/instability -> increase calibrated Tabero force target`.

## Technical Summary

- Aggregate FORTE-GT full SR: 0.783; Fixed Robust full SR: 0.993; SR ratio: 0.789.
- Aggregate FORTE-GT mean force: 3.039 N; Fixed Robust mean force: 3.511 N; GT-MinForce mean force: 2.597 N.
- Reactive strong tasks: []; partial tasks: [0, 5]; too-late tasks: [1, 2, 6].
- FORTE-GT was newly run in B4 at N=20 per task/friction cell; Fixed Low, Fixed Robust, and GT-MinForce rows combine B4 fixed-force supplement rows with frozen canonical B2-R2 references where protocol matched.
- QA note: task2 measured-force telemetry has near-zero rows in frozen references and B4 mid/high conditions; task2 force behavior should be read from target/update telemetry and SR.
- QA note: `SWEEP_ERROR_task0_dynamic.json` is the initial sandbox no-CUDA attempt; final dynamic audit files were produced by the later GPU run.

## Main Results

| task | object | low-mu SR | mid-mu SR | high-mu SR | FORTE mean F | robust SR | robust mean F | GT-MinForce mean F | classification |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | alphabet soup | 1.00 | 1.00 | 1.00 | 5.66 | 1.00 | 4.47 | 3.51 | `REACTIVE_PARTIAL` |
| 1 | cream cheese | 0.00 | 0.00 | 1.00 | 1.33 | 0.97 | 4.72 | 2.94 | `REACTIVE_TOO_LATE` |
| 2 | salad dressing | 0.10 | 1.00 | 1.00 | 0.76 | 1.00 | 0.71 | 0.71 | `REACTIVE_TOO_LATE` |
| 5 | tomato sauce | 1.00 | 1.00 | 1.00 | 4.74 | 1.00 | 4.60 | 3.38 | `REACTIVE_PARTIAL` |
| 6 | butter | 0.65 | 1.00 | 1.00 | 2.70 | 1.00 | 3.04 | 2.23 | `REACTIVE_TOO_LATE` |

## Low-Friction Rescue

| task | start F | needed F | mean updates | rescue SR | mean time to sufficient F |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 5 | 4.00 | 1.00 | 9.96 |
| 1 | 3 | 6 | 4.00 | 0.00 |  |
| 2 | 3 | 8 | 4.00 | 0.10 | 11.01 |
| 5 | 3 | 5 | 4.00 | 1.00 | 9.97 |
| 6 | 3 | 4 | 4.00 | 0.65 | 10.46 |

## Interpretation

The B4 classification is task-level, not just aggregate. A task is `REACTIVE_STRONG` only when FORTE-GT stays near Fixed Robust success while materially lowering force. A task is `REACTIVE_TOO_LATE` when low-friction rescue remains far below Fixed Robust even with privileged GT slip.

Task2 salad dressing is the stress test because the low-friction condition needs 8 N while mid/high friction need only 3 N. The reactive ladder has to climb 3->4->5->6->8 after instability; the task2 low-mu row in `FORTE_MAIN_RESULTS.csv` records whether that happens early enough.

## Files

- `FORTE_MAIN_RESULTS.csv` / `FORTE_MAIN_RESULTS.md`
- `TASK_LEVEL_CLASSIFICATION.csv`
- `FINAL_VERDICT.json`
- `TASK{0,1,2,5,6}_FORTE.csv`
- `DYNAMIC_FORCE_SERVO_AUDIT.csv`
- `plots/`

## Next Step

Freeze B4, run B5 Tabero Neutral, then freeze the baseline landscape before implementing final OURS.
