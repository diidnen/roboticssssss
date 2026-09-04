# P5-S0-C Final Report

STATUS:
PASS
METHOD_CHANGE: NONE
PROTOCOL_CHANGE:
PAIRED_HIDDEN_PHYSICS_BOUNDARY_DATA_AND_PROBE_VALUE_ADJUDICATION
ARTIFACTS:
/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542

DATA:
- Tasks: [0, 1, 5, 6]
- Task2 used: False
- Root groups: 48
- Train/dev/test roots: {'DEV': 8, 'TEST': 16, 'TRAIN': 24}
- Physical contexts: 144
- Full-task branches: 576
- Probe-qualified contexts: 144
- State parity: 576/576

HIDDEN PHYSICS:
- LOW range: [0.20, 0.30]
- MID range: [0.45, 0.60]
- HIGH range: [0.90, 1.00]
- Friction used as model input: False

PRIMARY AMBIGUITY FORCES:
- task0: 4.0 N
- task1: 5.0 N
- task5: 4.0 N
- task6: 3.0 N

BOUNDARY DIVERSITY:
- Decision-discordant TEST pairs: 14
- task0: 4
- task1: 4
- task5: 4
- task6: 2
- Boundary test adjudicative: True

MODELS:
- Task+F Threshold: 5 seeds
- Static-State Threshold: 5 seeds
- Probe-GRU Threshold: 5 seeds
- Q2F-Threshold: 5 seeds
- Q2F-GNP: 5 seeds

PAIRED PROBE VALUE:
- Best pairwise ranking accuracy: PROBE_GRU_THRESHOLD=0.9286
- Best paired NLL: STATIC_STATE_THRESHOLD=0.2597
- Best paired Brier: STATIC_STATE_THRESHOLD=0.0816
- Best false-sufficient rate: TASK_FORCE_THRESHOLD=0.0000
- Probe model beats Task+F: see P5S0C_PAIRED_RESULTS.csv
- Probe model beats Static-State: see P5S0C_PAIRED_RESULTS.csv

EVIDENCE PERTURBATION:
- LOW/HIGH swap degradation: see P5S0C_LOW_HIGH_SWAP_RESULTS.csv
- Within-task shuffle degradation: see P5S0C_PROBE_SHUFFLE_RESULTS.csv
- Mean-evidence replacement degradation: see P5S0C_MEAN_REPLACEMENT_RESULTS.csv
- Consistent across seeds: see classification reasons

OFFLINE FORCE SELECTION:
- Task+F success / mean force / under-force: 0.979 / 4.675 N / 0.021
- Static-State success / mean force / under-force: 0.925 / 4.050 N / 0.075
- Probe-GRU success / mean force / under-force: 0.975 / 4.213 N / 0.025
- Q2F-Threshold success / mean force / under-force: 0.988 / 4.498 N / 0.013
- Q2F-GNP success / mean force / under-force: 0.963 / 4.373 N / 0.037

PRIMARY_CLASSIFICATION:
P5S0C_PROBE_SIGNAL_PRESENT_BUT_NOT_CONVERTED_TO_DECISION

SCIENTIFIC_INTERPRETATION:
1. The boundary dataset is adjudicative: same-root LOW/HIGH TEST pairs include decision-discordant outcomes on every task.
2. Task+F is deliberately unable to distinguish those paired examples because task and requested force are identical inside each pair.
3. Static-state and dynamic-probe models isolate whether passive contact state or active probe response explains any gain.
4. Evidence perturbation tests are required for probe-value claims; aggregate NLL alone is not sufficient.
5. This remains offline model adjudication only; no fresh E2E rollout was run.

NEXT:
- If dynamic probe value confirmed: freeze the model and run fresh four-task E2E.
- If static state explains gain: revise the method claim and remove unsupported active-probe language.
- If probe signal is not converted: diagnose temporal representation before more E2E.
- If boundary test is not adjudicative: refine force/friction sampling without changing the model.
