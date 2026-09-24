# E3 task5 onboarding replay acceptance status

Status: `FIVE_OF_FIVE_TRAIN_REPLAY_GATES_PASS`.

The frozen task5 TRAIN onboarding set is IDs `[1,2,11,12,19]`. Each ID has an independently accepted real-tactile 13D `7dpf` replay gate:

- ID1: `ID1_REAL_TACTILE_7DPF_SMOKE_PASS`
- ID2: `DEMO2_REAL_TACTILE_7DPF_GATE_PASS`
- ID11: `DEMO11_REAL_TACTILE_7DPF_GATE_PASS`
- ID12: `DEMO12_REAL_TACTILE_7DPF_GATE_PASS`
- ID19: `DEMO19_REAL_TACTILE_7DPF_GATE_PASS`

Demo19's accepted HDF5 SHA-256 is `ca36d95a379be437843424ef7c8a2d5d9638d6733e1761076fb1fa9f25d19b85`; its independent gate is `TASK5_PI0_ONBOARDING_DEMO19_GATE.json` and its audit is `TASK5_PI0_ONBOARDING_DEMO19_RESULT.md`.

This status artifact is limited to the five replay-acceptance gates. It does not represent or attribute any downstream pipeline execution state. No TEST outcome, ActiveForcing outcome, Utility/Fmax target, synthetic tactile, or force-selector supervision was used for these acceptance decisions; the authoritative Utility is unchanged.
