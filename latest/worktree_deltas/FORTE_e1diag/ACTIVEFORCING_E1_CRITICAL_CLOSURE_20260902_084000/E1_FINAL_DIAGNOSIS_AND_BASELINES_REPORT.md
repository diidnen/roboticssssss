# Task6 E1 closure identifies Direct low-force overconfidence; faithful FORTE/Tabero numbers remain blocked

## Decision

The task6 ActiveForcing deficit is localized to low-boundary selection: all five Active failures select the lowest candidate, all three Query-success -> Active-failure pairs remain failures when GT friction replaces P4-B, and the next archived candidate succeeds. Primary attribution is `DIRECT`; the Expected-Utility force cost is secondary because it converts the overconfident probability into a low-force choice. The physical identifier is not primary.

No pi0 repair was performed. E1 is scripted downstream rather than native-pi0 E2E, so it cannot establish native nominal capability; more importantly, it provides no affirmative evidence that pi0 is the primary failure source required to authorize few-demo repair.

## Authoritative method/config

- Frozen authoritative pi0 -> P4-B query -> friction belief -> shared Direct full-task feasibility -> Expected Utility -> selected force as controller setpoint -> unchanged controller -> pi0 nominal continuation.
- Utility: `p(success)*(1-F/Fmax) + (1-p(success))*(-1)`, low-force tie break.
- Fmax: task0=5, task1=6, task5=5, task6=4 N.
- Config hash: `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`; exact expected-hash match: `True`.
- No hard-rho selector was restored. No TEST data was read or used to tune Fmax or Utility.

## Quantified task6 result

- ActiveForcing: 86.1111% SR, selected 3.2237 N, realized Utility 0.0214.
- Query-Ignored: 94.4444% SR, selected 3.2981 N, realized Utility 0.1065.
- Fixed-Max: 100.0000% SR, selected 3.9022 N, realized Utility 0.0245.
- GT-Friction Utility: 86.1111% SR, selected 3.1902 N, realized Utility 0.0298.
- Active task6 friction MAE/bias are recorded in `E1_TASKWISE_FAILURE_AUDIT.csv`; GT friction produces the same task6 SR and the same paired failing choices.

## External baselines

FORTE's real method is analog-tactile SVR + spectral slip detection + reactive Dynamixel impedance closure. The E1 archive lacks its sensor/actuator semantics, so prior oracle-slip and fixed/ladder surrogates are excluded.

Tabero's real method is the native 13-D Field+FS VTLA closed loop. A faithful executor and frozen checkpoint are recoverable, but no exact E1-tuple rollout exists and protected GPU workloads prevented a safe new Isaac run. Unpaired smoke rows are excluded.

## Data-quality and claim boundary

QA passed for the frozen candidate reconstruction: 720 unique branches, 144 paired episodes, five candidates each, with GT Direct maximum absolute reproduction error 2.97e-08. Selected/commanded force remains separate from measured force. Tracking MAE in failed branches is not treated as causal because post-loss zeros contaminate it. The evidence is TRAIN/root-heldout-OOF development, not sealed TEST.

## Final status

- `TASK6_FAILURE_DIAGNOSIS_COMPLETE`
- `PI0_REPAIR = NOT_NEEDED`
- `FORTE = FAITHFUL_REPRODUCTION_BLOCKED`
- `TABERO = FAITHFUL_REPRODUCTION_BLOCKED`

The diagnosis is ready to share with the stated caveats. The external-baseline table is structurally complete but scientifically incomplete until faithful paired rollouts exist.
