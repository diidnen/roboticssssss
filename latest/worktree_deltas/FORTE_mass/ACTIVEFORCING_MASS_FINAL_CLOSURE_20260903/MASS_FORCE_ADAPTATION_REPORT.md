# Mass force adaptation report

Offline primary benchmark: formal TEST roots 8200 and 8201, paired across mass, candidate force, and repeat. Every learned policy is fit on formal TRAIN branches only. Direct is a new small Mass-only model with `z = mass` plus the candidate force and fixed polynomial interaction; it uses full-task success labels. The frozen Expected Utility is `U(F)=p_success*(8-F)+(1-p_success)*(-1)`, with lower force as the tie-break. Hindsight Grid Oracle is diagnostic only.

## Policy summary (mean over 3 seeds)

| Policy | n | Full-task SR | Mean selected F | Measured F | Peak F | Under-force | Excess F | Realized U | GT-force agreement | Utility decision agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed-Max | 12 | 0.667 | 4.000 | 0.001 | 39.750 | 0.000 | 2.833 | 2.333 | 0.000 | 0.000 |
| Fixed-Robust | 12 | 0.667 | 4.000 | 0.001 | 39.750 | 0.000 | 2.833 | 2.333 | 0.000 | 0.000 |
| True NoQuery-Prior | 12 | 0.667 | 1.500 | 0.000 | 31.659 | 0.000 | 0.333 | 4.000 | 0.667 | 0.000 |
| ActiveForcing-Mass | 12 | 0.917 | 1.750 | 0.000 | 30.809 | 0.000 | 0.583 | 5.646 | 0.833 | 0.500 |
| GT-Mass | 12 | 0.833 | 1.833 | 0.000 | 31.159 | 0.000 | 0.667 | 4.972 | 1.000 | 0.333 |
| Hindsight Grid Oracle | 12 | 1.000 | 1.500 | 0.000 | 30.325 | 0.000 | 0.333 | 6.500 | 0.333 | 1.000 |

## Observed mass × force curves

| Mass | Force | n | Full-task SR | Lift SR | Transport SR | Placement SR | Failure stage |
|---|---:|---:|---:|---:|---:|---:|---|
| LOW (0.05 kg) | 0.5 | 4 | 0.000 | 1.000 | 0.000 | 1.000 | turning_transport:4 |
| LOW (0.05 kg) | 1.0 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | none:4 |
| LOW (0.05 kg) | 1.5 | 4 | 0.750 | 1.000 | 0.750 | 1.000 | none:3;turning_transport:1 |
| LOW (0.05 kg) | 2.5 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | none:4 |
| LOW (0.05 kg) | 4.0 | 4 | 0.250 | 1.000 | 0.250 | 1.000 | none:1;turning_transport:3 |
| MID (0.10 kg) | 0.5 | 4 | 0.000 | 1.000 | 0.000 | 1.000 | turning_transport:4 |
| MID (0.10 kg) | 1.0 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | none:4 |
| MID (0.10 kg) | 1.5 | 4 | 0.750 | 1.000 | 0.750 | 1.000 | none:3;turning_transport:1 |
| MID (0.10 kg) | 2.5 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | none:4 |
| MID (0.10 kg) | 4.0 | 4 | 0.750 | 1.000 | 0.750 | 1.000 | none:3;turning_transport:1 |
| HIGH (0.20 kg) | 0.5 | 4 | 0.000 | 1.000 | 0.000 | 1.000 | turning_transport:4 |
| HIGH (0.20 kg) | 1.0 | 4 | 0.000 | 1.000 | 0.000 | 1.000 | turning_transport:4 |
| HIGH (0.20 kg) | 1.5 | 4 | 0.500 | 1.000 | 0.500 | 1.000 | none:2;turning_transport:2 |
| HIGH (0.20 kg) | 2.5 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | none:4 |
| HIGH (0.20 kg) | 4.0 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | none:4 |

## Force-sensitivity finding

The benchmark is force-sensitive: 0.5 N fails transport in all three bands, HIGH mass also fails at 1.0 N, and HIGH reaches 1.0 SR at 2.5 N and above. LOW and MID have the same aggregate curve here and show non-monotonic degradation at 4.0 N, so the mass effect is real but not a clean monotone separation in this task. The downstream full-task force curves and selected-force behavior are retained as the primary scientific test; the result should not be described as a universal monotone mass frontier.

## Matched scope

All policies use the same task (LIBERO object task 2), root, mass, initial state, query state, candidate force grid, and repeats. No TEST branch outcome is used by the identifier or Direct fit.
