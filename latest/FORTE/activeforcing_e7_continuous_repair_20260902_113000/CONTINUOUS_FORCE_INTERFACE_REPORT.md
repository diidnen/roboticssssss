# Continuous Force Interface Calibration

## Verdict

Historical telemetry is sufficient for a TRAIN-only interface audit. The old complete-trajectory metric is invalid because it includes post-drop/no-contact samples. This repair uses attached bilateral-contact samples from the latter half of lift and transit only. Branch-hold controller overshoot and all post-drop/no-contact samples are excluded.

## Certified ranges

| task | safe_min_N | safe_max_N | effective_resolution_N |
|---|---|---|---|
| 0.0000 | 3.4000 | 5.0000 | 0.1600 |
| 1.0000 | 4.8000 | 6.0000 | 0.1300 |
| 5.0000 | 3.4000 | 5.0000 | 0.0700 |
| 6.0000 | 3.0000 | 4.0000 | 0.0400 |

Certification requires a contiguous upper stratum suffix with coverage >=0.80, p90 trajectory MAE <=0.50 N, absolute median bias <=0.35 N, and saturation rate <=0.10. Effective resolution is the 90th-percentile repeat-pair steady-force disagreement divided by the robust Theil-Sen command-to-measured slope, rounded up to 0.01 N.

## Phase rule and evidence

- PRE-CONTACT/TRANSIENT: branch_hold and non-steady lift; reported but excluded from calibration.
- CONTACT/LOAD-BEARING: latter-half lift or transit, bilateral contact, object-command attachment deviation <=0.06 m, measured force >=0.15 N.
- POST-DROP/NO-CONTACT: every remaining sample; excluded from calibration.
- Trajectories: 720; native commands: 360; load-bearing coverage: 0.9542.
- Median load-bearing MAE: 0.1922 N; median bias: -0.1243 N.

The required calibration CSV has one row per native command and retains both repeat telemetry paths and SHA-256 hashes. The 720-row trajectory-level sidecar preserves repeat diagnostics. No TEST data were read.
