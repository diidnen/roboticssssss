# Residual Data Alignment Audit

**BRANCH_ALIGNMENT = PASS**

All 720 residual rows were reconstructed from the authoritative pooled TRAIN telemetry population and the frozen OOF shards. The merge grain is one unique branch; root family, context, force cell, and repeat remain grouped inside one fold.

## Cardinality and grouping

- Rows / unique branch IDs: 720 / 720
- Duplicate branch IDs / duplicate `(context, repeat, force)` rows: 0 / 0
- Maximum folds per root family / context: 1 / 1
- Authoritative exact-field mismatches: 0
- Maximum force / friction reconstruction error: 8.88e-16 / 1.11e-16

## Twenty-row independent reconstruction

The deterministic 20-row sample had maximum errors p_D=2.97e-08, Current summary=0, PhysicsOnly summary=0.

| index | task | fold | branch | p_D error | Current error | PhysicsOnly error |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 0 | 0 | `pv_train_t0_r00_s5100_high_mu0.940189_TRAIN_S0_F3.33568986_R1_F3.33569` | 3.5e-09 | 0 | 0 |
| 44 | 0 | 1 | `pv_train_t0_r01_s5101_low_mu0.255969_TRAIN_S2_F4.14891842_R1_F4.14892` | 5.13e-09 | 0 | 0 |
| 89 | 0 | 2 | `pv_train_t0_r02_s5102_mid_mu0.571550_TRAIN_S4_F4.68620483_R2_F4.6862` | 1.65e-08 | 0 | 0 |
| 134 | 0 | 1 | `pv_train_t0_r04_s5104_low_mu0.298968_TRAIN_S2_F4.00310916_R1_F4.00311` | 5.05e-11 | 0 | 0 |
| 179 | 0 | 2 | `pv_train_t0_r05_s5105_mid_mu0.529333_TRAIN_S4_F4.96061737_R2_F4.96062` | 9.93e-09 | 0 | 0 |
| 180 | 1 | 0 | `pv_train_t1_r00_s5100_high_mu0.921500_TRAIN_S0_F4.03855904_R1` | 1.84e-09 | 0 | 0 |
| 224 | 1 | 1 | `pv_train_t1_r01_s5101_low_mu0.271996_TRAIN_S2_F5.05313634_R1` | 1.36e-11 | 0 | 0 |
| 269 | 1 | 2 | `pv_train_t1_r02_s5102_mid_mu0.509876_TRAIN_S4_F5.60687975_R2` | 3e-09 | 0 | 0 |
| 314 | 1 | 1 | `pv_train_t1_r04_s5104_low_mu0.223399_TRAIN_S2_F5.01620862_R1` | 2.77e-13 | 0 | 0 |
| 359 | 1 | 2 | `pv_train_t1_r05_s5105_mid_mu0.533126_TRAIN_S4_F5.75932344_R2` | 1.72e-08 | 0 | 0 |
| 360 | 5 | 0 | `pv_train_t5_r00_s5100_high_mu0.945004_TRAIN_S0_F3.37981224_R1` | 2.26e-08 | 0 | 0 |
| 404 | 5 | 1 | `pv_train_t5_r01_s5101_low_mu0.269928_TRAIN_S2_F4.02711881_R1` | 3.41e-09 | 0 | 0 |
| 449 | 5 | 2 | `pv_train_t5_r02_s5102_mid_mu0.569222_TRAIN_S4_F4.69379467_R2` | 2.97e-08 | 0 | 0 |
| 494 | 5 | 1 | `pv_train_t5_r04_s5104_low_mu0.296413_TRAIN_S2_F3.88303785_R1` | 1.31e-10 | 0 | 0 |
| 539 | 5 | 2 | `pv_train_t5_r05_s5105_mid_mu0.478580_TRAIN_S4_F4.91759859_R2` | 1.38e-08 | 0 | 0 |
| 540 | 6 | 0 | `pv_train_t6_r00_s5100_high_mu0.989348_TRAIN_S0_F3.12589117_R1` | 9.34e-09 | 0 | 0 |
| 584 | 6 | 1 | `pv_train_t6_r01_s5101_low_mu0.216789_TRAIN_S2_F3.55429544_R1` | 1.54e-08 | 0 | 0 |
| 629 | 6 | 2 | `pv_train_t6_r02_s5102_mid_mu0.568022_TRAIN_S4_F3.96664047_R2` | 1.21e-08 | 0 | 0 |
| 674 | 6 | 1 | `pv_train_t6_r04_s5104_low_mu0.226087_TRAIN_S2_F3.46632795_R1` | 1.51e-10 | 0 | 0 |
| 719 | 6 | 2 | `pv_train_t6_r05_s5105_mid_mu0.481964_TRAIN_S4_F3.90759393_R2` | 9.68e-09 | 0 | 0 |

## Task1 label caveat

Task1 has 40 direct labels and 140 reconstructed labels. On the 40 directly comparable rows, the frozen audit reports 0 mismatches. This is a data caveat, not a newly discovered merge error.

## Checkpoints and second-level cross-fitting

Current objective: `['Lphysics + LIE + lambda_feas*BCE']`. PhysicsOnly objective: `['Lphysics + LIE']` with outcome gradient disabled. All base fold/seed identities and TEST-used flags passed. All 27 residual checkpoints use `R_real-U_D`, MSE, positive finite TRAIN-only normalization, the other two folds as TRAIN, and the named fold as held-out; no scored fold enters its residual checkpoint training.
