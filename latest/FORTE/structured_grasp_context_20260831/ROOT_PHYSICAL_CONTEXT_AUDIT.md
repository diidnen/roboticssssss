# Root Physical Context Audit

## Direct answer

The dataset contains **18 captured pre-probe physical contexts per task, grouped into 6 root-seed families** (three friction-conditioned executions per root seed). The correct independent-count statement is therefore not simply 18 IID roots: the held-root CV holds out two entire root-seed families (six physical contexts) per fold.

For task0 root06 (F*0.8=4.25 N) versus root07 (3.75 N), recorded relative grasp geometry is effectively the same. The horizontal eccentricity differs by only 0.016 mm, vertical offset by 0.003 mm, gripper opening by 0.016 mm, and friction by 0.0065. The frozen first H8 remains branch_hold for both and has essentially zero displacement. The largest obvious recorded difference is a common scene translation (EEF/object y changes together by about 1.484 mm), not a different object-in-gripper offset.

This means the root06/root07 0.5 N frontier difference is **not explained by a material recorded grasp offset, orientation change, gripper opening change, or H8 demand change**. It is consistent with unlogged contact microstate / outcome stochasticity, or with a relevant variable outside the current audit interface.

## Captured population

| task | split | contexts | root_seed_groups | valid_frontiers |
| --- | --- | --- | --- | --- |
| 0 | DEV | 2 | 2 | 2 |
| 0 | TRAIN | 18 | 6 | 18 |
| 1 | DEV | 4 | 2 | 3 |
| 1 | TRAIN | 18 | 6 | 17 |
| 5 | TRAIN | 18 | 6 | 18 |

- task5 planned DEV roots were not physically collected, so task5 mechanism evaluation uses the existing retrospective TRAIN root-heldout CV only.
- task1 DEV collection is incomplete and remains previously seen; it is audited descriptively but is not used to tune the feature set.
- task1 TRAIN has the existing reconstructed-label caveat; task5 supplies direct-label replication.

## What roots actually vary in

Across captured contexts the logged variables that vary are absolute object/EEF placement, small residual object-to-EEF offsets, object quaternion, finger state/opening, friction, and raw visual features. Relative rotation cannot be reconstructed because EEF quaternion was not persisted.

The maximum TRAIN first-H8 command path length is 4.314e-07 m; all first-H8 rows are branch_hold. Therefore H8 motion is audited but excluded from x_grasp before model fitting.

## Deployment legality

The 11-D Structured vector is an oracle mechanism diagnostic: six dimensions depend on simulator GT object pose. EEF position, gripper opening, and finger-joint asymmetry are legal/derived from legal robot state. No deployable claim is made from the oracle vector.

## Integrity controls

- Frozen structured feature spec SHA256: `906c66e27d5775473149c3706f2a3c41dcc0593a5cc28381572290a91a92ae92`.
- One row per captured context; all rows link strict step-190 probe telemetry, a restorable snapshot, and a frozen visual feature.
- F*0.8 is a label-only field and never enters x_grasp.
- Correlations are descriptive and are not used for feature selection.
- Existing held-out roots are mechanism-analysis data, not a new untouched TEST.
