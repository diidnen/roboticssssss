# Mid vs high remainder divergence — 2026-09-16

Not paper data. Seed 200014 episode 1. Official deskbin collider, deskbin μ=0.30. Finger mid 0.575 / high 0.85. Same PRE → query 4 N / 12 mm → realize F → lift z=0.08, then **one T6 snapshot per F**, then mid/high remainder fork only.

## 1. Forces tested

| F (N) | Why | T6 squeeze (mid-setup) | Mid R7 | High R7 |
| --- | --- | --- | --- | --- |
| 1.0 | old typical mid retain / high fail | 2.09 N, bilateral | drop | drop |
| 2.5 | old typical mid retain / high fail | 4.63 N, bilateral | drop | **retain** |
| 5.0 | old both retain | 8.69 N, bilateral | retain | retain |

R0 after restore+2 contact-refresh steps: pose error vs packed T6 < 0.12 mm. Hull assignment can differ; not treated as obvious.

## 2. Remainder stage tables

R0–R1: not obvious (pose < 0.25 mm, both bilateral). R2 is a **shared** wrist-rotation contact valley (both unilateral, squeeze 0). Dense traces show the first mode loss at ~6° wrist (~0.53 s), not uniquely at the 25% checkpoint.

### F = 1.0 N

| Stage | Mid sq / mode | High sq / mode | Pose Δ | Slip Δ | Obvious |
| --- | --- | --- | --- | --- | --- |
| R0 T6 | 2.10 / bi | 2.01 / bi | 0.03 mm | 0 | no (hull only) |
| R1 wrist start | 2.30 / bi | 2.82 / bi | 0.07 mm | 0 | no (hull only) |
| R2 wrist 25% | 0 / uni, drop | 0 / uni, drop | 7.79 mm | 3.09 mm | yes, **both dropped** |
| R3 wrist 50% | 0 / none | 0 / none | 74 mm | 16 mm | yes, both falling |
| R5 wrist max | 0 / none | 0 / none | 129 mm | 50 mm | yes |
| R7 pour end | 0 / none | 0 / none | 129 mm | 51 mm | yes |

### F = 2.5 N

| Stage | Mid sq / mode | High sq / mode | Pose Δ | Obvious |
| --- | --- | --- | --- | --- |
| R0 T6 | 3.18 / bi | 3.72 / bi | 0.03 mm | no (hull only) |
| R1 wrist start | 4.69 / bi | 6.21 / bi | 0.25 mm | no |
| R2 wrist 25% | 0 / uni, drop | 0 / uni, drop | 3.43 mm | yes, **shared valley** |
| R3 wrist 50% | 0 / none | **9.39 / bi** | 37 mm | yes, **high recovered** |
| R5 wrist max | 0 / none | 4.60 / bi | 446 mm | yes, mid gone |
| R7 pour end | 0 / none | 4.64 / bi | 436 mm | yes |

### F = 5.0 N

| Stage | Mid sq / mode | High sq / mode | Pose Δ | Obvious |
| --- | --- | --- | --- | --- |
| R0 T6 | 8.67 / bi | 7.01 / bi | 0.09 mm | no (hull only) |
| R1 wrist start | 8.73 / bi | 8.65 / bi | 0.23 mm | no |
| R2 wrist 25% | 0 / uni | 0 / uni | 8.62 mm | yes, shared valley |
| R3 wrist 50% | 8.72 / bi | 7.66 / bi | 11.9 mm | yes, **both recovered** |
| R5 wrist max | 8.67 / bi | 8.57 / bi | 8.86 mm | pose/slip only |
| R7 pour end | 8.77 / bi | 8.66 / bi | 9.37 mm | pose/slip only, both hold |

## 3. First persistent fork

Wrist reorientation, after R1. Dense: both lose bilateral contact at ~6° (~0.53–0.56 s). Checkpoint R2 (≈29°) samples that valley.

That fork is **shared**, not high-worse.

## 4. Last still-same stage

**R1 wrist start** at every F (pose ≤ 0.25 mm, both bilateral, squeeze within 2 N).

## 5. First concrete difference after the fork

Not a high-only mode collapse.

- Squeeze: both → 0 at the valley. At F=2.5 R3, high squeeze 9.39 N vs mid 0.
- Contact mode: both unilateral at R2. At F=2.5 R3, high bilateral vs mid none.
- Hull: both go to a single `-y` lip (hull 15; high F=2.5 uses 20). Not a high-only hull jump that ejects the object.
- Slip / pose: grow after the shared valley because one or both objects are no longer pinched; at F=5 both hold with ~9 mm pose offset.
- Wrist angle: matched (same scripted first pour pose, ~117.6° at R5).

F=2.5 0.54–1.28 s: high briefly unilateral ~16 ms earlier, then **high regrasps at ~59°**; mid never does.

## 6. High-friction sticking / bad contact transition?

No high-worse sticking signature.

Wrist rotation does force a contact-mode transition (opposing `±x` hull 14 → one-finger `-y` lip) for **both** μ. After that:

- F=1.0: neither regrasps.
- F=2.5: **high regrasps, mid does not.**
- F=5.0: both regrasp; squeeze ~8.7 N vs 8.7 N at R7.

High is locally more able to recover a pinch under torque, consistent with the friction unit test, not with “stick then pop out.”

Optional pure-rotation vs pure-tangential control was **not** run: high was never the worse persistent branch.

## 7. Does this explain old 17/20 vs 4/20?

**No.** Old ranking is mid retain / high fail. Same T6 remainder produces:

- 1.0: both fail
- 2.5: high succeeds, mid fails (inverted)
- 5.0: both succeed

So 17/20 vs 4/20 is not a single remainder mechanism from one T6. It is more likely small **upstream rollout differences** (different T6s across the 20 F points / μ-specific histories), not “high μ ruins wrist/pour from the same lift-end state.”

## 8. Classification

**NO_CLEAR_DIVERGENCE**

Wrist is where contact first flickers, but that event is shared. High does not become the worse grasp. Pour (R6–R7) does not add a new high-worse mechanism after a held pinch.

Time-series plots: `plots/F_1.00_timeseries.png`, `F_2.50_timeseries.png`, `F_5.00_timeseries.png`.
