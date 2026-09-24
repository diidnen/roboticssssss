# P2 — Actionable Pre-Slip Intervention Window

**Not a method.** Oracle timing only: `ORACLE_INTERVENTION_TIMING`. Not GelSight-triggered.

Question: given P1’s ~300 ms deployable marker unloading, is a **4 N → 6 N** switch in that window enough to prevent μ=0.20 failure?

## Setup

- Env / task: `Isaac-Libero-Franka-Hybrid-Tactile-v0`, libero_object 1, cream cheese → basket
- μ=0.20, initial **4 N**, intervention **6 N** (one shot; not 4→6→8)
- Matched P1 seeds; this-session fixed-4 N baseline T_slip (P1 8 mm rule)
- Scripted pick-place (T1 freeze `d_pred` in transit). Cameras off (oracle, not tactile trigger)
- Tabero `3bda8114c07584d3d53ae43fd555ca6606c76274`, git clean, `METHOD_CHANGE=NONE`

## Baselines (n=15)

| condition | lift | full |
|-----------|------|------|
| fixed 4 N | 0/15 | 0/15 |
| fixed 6 N | 15/15 | 15/15 |
| 4→6 at lift onset | 15/15 | 14/15 |

No >10 N slam (peak ≤ 6.92 N). Measured force enters the 5.5–6.8 N band in the **same 50 ms step** as the target change (finger-z slots + servo). Latency does not dominate.

## Rescue curve (oracle lead vs this-session T_slip)

| lead | steps | n | lift SR | full SR | 6 N before T_slip |
|------|------:|--:|--------:|--------:|------------------:|
| 500 ms | 10 | 15 | 14/15 | **13/15 (0.87)** | 15/15 |
| 400 ms | 8 | 15 | 15/15 | **15/15 (1.00)** | 15/15 |
| 300 ms | 6 | 15 | **15/15** | **9/15 (0.60)** | 15/15 |
| 250 ms | 5 | 15 | 14/15 | 6/15 (0.40) | 15/15 |
| 200 ms | 4 | 15 | 12/15 (0.80) | 2/15 (0.13) | 15/15 |
| 100 ms | 2 | 5 | 2/5 | 0/5 | 5/5 |
| 0 ms | 0 | 5 | 3/5 | 0/5 | 0/5 |
| +100 ms | −2 | 5 | 0/5 | 0/5 | 0/5 |

Primary endpoint = **full** (this-episode lift AND basket contact > 0.05 N).

## Actionable window

- **Full SR ≥ 0.8:** latest command lead **400 ms** (range ~400–500 ms; 8–10 steps)
- **Lift SR ≥ 0.8:** latest **~200 ms** (4 steps)
- At P1’s 300 ms: lift is already saved; full task is not (≥0.8)

At 300–200 ms, 6 N is on **before** baseline T_slip and lift usually succeeds, but place often fails (**Case B** for the full task): the grasp is already too degraded to finish the downstream skill.

## P1 vs P2

- P1 deployable marker warning: **~300 ms** before T_slip
- P2 required full-task action: **~400 ms** before T_slip
- Marker warning early enough for **full** success: **NO**
- Marker warning early enough for **lift** success: **YES**

Status: `P2_WINDOW_EXISTS_BUT_P1_SIGNAL_TOO_LATE`

Do **not** start OURS. Task-native online correction from the current GelSight cue is not early enough for the qualified full-task endpoint. Next: probe / pre-contact prior (or a more conservative initial force). Marker-triggered 4→6 remains optional only if the research target is lift-only, which it is not.
