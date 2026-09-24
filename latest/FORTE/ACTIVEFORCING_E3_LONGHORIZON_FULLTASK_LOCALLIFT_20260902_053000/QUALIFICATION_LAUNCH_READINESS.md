# E3 minimal qualification launch readiness

Status at 2026-09-02 05:45 UTC: **READY, RESOURCE-GATED**. No E3 Isaac/GPU process has been started.

## Frozen execution contract

- Candidate order is fixed before outcomes: LIBERO-10 task3, then task9, then task5.
- First launch is exactly one task3 episode at seed 7300, friction 0.6, robust 8 N total squeeze.
- 8 N is qualification-only: it establishes nominal frozen-policy capability and is not a task Fmax or ActiveForcing result.
- Frozen server: project-local OpenPI PID lineage documented in `PI0_BACKEND_FREEZE.md`; no LeRobot substitution.
- Client: historical B5/OpenPI observation/action/environment path. The wrapper changes only the task object used by telemetry, the two normal-force setpoint slots after policy inference, and diagnostic goal telemetry.
- Final success remains the authoritative environment termination evaluator (AND across every task goal).

## Readiness checks

| Check | Result |
|---|---|
| Candidate config, USD assets, relationship/close evaluator | pass |
| Historical transform seam cardinality | pass |
| Transformed 93,401-byte source compilation (including read-only root-state hash) | pass |
| Authoritative B5 bundled Warp import (`1.8.2`, `warp.types.array`) | pass |
| Wrapper/analyzer `py_compile` | pass |
| Shell launcher syntax | pass |
| Generic query-state evidence | any measured gripper contact |
| Generic transport evidence | pre-registered object XY displacement >= 0.10 m |
| Placement/close evidence | per-goal authoritative evaluator telemetry |
| Paired-root evidence for force pilot | SHA-256 of relative scene state before actions |
| Mutable output isolation | timestamped E3 result directory only |

## Resource fail-closed gate

Before launch, MASS PID identity and health must be rechecked. Start only when the scheduler grants one worker and GPU utilization is below 70% with more than 10 GiB free. Do not start if free memory is below 8 GiB or utilization exceeds 90%. After task3 loads, resample; if MASS is abnormal or free memory is below 8 GiB, do not add work and report immediately.

The root filesystem had only about 1.5 GiB free during readiness. Heavy telemetry will therefore use a timestamped isolated lane under `/media/volume/newdata/exouser/activeforcing_e3/`; the small manifest and reports remain mirrored in `/home/exouser/FORTE_e3`.
