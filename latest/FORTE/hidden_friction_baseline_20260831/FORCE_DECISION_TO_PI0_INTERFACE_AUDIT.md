# Force-Decision-to-π0 Interface Audit

Audit time: 2026-08-31T12:45:17Z  
Static implementation status: **CONTINUOUS PHYSICAL FORCE OVERRIDE FOUND**  
Gate 3: **BLOCKED / CURRENT TASK0 RUNTIME VALIDATION NOT RUN**

## Exact executable interface

The frozen π0 checkpoint returns a 13-D action:

`[x, y, z, rx, ry, rz, gripper, fLx, fLy, fLz, fRx, fRy, fRz]`.

The authoritative B5 neutral client executes the raw 13-D output. A later paired frozen-π0 runner provides the already-executed Newton-level override required for a continuous force selector:

- File: `/home/exouser/Tabero/analysis/p6g1r1_controller_grasp_vla_handoff.py`
- Function: `run_vla_full(...)`
- Operation: retain π0 Cartesian, rotation, and gripper commands; zero force slots 7:13; write `F/2` to slots 9 and 12.
- Code: `executed[:, 7:13] = 0.0; executed[:, 9] = FORCE_HALF_N; executed[:, 12] = FORCE_HALF_N`.
- Unit: total requested squeeze is recorded in N as `FORCE_N`; each opposing z channel receives `FORCE_N/2`.
- Telemetry: requested total force, raw π0 force slots, executed force slots, measured squeeze, and measured applied force are recorded.

Dynamic per-branch force selection already exists in:

- `/home/exouser/Tabero/analysis/p7a_fixed_invocation_force_frontier.py::run_one`, which sets `FORCE_N=float(force)` and `FORCE_HALF_N=force/2` before frozen π0 execution;
- `/home/exouser/Tabero/analysis/p7b_gnp_physical_belief_force_planning.py::branch_from_query`, which uses the same override after restoring a matched query state.

The P7A executed audit (`analysis/results/p7a_fixed_invocation_force_frontier_20260826_153426`) reports all requested/measured force channels present and matched, so this is not a JSON-only decision field. It physically changes the actions sent to the simulator.

## Semantics and limits

- Interface type: **continuous physical force**, not force-language conditioning.
- No conversion such as `4.25 N → firmly` is needed or authorized.
- Historical executed P7A values include 3, 5, and 8 N. The wrapper accepts a floating-point force, but the current task0 3.00:0.25:5.00 N grid has not been runtime-validated through this exact restored-root pipeline.
- No explicit clipping is performed in the cited override before `env.step`; any controller/environment limits must be audited from the fixed environment configuration before a task0 PASS claim.
- The currently observed port-18881 process is a visual-feature service, not evidence that this action-chunk execution path is presently available.

## Gate decision

The codebase contains the correct kind of physical interface and historical evidence that it affects frozen-π0 execution. Gate 3 is nevertheless not marked PASS because task0, the recovered decision output, the current checkpoint/server call chain, and the exact 3–5 N range were not exercised in a DEV closed loop. Gate 1 blocked that execution.
