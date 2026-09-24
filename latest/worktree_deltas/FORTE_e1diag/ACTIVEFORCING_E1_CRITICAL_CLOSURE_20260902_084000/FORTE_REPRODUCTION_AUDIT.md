# FORTE faithful-reproduction audit

## Status

`FORTE = FAITHFUL_REPRODUCTION_BLOCKED`

## Recovered authoritative semantics

FORTE is a hardware reactive grasp controller, not a fixed-force baseline and not Direct+Utility. The checked-out implementation samples six analog pressure channels at 2 kHz, maps the current tactile state to a scalar force with a frozen 24-feature RBF-SVR checkpoint, computes a Welch spectral estimate over 10–50 Hz to detect slip, and, on slip, incrementally closes a Dynamixel gripper under impedance control (the showcase uses a 0.006 position decrement and 0.2 s cooldown after an initial load increment).

Runtime objective: tactile slip suppression through reactive impedance-position correction. Common evaluation, once faithfully rolled out, may use the project realized Utility; that evaluation must not replace FORTE's runtime logic.

## Provenance

- Isolated checkout: `/home/exouser/FORTE_e1diag`, commit `7f88d018`.
- Official showcase: `examples/gripper_showcase_vis_realtime.py`, SHA-256 `580359994a0ee8c401b6be6b8b8ab20bb7bc37a52cbd455d8568888b5f692dab`.
- Force/slip runtime: `forte/runtime/force_and_slip.py`, SHA-256 `881458a2b67d5ab8b716dd279dc4ec0f4513150a5080aadb4b06f67a4826016e`.
- Frozen SVR: `models/SVR_ckpt.pkl`, SHA-256 `9d9ba2449282196db1a85f31e1e41cdca7ccd56522130e2ff3deffd93e8017bc`; sklearn 1.6.1; 24 input features.
- Prior audit: `/home/exouser/Tabero/analysis/results/f1_forte_tabero_baseline_20260818_194501`.

## Why final paired E1 numbers are blocked

The E1 simulator archive contains contact-force trajectories, not FORTE's six-channel analog sensor stream, the calibrated SVR feature path, or a faithful Dynamixel impedance actuator adapter. No `/dev/ttyACM*` or `/dev/ttyUSB*` hardware interface is present in the prior audit. Replaying an oracle slip label or a fixed/ladder force policy would change the scientific method.

Archived `FORTE-INSPIRED REACTIVE` / oracle-slip outputs are explicitly excluded and are not renamed as FORTE. No same-task/root/friction/initial-state/seed native FORTE rollouts exist; missing table values remain NA rather than becoming surrogate results.

Engineering unblock requirement: an audited simulator adapter that produces the same six-channel input semantics and maps FORTE's impedance position increments to the simulated gripper without changing slip detection or control logic, followed by fresh same-tuple E2E rollouts with measured-force telemetry.
