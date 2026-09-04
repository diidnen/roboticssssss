# ActiveForcing core transfer bundle

This directory contains the implementation needed to continue ActiveForcing
integration on another machine.

## Frozen control contract

- The formal control path remains Tabero's native `ForcePositionAction`.
- The native 13-D action layout and squeeze metric are unchanged.
- Each step keeps the absolute end-effector arm target and uses measured-force
  feedback together with the Newton target.
- The model/executor boundary is one scalar: `target_force_n: float`.
- No object-specific force-to-gap lookup is used.
- The native 13-D policy mode remains unchanged. Persistent correction is only
  an opt-in scalar-Newton behavior.
- The executor/model has not been claimed as an executed end-to-end model
  rollout yet; this bundle provides the code and interfaces for that next step.

## Layout

| Path | Purpose |
| --- | --- |
| `forte/` | Sensor buffering, tactile force estimation, and slip runtime |
| `forte_gripper/` | FORTE Dynamixel gripper driver |
| `configs/` | Sensor and actuator configuration templates |
| `models/SVR_ckpt.pkl` | Force-estimation checkpoint used by the runtime |
| `examples/` | Sensor/force/slip demo entry points |
| `model/` | ActiveForcing model/planner and adjudication source |
| `tabero_adapter/` | Snapshot of the native Tabero force-position integration files |
| `ACTIVEFORCING_AUTHORITATIVE_PROTOCOL.md` | Current protocol and scope contract |

## Install the sensor/runtime package

From the repository root:

```bash
python -m pip install -e activeforcing
```

The runtime package requires Python 3.9+ and the dependencies listed in
`activeforcing/pyproject.toml`. Hardware serial ports and paths must be
updated in `activeforcing/configs/` on the destination machine.

## Use with Tabero

The files under `tabero_adapter/` are source snapshots for the existing
Tabero tree, not a second controller package. Keep the official Tabero
`ForcePositionAction` path and use the model/executor boundary as:

```python
target_force_n: float = model_decision.target_force_n
```

The model scripts under `model/` still require the corresponding training
dataset/checkpoints and machine-local paths. They are included as the
authoritative source for the next executor/model integration; no generated
experiment outputs are bundled here.

## Hardware safety

Run the sensor/gripper demos only after checking serial device names,
workspace limits, force limits, and emergency-stop behavior on the target
machine. The included checkpoint estimates force; it is not a substitute for
the simulator/controller safety gates.
