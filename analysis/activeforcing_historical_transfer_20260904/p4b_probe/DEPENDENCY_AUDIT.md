# P4-B dependency audit

This is a source-preserving transfer of the executed P4-B probe. It has not
been modified for this bundle and must not be invoked as part of an offline
parity check.

## Local source dependencies

The probe entrypoint is `p4_collect_probe.py`. It has no imports of another
local Python helper module. Its local runtime root is the Tabero checkout and
it sets `HDF5_TRAJ_SOURCE_DIR`, `LIBERO_CONFIG_DIR`, and
`LIBERO_ASSETS_DATA_DIR` relative to that checkout.

## Environment dependencies

Live execution requires the existing Tabero/Isaac environment, including
`torch`, `numpy`, `isaaclab`, `isaacsim`, `gymnasium`, `tac_manip`, the
Franka Hybrid Tactile task, and the LIBERO assets/configuration. Those are
environment dependencies, not silently copied into this historical bundle.

## Frozen semantics visible in the source

- `approach=45`, `descend=35`, `close=70`, `hold=40`;
- `probe_out=10`, `probe_back=10`, `probe_hold=5` in the historical AFI run;
- P4-B preload starts at 3.0 N, with its existing adaptive/cap rules;
- common contact-frame shear direction;
- contact-loss, normalized shear/rho, displacement, and tactile-marker
  safety/quality checks;
- `_apply_friction` writes both PhysX static and dynamic friction channels;
- telemetry is emitted as `ProbeStep` rows and an aggregate probe record.

`TRUE_FRICTION_IS_PRIVILEGED_ANALYSIS_ONLY = YES`. The friction field is
written for audit labels and is excluded by the transferred 46-D feature
builder.
