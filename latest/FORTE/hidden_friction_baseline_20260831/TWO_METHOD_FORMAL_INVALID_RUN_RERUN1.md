# Invalidated TEST Run

The first formal worker was stopped after `t0_s5174_mu0.2_r1` exposed a
runner-order contamination: Neutral was executed before Direct P4-B probe in
the same Isaac worker. The Direct probe failed its frozen validity check.

All output under `TWO_METHOD_FORMAL_RUNTIME` is invalid and excluded from
scientific tables. No method parameter changed. The rerun uses a fresh output
namespace, Direct probe first from R0, explicit frozen friction restoration,
and stops before recording a Direct rollout if P4-B is invalid.
