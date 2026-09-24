# π0 action-cache audit

Date: 2026-08-31 UTC

## Verdict

Whole-trajectory π0 caching is **not enabled** and is not legal for the current authoritative root-scaling collector.

The current collector captures the frozen visual feature at the pre-probe boundary, then delegates each force branch to the P5-S0-C runner. That runner uses the frozen scripted P4-B Cartesian phases and constructs an action on each simulation step; it does not execute a branch-level open-loop π0 action trajectory. Replacing that behavior with cached π0 actions would change the downstream protocol.

The separate action-capable π0 server is not evidence that this collector is open-loop. The existing downstream audit documents missing/undefined formal ActiveForcing π0 chain semantics and state needed for exact replay, including observation-conditioned inference and temporal/server state. Under the stated rule, force changes state, state changes RGB/state, and π0 action may change. A cached trajectory would therefore be a closed-loop semantic change unless a frozen protocol explicitly authorized nominal-motion replay and sequential-vs-cached parity passed.

## Checks

- candidate-force branches do not share a frozen π0 action trajectory in the current runner: **not applicable**;
- strict open-loop/frozen-nominal protocol: **not established**;
- sequential-vs-cached parity: **not run**, because no legal open-loop cache target exists;
- TEST inference: **not used for this audit**;
- cache enabled in authoritative TEST/TRAIN: **no**.

The authoritative collector is consequently frozen with `pi0_action_cache=false`.
