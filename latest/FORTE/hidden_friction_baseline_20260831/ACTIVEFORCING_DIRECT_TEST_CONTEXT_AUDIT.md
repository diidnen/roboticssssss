# ActiveForcing-Direct TEST Context Audit

Status: **PASS — FROZEN_TEST_TIME_CONTEXT_BUILDER**

The corrected runtime audit establishes an executable TEST-time context path
without pre-generated TEST context files and without loading TEST outcomes.

## Frozen online path

```text
fresh TEST R0
→ frozen P4-B probe (raw telemetry)
→ exact 46-feature probe builder + TRAIN-only probe normalization
→ frozen FrictionGRU checkpoint → online mu_hat, sigma_mu
→ branch_hold query skeleton from post-probe EEF command
→ frozen Direct FEASIBILITY_ONLY scoring for 3.00–5.00 N
→ minimum score >= 0.5, otherwise existing max-force fallback 5.00 N
```

The branch-hold skeleton has eight rows because the frozen feasibility decoder
uses the first `H=8` segment. The 215-step P4-B probe is not fed directly as a
215-step Direct sequence; it is consumed by the frozen probe feature builder
and estimator. No full-task outcome, frontier label, GT friction, AFI rollout,
post-probe physical state, or new visual feature enters Direct inference.

## DEV equivalence evidence

Artifact: `DEV_CONTEXT_REBUILD_EQUIVALENCE.json` / `.md`

- strict pre-probe state rebuilt from raw P4-B telemetry: max diff `0`
- raw probe endpoint to canonical branch-hold command: max diff `6.33e-8`
- rebuilt-vs-reference Direct score max diff: `0`
- endpoint-skeleton-vs-reference Direct score max diff: `1.39e-8`
- all three audit paths selected `4.25 N`
- historical nine-score curve reproduced exactly
- TEST roots accessed before formal opening: none

The historical GT μ used in the score-reproduction subtest is explicitly
audit-only. Formal TEST Direct uses only the online frozen `mu_hat` produced
from that case's raw probe telemetry.

This resolves the previous `ACTIVEFORCING_DIRECT_TEST_CONTEXT_CONTRACT_MISSING`
artifact by adding the pre-TEST runtime builder; it does not modify the frozen
model, checkpoint, feature ordering, normalization, grid, threshold, or
fallback.
