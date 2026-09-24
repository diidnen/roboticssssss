# Phase-0 Historical Blocker Reassessment

## Amendment status

The prior artifact and status are retained. This amendment does not delete or overwrite the earlier conclusion.

Previous blocker: `NOT_ESTIMABLE_PHASE0_BLOCKER_ACTIVEFORCING_PROBE_HANDOFF_UNDEFINED`.

Reassessment: `SUPERSEDED_BY_RECOVERED_HISTORICAL_PROBE_SEMANTICS` for the sensing handoff, with remaining evaluation blockers.

## Why the sensing blocker is superseded

The earlier audit assumed the method had to consume a π0-reached grasp and therefore searched for a 215-step-to-probe adapter. Historical executed artifacts show that assumption was wrong. The frozen P4-B sensing method is itself a root-start procedure:

`root reset → approach → descend → close → hold → shear out/back/hold → 215×46 features → friction GRU → μ̂/σ`.

The task0 P4-B run, P5-S0-C collection, and Active Friction Imagination E2E provide actual trajectories, hashes, checkpoint provenance, numeric predictions, and decisions. A fresh non-TEST root 5126 trajectory also passed the exact current estimator path.

## What remains blocked

1. The historical AFI selector uses a 0.9 imagination rule; an exact recovered `FRICTION_GRU → Direct ρ=0.8` executable was not found.
2. The strongest historical E2E restores post-query state, not the exact original task root.
3. Its downstream is scripted. It cannot be used as frozen-π0 formal evidence.
4. Exact original-root state capture/restore and the fresh frozen-π0 downstream runner still require independent infrastructure validation.

These remaining items are not evidence that the probe itself was undefined. They prevent formal TEST opening until resolved without scientific changes.

## Operational status

- ActiveForcing probe: recovered.
- Friction estimator: recovered and reproduced.
- Historical force decision: recovered, but η=0.9 semantics only.
- Exact original-root reset: not recovered.
- Frozen π0 downstream replacement: not executed this turn.
- TEST roots 5174–5179: not opened.
- Formal rollout counts: frontier 0/810, methods 0/720.
