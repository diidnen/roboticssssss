# Irreversible-failure early-stop validation

Date: 2026-08-31 UTC

## Verdict

No runtime irreversible-failure early-termination condition is enabled. The historical data do not provide a sufficiently validated, protocol-independent condition for authoritative use.

## Retrospective evidence

The completed non-TEST TRAIN trajectories contain 520 full branches: 391 final successes and 129 final failures. The existing `dropped` signal is `0` for all 520 rows, so it cannot validate an online dropped-object termination rule. The existing `lost_in_transit` field is `1` for all 520 rows, including all 391 successes, so it is not a safe irreversible-failure predicate.

A conservative exploratory condition—object z at least 2 mm below the branch-start z during lift/transit/over-basket/place—was present in only 5 historical episodes, all failures and zero successes. This sample is too small to establish the required “after this point success is impossible” claim, and the condition is not part of the frozen evaluator contract. The 5 mm version occurred only twice. These observations are retained as diagnostics, not as a scientific selection rule.

## Gate result

The required zero-false-positive retrospective check is not sufficient here because the only plausible geometric condition has inadequate support and no runtime shadow implementation was parity-tested. Therefore:

```text
runtime_early_stop = false
scientific_retry = 0
outcome_semantics = unchanged
```

No historical early termination was substituted for a full-task failure, and no TEST trajectory was truncated.
