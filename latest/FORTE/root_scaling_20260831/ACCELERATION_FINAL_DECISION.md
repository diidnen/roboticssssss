# Acceleration final decision

Date: 2026-08-31 UTC

## Final authoritative choice

Use the existing sequential collector for both task0 and task5, with `num_envs=1` and `worker_count=1`. No physics, timestep, image resolution, π0 inference frequency, horizon, timeout, force grid, repeat count, branch mapping, success semantics, model, split, or learning-curve gate changed.

## Priority decisions

1. **Legacy excluded work — PASS.** The inherited task0 collector stopped at an atomic context boundary after root63. All remaining unperformed targets (`root64`–`root73`) are preregistered-excluded. Required S50 roots `root12`–`root55` are complete and independently QA PASS.
2. **Vectorization — REJECTED / NOT ENABLED.** There is no frozen vectorized implementation that can meet the requested field-by-field parity gate. Candidate `num_envs=2` and `4` were not run; measured speedup is therefore N/A, not an assumed gain. The sequential reference is 0.035754 completed execution branches/s by the stage-log calculation.
3. **Irreversible-failure early stop — NOT ENABLED.** Existing failure signals are either degenerate or too weakly supported for a runtime zero-false-positive claim. Full-task rollout remains unchanged.
4. **π0 caching — NOT ENABLED.** The current collector is scripted P4-B downstream, not a legal frozen open-loop π0 trajectory replay. Closed-loop/stateful cache semantics are not established.

## Population accounting

```text
collected_total = 52
included_in_S50 = 44
excluded_legacy_collected = 8
excluded_legacy_remaining = 10
```

The 8 collected excluded roots are retained only as legacy audit material. None of `root56`–`root73` enters PCA, normalization, TRAIN, S50, metrics, or model selection. The accidental partial task0 TEST launch produced no authoritative context/branch/parity row or commit; its files are quarantined and excluded.

## Gate

The supervisor has been changed from the legacy `62/620` requirement to the S50 required-root QA gate, and it additionally requires this acceleration freeze before launching untouched TEST. TEST remains the first scientific inference stage after this freeze; task0 and task5 use the same frozen sequential implementation.
