# Formal TEST Entry Gate

Status: **FORMAL_TEST_ENTRY_PASS**

All four required gates passed before TEST roots were opened:

1. `DIRECT_SELECTOR_SEMANTICS_PASS`
2. `ORIGINAL_ROOT_RESTORE_EQUIVALENCE_PASS`
3. `TASK0_NEWTON_FORCE_TO_PI0_PASS`
4. `ACTIVEFORCING_DIRECT_DEV_CLOSED_LOOP_PASS`

Gate 4 includes the corrected runtime context audit. Raw P4-B telemetry is
converted online by the frozen probe feature builder and frozen estimator;
the Direct decoder then scores all nine forces. The historical equivalence
audit reproduced the known nine-score curve and selected 4.25 N in all audit
paths. The actual DEV online probe path restored R0 exactly and executed the
frozen π0 loop.

TEST is now opened for the preregistered two-method matched comparison only:

```text
task0
roots 5174–5179
mu = 0.2, 0.5, 1.0
10 repeats per root×mu×method
```

This is 180 rollouts per method, 360 total. The previous 810-rollout
frontier/privileged scope is not run in this round. `rho_frontier=0.8` remains
evaluation-only; the Direct online threshold is `0.5`.
