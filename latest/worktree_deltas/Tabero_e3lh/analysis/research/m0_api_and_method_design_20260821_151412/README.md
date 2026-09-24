# M0 API And Method Design

Status: `M0_METHOD_DESIGN_COMPLETE_WITH_DELIGRASP_PENDING`.

Timestamp: `2026-08-21T15:14:12Z`.

This directory records two intentionally separated decisions:

1. A non-blocking DeliGrasp authentication audit.
2. A design-only pass for the final OURS method formulation.

No DeliGrasp venv was created, no package was installed, no network call was made, no prompt was sent, and no credential value was printed or copied. DeliGrasp remains a pending baseline, not a blocking dependency for the project.

No training, dataset creation, probe modification, threshold tuning, OURS implementation, B3 evaluation rerun, D2 modification, or Tabero source edit was performed.

## DeliGrasp Status

```text
API_KEY_AVAILABLE = NO
AUTH_MODE = CHATGPT_LOGIN
DELIGRASP_API_AVAILABLE = NO
B3_BLOCKED_BY_DELIGRASP_API remains true
```

Reason: the current shell does not expose an `OPENAI_API_KEY` for the Python SDK path. Codex auth metadata indicates ChatGPT login, which is not a reusable `OPENAI_API_KEY` for DeliGrasp. The `openai` Python package is also not available in the current Python environment.

Because there is no SDK key, the conditional artifacts below were not created:

```text
DELIGRASP_PROMPTS.jsonl
DELIGRASP_RAW_RESPONSES.jsonl
```

## Current Method Decision

The strongest current OURS formulation is:

```text
early probe evidence
-> task-conditioned force-sufficiency belief
-> minimum sufficient force for full downstream success
```

The paper-facing target should be:

```text
P(full-task success | probe evidence, task/context, candidate force F)
```

not:

```text
estimate mu -> force lookup
```

## Evidence Used

Decision-positive tasks:

```text
task0 alphabet soup: 5 / 4 / 3 N
task1 cream cheese: 6 / 5 / 3 N
task2 salad dressing: 8 / 3 / 3 N
task5 tomato sauce: 5 / 4 / 3 N
task6 butter: 4 / 3 / 3 N
```

Fixed-low negative controls:

```text
task3, task7, task9
```

Task8 has no feasible low-friction range and is not part of the decision-positive force benchmark.

D2 task1 showed a qualified mechanism POC:

```text
small pre-lift probe
-> deployable force/tactile evidence
-> explicit physical decision
-> explicit force
-> full task
```

with full SR 1.00 and mean selected force about 5.07 N, versus Fixed Robust at about 0.983 SR and 6 N. This validates the mechanism, but its handcrafted cream-cheese thresholds are not the final method.

## Artifacts

- `DELIGRASP_AUTH_AUDIT.md`
- `DELIGRASP_AUTH_STATUS.json`
- `FINAL_METHOD_DESIGN_SPACE.md`
- `METHOD_NOVELTY_MATRIX.md`
- `METHOD_INPUT_OUTPUT_SPEC.md`
- `METHOD_TRAINING_TARGET_OPTIONS.md`
- `METHOD_GENERALIZATION_REQUIREMENTS.md`

