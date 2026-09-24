# M0 API and Method Design

Status: `M0_METHOD_DESIGN_COMPLETE_WITH_DELIGRASP_PENDING`.

This directory handles two tasks with different priorities:

1. Non-blocking DeliGrasp authentication check.
2. Main task: paper-level design of the proposed Tabero physical-force method.

DeliGrasp remains a pending baseline, not a project blocker. No D2, B2/B2-R2, E2E, B3, Tabero core, frozen benchmark, probe rule, oracle, network, dataset, or training code was modified.

## Key Results

- `DELIGRASP_API_AVAILABLE = NO` for the DeliGrasp Python SDK path.
- Codex auth metadata has API-key-shaped metadata, but no `OPENAI_API_KEY` is exported to the shell. Per safety rules, no Codex credential was extracted, printed, copied, or reused.
- B3 remains `B3_BLOCKED_BY_DELIGRASP_API` and is now a pending baseline.
- Current best method formulation is **Direct Force-Sufficiency Belief**:

```text
probe evidence + task/context + candidate force F
-> P(full-task success | evidence, task, F)
-> choose minimum F above tau
```

## Produced Artifacts

- `DELIGRASP_AUTH_AUDIT.md`
- `DELIGRASP_AUTH_STATUS.json`
- `FINAL_METHOD_DESIGN_SPACE.md`
- `METHOD_NOVELTY_MATRIX.md`
- `METHOD_INPUT_OUTPUT_SPEC.md`
- `METHOD_TRAINING_TARGET_OPTIONS.md`
- `METHOD_GENERALIZATION_REQUIREMENTS.md`

## Evidence Inputs

- B2-R2 benchmark breadth: 5 decision-positive tasks `[0, 1, 2, 5, 6]`.
- D2 mechanism POC on task1: small pre-lift probe -> deployable evidence -> explicit force -> full task, with full SR 1.00 and mean selected force about 5.07 N.
- R1/FORTE-style reactive evidence: post-slip correction was too late on low-friction task1.
- B3 DeliGrasp: official code inspected, quantitative baseline pending API.

