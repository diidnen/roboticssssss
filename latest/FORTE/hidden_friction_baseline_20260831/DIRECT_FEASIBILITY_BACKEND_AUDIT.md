# Direct Feasibility Backend Audit

Status: **PASS — exact backend identified; no retraining**  
Gate 1: `DIRECT_SELECTOR_SEMANTICS_PASS`  
Audit scope: pre-TEST method-definition and executable contract only.

## Decision

The authoritative proposed backend is the frozen method-development
`FEASIBILITY_ONLY` ensemble from the 2026-08-30 GNP-style continuous pivot:

`/home/exouser/FORTE/gnp_style_continuous_20260830_125107/`

It is a prospective full-task feasibility model. It is not the historical AFI
physics-imagination selector, not the `JOINT` backend, and not the old visual
`rho=0.8` evaluator.

## Contract audit

| Field | Frozen value / evidence |
|---|---|
| Exact script | `/home/exouser/FORTE/gnp_style_continuous.py`, `FEASIBILITY_ONLY` load/predict path |
| Model class | `/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py::FeasibilityOnly` |
| Checkpoints | `GNP_STYLE_CONTINUOUS_FEAS_seed0.pt`, `seed1.pt`, `seed2.pt` |
| Checkpoint hashes | Recorded in `ACTIVEFORCING_DIRECT_SELECTOR_CONTRACT.md` and manifest |
| Input schema | normalized strict pre-probe P4-B 215-step trace; command `:17`, condition `17:` |
| Context representation | existing strict pre-probe last P4-B hold; state/mask/nominal trace from frozen builders |
| Candidate force encoding | candidate `F` inserted through existing nominal trace builder; no new fields |
| Predicted quantity | binary full-task success/feasibility logit, sigmoid converted to probability |
| Ensemble aggregation | arithmetic mean of three seed sigmoid probabilities |
| Online threshold | `p_success >= 0.5` |
| Fallback | no passing candidate → maximum candidate `5.00 N`; record `fallback_used=true` |
| Candidate grid | `3.00:0.25:5.00 N`, nine candidates |
| Normalization | TRAIN-only `GNP_STYLE_TRAIN_NORMALIZATION.json`, stored `x_mean/x_std` |
| Calibration semantics | raw sigmoid ensemble is primary executable score; TRAIN-only isotonic is secondary diagnostic and is not applied |
| AFI separation | no `FRICTION_GRU.pt`, no physics imagination, no `eta=0.9` in Direct path |

## Static verification

The frozen checkpoints load with the expected state dictionary:

```text
command_gru.weight_ih_l0: (192, 17)
command_gru.weight_hh_l0: (192, 64)
condition.0.weight:       (64, 54)
head.0.weight:            (64, 128)
head.2.weight:            (1, 64)
```

The saved checkpoint metadata reports variant
`GNP_STYLE_CONTINUOUS_FEAS`, 1008 TRAIN branches, and the same stored
normalization fields. The 2026-08-30 manifest records `DEV_used=false` and
`TEST_used=false` at freeze time.

## Semantic amendment

The prior artifact `ACTIVEFORCING_DECISION_SEMANTICS_AUDIT.md` remains
unchanged as historical evidence. Its blocker is superseded before TEST by the
user freeze:

`ACTIVEFORCING_DECISION_SEMANTICS_AMBIGUOUS` →
`RESOLVED_BY_PRETEST_USER_FREEZE`.

No historical provenance conflict is used to block runtime Gates 2–4. If a
future runtime issue concerns only missing executable fallback, it must be
reported as a fallback blocker; it must not reopen AFI-vs-Direct semantics.
# Amendment 2026-08-31 — formal TEST context readiness

The pre-TEST user freeze resolves the old semantics ambiguity: the proposed
method is `ActiveForcing-Direct`, its online threshold is `p_success >= 0.5`,
and `rho_frontier=0.8` is evaluation-only. Gate 1 therefore passes.

The exact frozen implementation was then checked for formal TEST readiness.
`gnp_style_continuous.py` uses `preprobe_full_task_feasibility` and its frozen
evaluation helper constructs only `all_contexts_for_split("DEV", active)`;
`guarded_manifest()` explicitly discards TEST rows. No frozen runtime context
builder exists for the requested roots 5174–5179. The available 5174–5179
material is incomplete/aborted collection output and is not a valid exact
Direct context contract for all 18 formal root×μ cells.

This is recorded as `ACTIVEFORCING_DIRECT_TEST_CONTEXT_CONTRACT_MISSING`.
Adding a context builder or swapping to the visual backend would change the
frozen input/backend contract, so formal TEST remains sealed. The DEV Gates 2–4
were nevertheless completed and passed; no TEST simulator rollout was run.
