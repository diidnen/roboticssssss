# Fmax provenance audit

## Verdict

For tasks 0, 1, 5, and 6, the recovered `Fmax` values are the endpoints of an explicit task-level “robust force” map and the corresponding task-specific collection/candidate supports. They are not documented controller-certified physical or safety limits. The later Utility implementation reused those endpoints as normalization constants.

Therefore:

`FMAX_GRID != PHYSICAL_COST_SCALE`

The strongest recovered classification is **A + C + E**: candidate-grid maximum, dataset-collection upper bound, and an explicit historical task parameter whose original tuning rationale is not documented in the inspected sources. It is not B (controller physical/safety limit) and was not originally introduced as D (reward normalization constant), although it was later reused for that purpose.

## Authoritative values

| Task | Current Fmax | Candidate/collection support | Recovered origin | Physical/safety limit? |
|---:|---:|---:|---|---|
| 0 | 5 N | 3–5 N | `ROBUST_FORCE_BY_TASK[0]=5`; `TASK_SUPPORT[0]=(3,5)` | No evidence |
| 1 | 6 N | 4–6 N | `ROBUST_FORCE_BY_TASK[1]=6`; `TASK_SUPPORT[1]=(4,6)` | No evidence |
| 5 | 5 N | 3–5 N | `ROBUST_FORCE_BY_TASK[5]=5`; `TASK_SUPPORT[5]=(3,5)` | No evidence |
| 6 | 4 N | 3–4 N | `ROBUST_FORCE_BY_TASK[6]=4`; `TASK_SUPPORT[6]=(3,4)` | No evidence |

The frozen values are present in:

- `/home/exouser/FORTE/UTILITY_FINAL_CONFIG.json` (`task_Fmax_N`)
- `/home/exouser/FORTE/utility_and_causal_ablation_closure.py:40`
- `/home/exouser/FORTE/ACTIVEFORCING_AUTHORITATIVE_PROTOCOL.json:37`
- `/home/exouser/FORTE/analysis/results/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260902_053006/e5_expected_utility_runtime.py:23`

The protocol assembler is unusually explicit: `/home/exouser/FORTE/assemble_activeforcing_closure.py:203` describes Fmax as **“task/archive maximum; frozen Direct runtime uses candidate-grid max.”**

## Why task6 equals 4 N

The earliest recovered git occurrence is Tabero commit `80ab3be09ce884f86cfc2037d3af30bc28061426` (2026-09-01), where `/home/exouser/Tabero/analysis/p5s0d_fresh_e2e_q2f.py:78` defines:

```text
ROBUST_FORCE_BY_TASK = {0: 5.0, 1: 6.0, 5: 5.0, 6: 4.0}
```

The same task6 endpoint is used by the candidate-generation sources:

- `/home/exouser/FORTE/prospective_visual_context_pipeline.py:34`: task6 support `(3.0, 4.0)`
- `/home/exouser/FORTE/gnp_style_continuous.py:69`: task6 support `(3.0, 4.0)`

The inspected archive contains task6 candidates from 3.0015 to 3.9966 N. No inspected source calls 4 N a certified physical limit, damage threshold, or safety limit. The defensible provenance is therefore: **4 N is a historical task6 robust-force/candidate-support endpoint that was reused as Utility’s denominator.** The evidence does not recover why that support endpoint was originally chosen, so a stronger “manually optimized” claim would be unwarranted.

## Global physical/action-range evidence

No controller-certified safe global range was found. The next admissible provenance tier is an authoritative project-wide force action range:

- `/home/exouser/Tabero/analysis/p5s0a_true_matched_dataset.py:61` defines global training forces `[3,4,5,6,8]` N and held-out intermediate forces through 7.5 N.
- `/home/exouser/Tabero/analysis/p6g1r1_controller_grasp_vla_handoff.py:47` executes an 8 N controller setpoint.
- `/home/exouser/Tabero/analysis/p7b_gnp_physical_belief_force_planning.py:300` records a shared grip-force runtime action space.

Accordingly, the primary counterfactual uses `Fref_global=8 N`, and range normalization uses `[3,8] N`. This is a project action-support reference, **not a safety certification**. Sensitivity references `{5,6,8}` N are taken from existing project supports and were not chosen from task6 outcomes.

## Integrity and scope

- Utility config SHA-256: `19c93ad7a57ff5823c53e0171cf297424256d1bbac62df79b7736f0c680415e1`
- Frozen candidate chain SHA-256: `cedcba10eb116eb7247114bc63ce95f65d31794c7225dded056f295787e35073`
- The counterfactual study is TRAIN/DEV, already-observed-data method diagnosis—not fresh TEST evidence.

