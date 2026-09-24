# E-AF-STRICT-NOQUERY-PREPROBE-PAIRED-20260910

Status: completed; independent audit passed.

## Question and claim boundary

Does ActiveForcing improve full-task success over a strict No-Query baseline when both branches
start from one established-grasp pre-probe state and use the same VLA noise-seed schedule?

The previous 18/24 run is excluded from this comparison. It used a TRAIN friction prior integrated
through the feasibility model and is retained only as exploratory
`NO_PROBE_PRIOR_MARGINALIZED_FEASIBILITY` evidence.

## Frozen baseline selection

- Source: `FEASIBILITY_VALID_ROWS.csv`, SHA-256
  `73fe1b124278fc3685ecce480e339bb0cea9c0b428007388ca80c16353ee5890`.
- Selection split: TRAIN only. TEST outcomes are ignored. No simulator action is executed during
  selection.
- Candidate support: empirically executed forces 3.00, 3.25, ..., 5.00 N.
- Objective: mean `p*(5-F)/5 + (1-p)*(-1)`; exact utility ties choose the lower force.
- Primary estimator: complete paired TRAIN contexts. One incomplete context
  (`task5/root5100/LOW`, missing 4.25 N) is removed for every force. Available-row results are a
  sensitivity analysis only.
- Global optimum: 4.25 N.
- Frozen stronger per-task policy: task 0 = 4.25 N, task 1 = 4.25 N, task 5 = 4.25 N,
  task 6 = 3.50 N.
- TRAIN comparison on 47 complete contexts: global 43/47, utility 0.0521277; per-task 43/47,
  utility 0.0904255. Post-freeze VAL diagnostic: both 11/12, utility 0.0541667 vs 0.0916667.
- Authoritative remote freeze: `/home/exouser/strict_noquery_freeze_20260910/FIXED_FORCE_FREEZE.json`,
  SHA-256 `ee9ee3f537348ca8b527f64216979af460e4b40a45be3a5df158661c668375af`.

## Treatments and invariants

- Unit: each of the existing 24 evaluation contexts in the frozen final `DEV_PLAN.json`.
- Common origin: the exact pre-probe state after the canonical 45 approach, 35 descend, 70 close,
  and 40 hold steps, including exposed environment state, controller/manager/sensor caches, joint
  targets, material readback, and Python/NumPy/Torch/CUDA RNG.
- Strict No-Query: zero `probe_out`, `probe_back`, or `probe_hold` actions; no belief module; no
  feasibility module; force is the frozen task lookup above.
- ActiveForcing: unchanged physical probe suffix, frozen belief, frozen feasibility model, and its
  selected force.
- Both branches: same context ID and therefore the same deterministic VLA noise seed at every common
  replan step; same checkpoint, observation adapter, online 50-action/10-executed schedule,
  arbitration, force controller, 350-step evaluator, task instruction, scene root, and hidden
  simulator friction.
- Branch order: strict runs from the live suspended pre-probe state; the saved state is then restored
  exactly before the ActiveForcing probe suffix resumes.

## Gates

1. Before strict inference, runtime module inventory contains no belief or feasibility module.
2. Strict prefix has exactly 190 steps and only approach/descend/close/hold phases; last ten hold
   rows have bilateral contact; physical-probe action count is zero.
3. Material readback equals the context's simulator friction, but strict selection never reads it.
4. After strict rollout, all six exposed snapshot groups (`state`, `environment`, `objects`,
   `joint_targets`, `materials`, `rng`) restore with exact recursive equality.
5. ActiveForcing executes at least one `probe_out` action and passes the frozen probe qualification.
6. Initial VLA noise seed and noise bytes match across the pair; every recorded RPC uses the frozen
   context-derived seed schedule.
7. Both branches have verified online VLA provenance and a valid full-task label. Task failure is a
   scientific outcome, not an execution failure.

Any gate failure stops that context and the 24-pair formal run. Fixing an implementation defect
requires a new output namespace and a repeated smoke before formal execution.

## Metrics and decision rule

- Primary descriptive endpoint: paired full-task-success difference, ActiveForcing minus strict
  No-Query, across all 24 contexts.
- Report exact discordant counts (AF-only success and strict-only success), task-stratified counts,
  force selections, and all invalid/excluded units.
- With only 24 pairs, uncertainty must be shown using the exact paired/binomial or McNemar framing;
  no post-hoc threshold turns a descriptive result into confirmation.
- The run is paper-grade causal evidence for these frozen 24 contexts only if every pairing and
  provenance gate passes. The snapshot API does not expose PhysX contact/solver warm-start state;
  this is a declared simulator-internal limitation and must remain in the paper caveat.

## Resource and stopping budget

- Smoke: one context, at most two 350-step rollouts and 70 online VLA RPCs.
- Formal: 24 pairs, at most 48 rollouts and 1,680 RPCs, only after the smoke passes.
- Output filesystem: `/media/volume/data` (198 GB free at freeze time); do not write the formal data
  to nearly full `/home` or `/media/volume/newdata`.

## Execution log

- `smoke_v1` and `smoke_v2` stopped before any strict VLA rollout because the information gate
  detected an indirect import of `activeforcing_feasibility_features`.
- `smoke_v3` was aborted before a valid environment launch when an unrelated 30.8 GB GPU job caused
  Isaac to report device OOM and fall back to CPU physics. That external process was not modified.
- `smoke_v4` passed: strict and AF each completed 350 steps and 35 online VLA RPCs; strict used
  4.25 N, loaded no belief/feasibility module, and executed zero probe actions; all six exposed
  snapshot groups restored exactly; all 35 paired VLA noise seeds and noise-byte hashes matched.
  Both branches succeeded in the smoke context; AF selected 4.45 N.
- Formal output namespace: `/media/volume/data/exouser/af_strict_noquery_paired_20260910/formal_v1`.

## Frozen result

- Valid pairs: 24/24. Strict No-Query succeeds 19/24 (79.17%); ActiveForcing succeeds 18/24
  (75.00%); paired difference AF minus strict is -1/24 (-4.17 percentage points).
- Discordant pairs: one AF-only success (`t1_r170048_mid`) and two strict-only successes
  (`t0_r170048_high`, `t5_r170049_mid`). Exact two-sided McNemar p = 1.0.
- Per task (strict versus AF): task 0 is 5/6 vs 4/6; task 1 is 3/6 vs 4/6; task 5 is
  5/6 vs 4/6; task 6 is 6/6 vs 6/6.
- All 24 strict information gates pass, all 24 exposed pre-probe restore gates pass, and all
  840 paired common VLA replan steps have identical noise seeds and noise-byte hashes. The
  independent audit checked 1,680 total RPC receipts.
- Result SHA-256: `84b91ccfdd9ecc15083f1865aad539ff403d747bd2f1fb8cdbf91aa1c0e7940a`.
  Independent audit SHA-256: `eb9d58eda35f77527c9bb08339e2fb3059e39a93275745d6d386fe5305cc2444`.
- Decision: on this frozen 24-context paired set there is no evidence that AF improves over the
  stronger strict No-Query baseline. The point estimate slightly favors strict No-Query, but only
  three pairs are discordant, so the experiment is underpowered for a general superiority claim.
- The prior 18/24 prior-marginalized-feasibility run remains exploratory and excluded.
