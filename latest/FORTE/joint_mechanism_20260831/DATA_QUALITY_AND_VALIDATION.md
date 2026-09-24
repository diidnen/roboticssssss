# Data quality and validation

Status: complete. TRAIN/root-heldout diagnostics, all taskwise prospective DEV evaluations, matched pooled DEV, and original pooled taskwise recomputation are validated.

## Unit of analysis

The independent unit is a pre-branch physical context `(task, simulator root, sampled friction)`. A force branch is a repeated execution conditional on that context. Physical timesteps are trajectory observations and are never counted as independent contexts.

Current taskwise TRAIN has 18 contexts, six simulator roots and 180 branches per task. Each context has five continuous-force strata and two repeats per selected force. The pooled current population has 72 contexts, 24 roots and 720 branches. The old continuous outcome pool has 72 contexts, 24 roots and 1,008 outcomes: 720 new continuous branches plus 288 historical coarse branches.

## Completeness and alignment

| task | contexts | roots | branches | corrected telemetry | visual alignment | label source |
|---:|---:|---:|---:|---|---|---|
| 0 | 18 | 6 | 180 | 180/180 | 18/18 | direct |
| 1 | 18 | 6 | 180 | 180/180 | 18/18 | 40 direct, 140 reconstructed |
| 5 | 18 | 6 | 180 | 180/180 | 18/18 | 180 direct |
| 6 | 18 | 6 | 180 | 180/180 | 18/18 | 180 direct |

task1 is `PASS_WITH_CAVEAT`: the available 40 direct labels agree with the frozen fallback 40/40, but 140 labels cannot be directly recovered. task5 and task6 are the stronger replication evidence because all labels are direct. task5/task6 each show two direct-versus-fallback discrepancies, confirming that fallback labels are not treated as interchangeable with direct labels.

## Leakage controls

- Root-heldout CV holds out two complete roots per fold, six contexts and 60 branches.
- Fold normalization is fit on training roots only.
- Visual-model PCA is fold-local in the original CV. Joint-NoVisual never consumes visual/PCA input.
- Full-data checkpoints are frozen before prospective DEV is read.
- DEV and previously viewed root splits are retrospective diagnostics only and cannot select a model, λ, PCA rank, hidden size, epoch, seed or force grid.
- Pooled training is evaluated within task on held-out roots and is not described as unseen-task generalization.

## Independent arithmetic validation

`VALIDATION_RESULTS.json` independently recomputes:

1. fold means for Base and Joint-NoVisual on task0/1/5/6;
2. fold means for pooled Base and pooled Joint-NoVisual, separately by task;
3. task0 DEV probability MAE, Brier and NLL for four models;
4. task1/task5/task6 DEV probability MAE, Brier and NLL for four models per task (12 checks);
5. matched pooled DEV probability MAE, Brier and NLL for Base/JNV by task (eight checks);
6. original pooled Visual taskwise probability MAE, Brier and NLL for four models by task (16 checks).

All checks pass within a declared `1e-8` tolerance, which covers only CSV/float32 serialization rounding. The taskwise and pooled assemblers consume frozen CSVs only; they do not call training or selection code.

Prospective DEV contains task0=2 contexts/2 roots/90 branches, task1=4 contexts/2 roots/180 branches, task5=2 contexts/2 roots/90 branches, and task6=1 context/1 root/25 branches. Task6 is reported but cannot independently support a multi-context generalization claim.

## Metric interpretation caveat

The root-CV empirical frontier is reconstructed from five random continuous forces with two repeats per context. It is useful for within-protocol mechanism comparison but is coarser and noisier than the prospective DEV grid. NLL/BCE, probability MAE and Brier are therefore reported alongside frontier, under-force and monotonicity. A model is not called recovered merely because one probability metric improves while safety or calibration worsens.

## Excluded evidence

The pre-existing partial old nested-context run is excluded from the primary conclusion. Its `TASK0_ONLY_N18` setting was evaluated against DEV contexts from all tasks rather than task0 only, and only some nested settings completed before the run was stopped. Reusing it would mix task-distribution shift with context count.

The original pooled Visual Joint is also excluded from the matched causal contrast because it changes the physical target, architecture and IE semantics. It remains reportable as an implementation-confounded historical pipeline result.

The portable HTML report passed schema, data-reference, source-provenance and structural verification. Browser viewport/source-interaction verification was unavailable because the environment has no installed Chromium executable; this affects presentation QA only, not metric validation.
