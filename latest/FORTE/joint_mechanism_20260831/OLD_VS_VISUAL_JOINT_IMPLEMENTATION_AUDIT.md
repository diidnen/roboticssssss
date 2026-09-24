# Old continuous Joint versus current Visual Joint: implementation audit

Status: **PHASE 0 COMPLETE — READ-ONLY AUDIT**  
Audit cutoff: 2026-08-31 08:27 UTC  
Scope: old GNP-style continuous, task0/task1/task5 taskwise visual runs, the in-progress task6 path, and the original pooled visual path. No model was trained during this audit.

Post-audit completion note: task6 subsequently froze 18 TRAIN contexts/six roots/180 direct-label branches, and the existing DEV collector completed all nine DEV contexts/385 branches without restart. The original pooled pipeline then completed 12 checkpoints and evaluation under its unchanged, implementation-confounded definition. These later facts update completeness only; they do not retroactively alter this Phase 0 implementation audit.

## Executive answer

The old Joint and the current taskwise Visual Joint share the same core physics model, trajectory target, target-matched intervention-effect loss, optimizer family, learning rate, weight decay, batch cap, epochs, seeds, force-stratified sampling pattern, and loss weights. Their most important intended differences are indeed (i) no visual versus PCA visual conditioning and (ii) 72 pooled contexts versus 18 contexts per task.

However, those are **not the only differences**. The old run also pooled four tasks, added 288 historical coarse branches to the feasibility BCE pool, used pooled normalization, used a different simulator-root population, and used a different held-out benchmark population. These are material data/evaluator confounds. The taskwise Visual Joint also offsets only its initialization RNG by `+3000`; that is a minor reproducibility confound, not a capacity or loss change.

More seriously, the currently implemented **original pooled Visual Joint is not the same Joint formulation** as either the old continuous Joint or the taskwise Visual Joint. It changes the physical target from H8×13 to a repeated H8×4 output, removes the authoritative Physics-GRU trajectory head, and replaces target-matched IE supervision with an absolute adjacent-prediction-difference penalty. Its result cannot isolate context diversity and must be reported as implementation-confounded.

## Dataset grain and independent-context accounting

An independent context is one `(task, simulator root, sampled friction)` state before branch-force intervention. Physical timesteps and repeated executions are not independent contexts.

| Run | Tasks | Independent contexts | Simulator roots | Outcome executions | Executions/context | Adjacent IE pairs | Held-out benchmark |
|---|---:|---:|---:|---:|---:|---:|---|
| Old continuous | 0/1/5/6 | 72 = 18/task | 24 = 6/task | 1,008 BCE outcomes: 720 new continuous + 288 historical coarse | 10 new + 4 coarse | 576 = 8/context | 9 contexts, 7 roots, 135 executions |
| task0 visual | 0 | 18 | 6 | 180 | 10 | 144 = 8/context | 2 DEV contexts/roots, 90 executions; plus 3-fold TRAIN root CV |
| task1 visual | 1 | 18 | 6 | 180 | 10 | 144 = 8/context | 4 DEV contexts from 2 roots, 180 executions when available; plus 3-fold TRAIN root CV |
| task5 visual | 5 | 18 | 6 | 180 | 10 | 144 = 8/context | 2 DEV contexts/roots, 90 executions when available; plus 3-fold TRAIN root CV |
| task6 visual path | 6 | 18 | 6 | 180 | 10 | 144 | 1 DEV context/root, 25 executions; TRAIN root CV is primary |
| Original pooled visual path | 0/1/5/6 | 72 | 24 | 720 | 10 | not target-matched | 9 contexts, evaluated taskwise and aggregate |

At the cutoff, the existing task6 collector was still running (host PIDs 137685/227346) and had 170/180 cumulative branch rows plus an eighteenth aligned visual context in progress. It was not restarted.

## Model and training implementation table

| Field | Old continuous Feas / Joint | Current taskwise Base / Full / Visual Joint | task6 | Original pooled visual pipeline |
|---|---|---|---|---|
| Visual input | none | Base: none; Full/Joint: 4096-D frozen feature reduced by per-task TRAIN-only PCA17 | frozen identical taskwise path; PCA17 | 4096-D frozen feature reduced by pooled TRAIN-only PCA64 |
| Non-visual input | 17-D future step sequence + 54-D static condition | identical 17-D sequence + 54-D condition | identical | nominally 17-D sequence + 54-D condition, but built by a separate simplified array builder |
| Hidden dimension | GRU hidden 64 | GRU/Physics-GRU hidden 64; visual projection 16 for taskwise Full/Joint | identical frozen taskwise definition | GRU hidden 64; visual projection 32 |
| Physical trajectory target | H=8 × 13 normalized state-delta channels | H=8 × 13 same normalized state-delta channels | identical | **H=8 × 4** (`measured_force`, object-z, left/right contact), produced by repeating one 4-vector over H |
| IE target | SmoothL1 of predicted trajectory difference versus true normalized trajectory difference for matched adjacent-force branches | identical | identical | **no true IE target**; sums absolute adjacent predicted differences and minimizes them |
| λphys | 1.0 native coefficient | 1.0 | 1.0 | 1.0 inside a different loss |
| λIE | 1.0 | 1.0 | 1.0 | nominally added at 1.0, but semantic objective differs |
| λfeas | 0.3 | 0.3 | 0.3 | 0.3 |
| Optimizer | AdamW, lr 8e-4, weight decay 1e-4, gradient clip 1.0 | identical | identical | same optimizer scalar settings |
| Epochs / seeds | 80; seeds 0/1/2 | 80; seeds 0/1/2 | frozen 80; 0/1/2 | 80; seeds 0/1/2 |
| Batch cap | 64 branches/physical segments | 64 | 64 | full-batch, one optimizer step/epoch |
| Outcome supervision | BCE on 1,008 real branches | BCE on 180 real branches/task | BCE on 180 planned | BCE on pooled 720 planned |
| Force sampling | 5 equal-width support strata × 1 uniform force × 2 repeats/context | same 5-stratum × 2-repeat design | frozen same | same prospective manifests |
| Force support | task0/5 3–5 N; task1 4–6 N; task6 3–4 N | same task-specific supports | 3–4 N | same task-specific supports |
| Repeats | 2/new continuous force; historical coarse adds 4 branches/context | 2/force | 2/force | 2/force |
| PCA/normalization | no PCA; pooled TRAIN normalization over four tasks and 1,008/720 pools | per-task centered PCA rank17 and per-task TRAIN-only normalization | identical | pooled PCA64 and separate pooled normalization |
| Capacity | Base 27,777; Joint 39,337 parameters | Base 27,777; Full/PCA17 29,089; Visual Joint/PCA17 40,137 | same | Base 27,777; Full/PCA64 31,905; Joint/PCA64 36,325 |
| Joint initialization | authoritative pretrained H8 Physics-GRU; seed as 0/1/2 | same authoritative pretrained checkpoint; Joint RNG uses seed+3000 | identical | no authoritative pretrained Physics-GRU |

## Input semantics

The shared 17-D step input contains future command/phase/task/force terms from the strict pre-probe representation. The 54-D condition contains force, friction and pre-probe state/masks according to the authoritative trajectory-imagination builder. Neither Base nor Joint-NoVisual will receive RGB, PCA components, visual IDs, post-probe outcomes, or frontier labels.

Visual features are aligned once per strict pre-probe context and are repeated across all force branches of that context. This makes them a plausible root/context identifier even though they do not directly contain the outcome label.

## Supervision and label quality

- Old continuous: all models receive the same 1,008 branch outcomes. Joint alone receives corrected physical/IE targets from only the 720 new branches. Historical coarse branches contribute BCE only.
- task0: 180/180 direct labels and corrected telemetry; 141 successes, 39 failures.
- task1: 40/180 labels were directly retained; 140/180 were reconstructed by the frozen fallback. The 40 comparable direct rows agreed 40/40, but this remains a **material label caveat**.
- task5: 180/180 direct labels; fallback would differ on 2/180. This is the clean direct-label replication and must not be merged with task1 as equivalent evidence.
- task6: 180/180 direct labels and corrected telemetry; 175 successes and five failures. This was verified after the Phase 0 cutoff.

## Root-heldout split and leakage audit

The taskwise TRAIN CV holds out two complete simulator roots per fold: six contexts and 60 branches. PCA and normalization are fit within each fold; seed 0 is frozen for this retrospective diagnostic. This prevents the same visual root from appearing on both sides of a fold.

The already-viewed root-heldout results and DEV roots are retrospective mechanism diagnostics only. They cannot select models, λ values, PCA rank, architecture, epoch count, seeds, or force grids, and they are not relabeled as untouched DEV.

## Confound register

| Confound | Severity | Why it matters | Required handling |
|---|---|---|---|
| 72 pooled contexts vs 18 single-task contexts | target variable | This is Hypothesis B | isolate with fixed Joint-NoVisual and current pooled Base/JNV |
| visual PCA feature present only in current taskwise Joint | target variable | This is Hypothesis A | isolate with Joint-NoVisual |
| old feasibility pool has 288 extra coarse outcomes | material | Changes outcome count and force distribution independently of context count | report separately; do not attribute old/current gap solely to visual/context count |
| old pooled task composition vs current single task | material | Task diversity can regularize representation and changes conditional distribution | quantify taskwise pooled performance; never call it unseen-task generalization |
| pooled vs per-task normalization | material | Changes feature scaling and optimization | preserve each frozen protocol; flag when comparing regimes |
| different root populations and DEV force cells | material | Dataset/evaluator shift can change frontier and probability metrics | compare only within each benchmark; no direct pooled p-value claim across benchmarks |
| Joint RNG seed offset `+3000` in taskwise Visual Joint | minor | Changes initialization draw while nominal seed IDs remain 0/1/2 | retain frozen behavior; do not tune or cherry-pick |
| task1 reconstructed labels | material for task1 | Could alter outcome supervision | report separately; require consistency with task5 direct-label evidence |
| original pooled Joint changes architecture/physical target/IE semantics | **critical** | It is not a same-formulation context-diversity test | report as implementation-confounded; use a separately frozen matched pooled JNV diagnostic for the core Base↔JNV question |
| evaluator/calibration differences | material | old reports raw ensemble primary plus TRAIN-isotonic secondary; taskwise uses raw ensemble | use raw ensemble as common primary; isotonic remains secondary only |

## Does implementation explain the old/current result gap?

It can plausibly contribute, but Phase 0 alone cannot assign causality. For the **authoritative old versus taskwise Visual Joint**, there is no architecture/loss change in the physics auxiliary large enough to dismiss the visual-shortcut/context-diversity hypotheses: the core H8×13 Physics-GRU and target-matched IE objective are the same. The outcome-pool, normalization, root-population, evaluator, and RNG differences are nonetheless real confounds and prevent a clean historical causal comparison.

For the **original pooled Visual Joint**, implementation differences are large enough to explain a result difference by themselves. That pipeline cannot answer whether more contexts rescue the current authoritative Joint.

## Phase 0 decision

Proceed with the frozen Joint-NoVisual diagnostic. The clean primary contrasts are:

1. taskwise `Joint-NoVisual − Base`, which isolates the physics/IE auxiliary without visual input;
2. taskwise `Visual Joint − Full`, which measures the incremental physics/IE effect when visual conditioning is already present;
3. current pooled matched `Joint-NoVisual − Base`, evaluated separately for task0/1/5/6 held-out roots, which tests whether more independent contexts rescue the same no-visual formulation.

The original pooled Visual Joint remains reportable, but only with a critical implementation-confound flag.

## Audited sources

- `gnp_style_continuous.py` and `gnp_style_continuous_20260830_125107/`
- `task0_visual_context_early.py` and `task0_visual_context_early_20260831_025000/`
- `per_task_visual_context_early.py` and `taskwise_visual_context_20260831_074000/`
- `prospective_visual_context_pipeline.py`
- `/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py`
- `/home/exouser/Tabero/analysis/trajectory_physical_imagination.py`
- `/home/exouser/Tabero/analysis/counterfactual_force_world_model.py`
