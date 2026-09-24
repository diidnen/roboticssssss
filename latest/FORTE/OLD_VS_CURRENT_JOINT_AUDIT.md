# Old versus current Joint audit

## Scope and authority

This is a read-only provenance audit.  It makes no unseen-task or cross-object
claim.  The target scope is same task0, same object/task distribution, and
held-out execution roots/contexts.

Authoritative artifact roots:

- old continuous Feas/Joint: `/home/exouser/FORTE/gnp_style_continuous_20260830_125107`
- current task0 TRAIN models: `/home/exouser/FORTE/task0_visual_context_early_20260831_025000`
- current inspected task0 roots 06/07: `/home/exouser/FORTE/task0_gpu_sidecar_20260831_050050`
- current prospective source data: `/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000`

The path named in the request as
`Tabero/analysis/results/gnp_style_continuous_20260830_125107` is represented by
the complete artifact directory under FORTE above.  Its internal provenance
still points to the Tabero runner and source datasets.

## Context and sample-count audit

| quantity | old continuous | current task0 TRAIN | current inspected roots 06/07 |
|---|---:|---:|---:|
| tasks represented | 4: task0, task1, task5, task6 | 1: task0 | 1: task0 |
| simulator root IDs | 24 total; 6/task | 6 | 2 |
| independent pre-probe contexts / distinct x | 72 total; 18/task | 18 | 2 |
| force cells | 360 new continuous + 288 old coarse cells | 90 | 18 |
| outcome branches | 720 new continuous + 288 old coarse = 1008 | 180 | 90 |
| repeats for new continuous cells | 2 | 2 | 5 |
| raw recorded physical timesteps used/available for corrected auxiliary pool | 138,857 from 720 branches | 35,669 from 180 branches | evaluation only |
| constructed physical training units recorded in checkpoints | 33,383 | 8,581 | none |
| adjacent-force IE pairs | 576 | 144 | none |

The effective independent `x_context` count is 72 for the old pooled model and
18 for the current task0 model.  Branches, force cells, repeats, H8 trajectory
targets, and physical timesteps reuse the same x within a context and therefore
do not increase the independent context count.  There is also hierarchy in the
old data: each simulator root seed contributes three separately restored
friction/state contexts (LOW/MID/HIGH).  Thus 72 is the distinct-context count,
while 24 is the stricter simulator-root-cluster count.

Per old task:

| task | contexts | simulator roots | friction/state contexts per root |
|---:|---:|---:|---:|
| 0 | 18 | 6 | 3 |
| 1 | 18 | 6 | 3 |
| 5 | 18 | 6 | 3 |
| 6 | 18 | 6 | 3 |

Therefore the old task0 slice did **not** have more contexts than the current
task0 TRAIN set.  The old Joint's 72-context exposure came from pooling four
tasks.  A context-count diagnostic on that dataset cannot, by itself, cleanly
separate number of contexts from pooled task diversity.

## Collector, restore, force, and outcome semantics

Both corrected continuous datasets call the same authoritative P5-S0-C
full-task runner.  The branch controller executes the same phase sequence and
defines `full_task_success_y` as:

`lift_success == 1 AND max_basket_contact > 0.05 AND dropped == 0`.

Both use the strict last pre-probe hold (step 190), restore the same context
state before candidate-force branches, use native continuous force commands,
and prohibit outcome-driven scientific retries.  Corrected physical telemetry
uses local gripper force directly (no second rotation).  Visual capture in the
current collector is after the final pre-probe hold and before any probe/action
branch; the diagnostic feature is not fed back to the frozen VLA.

The outcome **definition** is therefore equivalent for the old corrected
continuous branches and current prospective branches.  The outcome
**supervision population is not equivalent**:

- old Feas and old Joint both receive BCE on 1,008 branches: 720 corrected
  continuous branches plus 288 earlier coarse-force branches;
- old physics/IE losses use only the 720 corrected branches;
- current task0 Base/Full/Joint receive BCE on only the 180 prospective task0
  continuous branches;
- current task0 physics/IE losses use those same 180 branches;
- force draws use the same five equal-width strata and two repeats protocol,
  but the frozen RNG seeds and the actual sampled force values differ between
  the old and prospective collections.

Thus it is false to describe the two training datasets as identical outcome
supervision with visual added.

## Visual feature and preprocessing

Old Joint has no RGB or visual feature input.  Its feasibility logit is a head
over the 64-D final Physics-GRU hidden state.

Current task0 visual extraction is the frozen pi0 PaliGemma/SigLIP image-token
path before language/action suffix decoding.  Valid tokens are mean-pooled per
camera and concatenated to a 4,096-D float32 feature.  The two cameras are
agent-view and eye-in-hand.  The feature is aligned to the same restored
pre-probe x and never changes the policy action.

PCA is fit on task0 TRAIN only.  Although 64 components were originally
specified, 18 centered contexts have rank at most 17, so the executed task0
models use 17 standardized PCA components.  Roots 06/07 only transform through
this frozen PCA; they do not participate in fitting.

## Architecture audit

Common input construction for all nonvisual backbones:

- H = 8
- step sequence: 17 dimensions
- condition vector: 54 dimensions, including the candidate force and physical
  condition used by the authoritative model

| component | old Joint | current Visual Joint |
|---|---|---|
| physics recurrent core | GRU input 71, hidden 64 | identical |
| trajectory head | 64 -> 64 -> 104 = H8 x 13 | identical |
| feasibility input | physics hidden 64 | physics hidden 64 + visual projection 16 = 80 |
| feasibility hidden | 32 | 32 |
| visual branch | none | Linear(17,16) + ReLU |
| final output | one feasibility logit | one feasibility logit |
| physics initialization | same seed-specific frozen `PHYSICS_GRU_FORCE_IE_lambda1.0` checkpoint | same checkpoint and hash |

The current Base is architecture-identical to old Feas-only.  Full Visual uses
the Base GRU(17,64), condition MLP(54,64), a visual Linear(17,16), and a fused
144 -> 64 -> 1 head.  Visual Joint therefore does not change physics hidden
width, trajectory horizon, or feasibility hidden width; it adds a 16-D visual
conditioning channel and expands only the feasibility head input from 64 to 80.

## Objective and training audit

Both Joint runs use the same native objective:

`L_physics + 1.0 * L_IE + 0.3 * L_feas`.

For both runs:

- `L_physics` is masked Smooth-L1 on normalized H8 trajectory deltas;
- `L_IE` is masked Smooth-L1 on adjacent-force predicted-delta differences;
- `L_feas` is BCEWithLogits on individual full-task outcomes;
- no event loss and no monotonic loss;
- AdamW, learning rate 0.0008, weight decay 0.0001;
- batch size 64, gradient clipping 1.0;
- 80 epochs, no early stopping, 3 seeds (0, 1, 2);
- TRAIN-only input/output normalization.

Actual differences:

- old training ran with CUDA required; the executed task0 early run ran CPU;
- old Joint exposes 1,008 feasibility branches and 720 physical branches per
  epoch population; current Joint exposes 180 for both;
- old sampling balances `(task, friction_band, source, outcome, sampling_role)`;
  current task0 sampling balances `(friction_band, outcome, stratum)`;
- old model initialization calls the seed directly; current Visual Joint calls
  `seed_everything(seed + 3000)`.  Unit batching still uses the recorded model
  seed and feasibility sampling uses `seed + 100` in both;
- actual optimizer steps differ strongly because physical-unit populations
  differ: 33,383 physical units versus 8,581, and the recorded final optimizer
  steps are correspondingly larger for old Joint;
- current Visual Joint contains extra learnable visual projection parameters
  and a wider feasibility-head input; old Joint has no such memorization path.

## Frozen evaluator audit

The old evaluator uses a 0.05 N model query grid, raw three-seed ensemble
probabilities as primary, rho = 0.80, and a frozen 9-context/27-cell/135-branch
repeated DEV benchmark.  TRAIN-only isotonic calibration is secondary and does
not select the winner.

The current task0 evaluator uses raw three-seed ensemble probabilities, rho =
0.80, the same 0.05 N dense query spacing, and the frozen roots 06/07 dataset
with 9 forces from 3.00 to 5.00 N and 5 repeats per force.  No DEV calibration
or checkpoint selection is performed.  Roots 06/07 have already been inspected
and are retrospective diagnostics from this point onward.

The evaluators share probability/frontier intent, but their empirical
populations differ: old pooled 9-context DEV versus current task0-only 2-context
held-out roots.  Consequently the published old and current metrics are not a
matched estimate of a single distribution.

## Reproduced checkpoint facts

- old Feas and current Base state-dict tensor shapes are identical;
- old and current Physics-GRU and H8 trajectory-head tensor shapes are
  identical;
- old Joint seed0 initial physics checkpoint SHA256 and current Visual Joint
  seed0 initial physics checkpoint SHA256 are both
  `e6fa4c64317c81db906d9a55ca7168a79cdf37c98b3af4cd2c2980d04f278f2a`;
- old Joint seed0 records 1,008 feasibility branches, 720 corrected physical
  branches, 576 IE pairs, and 33,383 units;
- current Visual Joint seed0 records 180 feasibility/physical branches, 144 IE
  pairs, and 8,581 units.

## Provenance cautions

The frozen old checkpoint manifest records training implementation SHA256
`cdf62ab9d12a58ec3c0a235c3c1571dc579acd7f7cd74323be30cf68875a5985`.
The current working copy of `gnp_style_continuous.py` has a later hash, so any
new subset diagnostic must explicitly record that it reuses the executed
architecture/objective/constants and frozen evaluator semantics from the
artifacts, while being a retrospective mechanism diagnostic rather than a
bitwise replay.  The old collector manifest records the final collector hash
`176a09431eff6df428401da689d4442901c2fb4b040c13607a04ceb807cb7103`.

Current task0 early training implementation SHA256 is frozen and still matches
the working file:
`af027174938106b29d081da26055b02a682c390ecaa7a817d1be4b65c6f55cd7`.

## Audit conclusion before diagnostics

The existing artifacts support the sample-complexity hypothesis as plausible,
not proven.  Old Joint saw four times as many distinct contexts and no visual
memorization channel, but it also saw pooled task diversity, 5.6 times as many
outcome branches, a different sampling balance, and a different evaluation
population.  A controlled old-data context-count curve plus task0
Joint-NoVisual is required before choosing mechanism classification A--E.
