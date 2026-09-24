# Joint context sample-complexity diagnostic

## Mechanism classification

`VISUAL_JOINT_ADDS_CONTEXT_OVERFIT_BEYOND_PHYSICS_AUXILIARY` (classification B).

This is the best match to the existing evidence.  On the already inspected
task0 roots 06/07, Joint-NoVisual improves probability MAE and slightly
improves frontier MAE over Base while keeping the same under-force rate.  In
contrast, adding the same physical/IE auxiliary to Full Visual makes Visual
Joint substantially worse than Full Visual.  This supports an extra
visual-by-Joint context-overfit/sample-complexity cost.  It does **not** show
that Joint-NoVisual is a reliable controller: one of two roots is densely
nonmonotonic, and the sample is only two retrospective roots.

The old-data curve does not justify the stronger classification
`JOINT_GAIN_DEPENDS_ON_CONTEXT_DIVERSITY`.  Pooled Joint has a frontier signal
at every N, but the signal is not monotone over N=18/36/54/72.  Moreover,
task0-only N18 and pooled N18 differ sharply at identical N, so old context
count cannot be separated cleanly from pooled task diversity.

## Counts: what is and is not independent

| dataset | simulator root clusters | independent pre-probe contexts / x | force cells | outcome branches | corrected raw physical timesteps | physical units | IE pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| old pooled continuous | 24 | 72 | 360 new + 288 coarse | 1008 | 138,857 | 33,383 | 576 |
| old task0 slice | 6 | 18 | 90 new + 72 coarse | 252 | subset-derived | subset-derived | 144 |
| current prospective task0 TRAIN | 6 | 18 | 90 | 180 | 35,669 | 8,581 | 144 |
| inspected task0 roots06/07 | 2 | 2 | 18 | 90 | evaluation only | none | none |

The effective independent x counts are 72, 18, 18, and 2 respectively.  Force
branches, repeats, H8 targets, units, and timesteps are conditional observations
under an already reused x; they are not hundreds of independent contexts.

## Old Joint fixed-subset diagnostic

The pooled subsets are fixed and nested.  Context IDs are sorted within task
and interleaved in the fixed task order `[0,1,5,6]`; no population was redrawn
after seeing a result.  `TASK0_ONLY_N18` is a separate matched diagnostic.
Every setting keeps the executed old architecture, native
`Lphysics + LIE + 0.3 Lfeas`, 80 epochs, AdamW constants, seeds 0/1/2, TRAIN-only
normalization, and frozen old evaluator.

### Ensemble results on the old frozen DEV

| setting | Feas prob MAE | Joint prob MAE | Feas frontier MAE N | Joint frontier MAE N | Feas under-force | Joint under-force | Joint nonmono contexts |
|---|---:|---:|---:|---:|---:|---:|---:|
| task0-only N18 | 0.535 | 0.410 | 0.200 | 0.300 | 75.0% | 100.0% | 0/9 |
| pooled N18 | 0.289 | 0.280 | 0.364 | 0.200 | 37.5% | 62.5% | 0/9 |
| pooled N36 | 0.263 | 0.216 | 0.307 | 0.163 | 37.5% | 62.5% | 3/9 |
| pooled N54 | 0.209 | 0.202 | 0.271 | 0.188 | 37.5% | 62.5% | 3/9 |
| pooled N72 | 0.245 | 0.206 | 0.317 | 0.131 | 25.0% | 37.5% | 6/9 |

The N72 replay closely matches the authoritative checkpoints: replay
Feas/Joint frontier MAE is 0.3167/0.1313 N versus authoritative 0.325/0.125 N;
replay probability MAE is 0.2452/0.2056 versus authoritative 0.2449/0.2075.
Under-force and Joint nonmonotonicity match exactly.  This supports using the
replay for a mechanism curve despite CPU/kernel and later-working-copy
differences.

Three-seed direction is strongest at N36: all three Joint seeds improve
frontier MAE over their matched Feas seed by 0.129--0.159 N.  At N54 the
frontier direction stays favorable but ranges from 0.025 to 0.160 N, and
probability direction is inconsistent.  At N72 frontier remains favorable for
all seeds, while one seed has slightly worse probability MAE.  Under-force is
never improved by Joint in the pooled curve.

### Interpretation of the old gain

- More pooled contexts improve some localization/probability points, but the
  curve is not monotone and therefore does not establish a pure N effect.
- At fixed N=18, pooled data change the conclusion from Joint frontier worse
  than Feas (task0-only) to Joint frontier better than Feas.  Pooled task
  diversity/distribution is a material confound and likely contributor.
- Old Joint still exhibits the previously known pattern: localization signal
  without controller reliability, because safety and monotonicity degrade.
- No cross-task transfer claim is made; pooling is used only as a mechanism
  diagnostic of the historical result.

## Current Joint-NoVisual control

Joint-NoVisual receives the exact Base nonvisual inputs.  It adds the same H8
trajectory auxiliary, adjacent-force IE auxiliary, weights 1.0/1.0/0.3,
physics initialization, optimizer, 80 epochs, and three seeds as current Visual
Joint.  It has no visual projection or PCA input.  Roots06/07 are retrospective
only and cannot select a backend.

| model | prob MAE | Brier | NLL | frontier MAE N | under-force | monotonic roots | safe->unsafe reversals |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.1824 | 0.1113 | 0.4814 | 0.400 | 50% | 2/2 | 0 |
| Full Visual | 0.1590 | 0.0904 | 0.3865 | 0.300 | 100% | 2/2 | 0 |
| Visual Joint | 0.2110 | 0.1446 | 1.1700 | 0.450 | 100% | 2/2 | 0 |
| Joint-NoVisual | 0.1536 | 0.0820 | 0.3131 | 0.375 | 50% | 1/2 | 0 |

Matched differences:

- Joint-NoVisual minus Base: probability MAE -0.02885, frontier MAE -0.025 N,
  under-force +0 percentage points.
- Visual Joint minus Full Visual: probability MAE +0.05201, frontier MAE
  +0.150 N, under-force +0 percentage points.

This is the direct evidence for classification B.  The nonmonotonicity caveat
prevents a claim that the physical auxiliary is already reliable at N=18.

An independently produced retrospective three-fold task0 TRAIN root-heldout
control (two complete roots/six contexts held out per fold; frozen seed0)
corroborates the split conclusion.  Joint-NoVisual improves cell-level
probability MAE over Base (0.0665 versus 0.1210) but worsens frontier MAE
(0.236 versus 0.191 N), under-force (77.8% versus 44.4%), and context
monotonicity (61.1% versus 100%).  On the same root-CV, Visual Joint's BCE is
0.481 versus Full Visual's 0.215.  This is consistent with extra visual-Joint
overfit and with the warning that the physical auxiliary is not yet a reliable
small-N controller.

A separate current prospective pooled matched Base/Joint-NoVisual root-CV
(48 pooled training contexts per fold) improves probability MAE but does not
reliably improve task0 frontier/safety: task0 frontier is 0.244 versus 0.169 N
and under-force is 77.8% versus 55.6%.  This extra diagnostic uses a different
prospective outcome pool from the old continuous run, so it is not appended to
the old nested curve.  It nevertheless reinforces that 72 pooled contexts
alone do not fully explain the historical frontier gain; the old coarse-outcome
pool, task mix, normalization, and evaluator population remain material.

## Why Phase 3 remains justified

Existing evidence does not reject the context-sample hypothesis.  It shows
that visual conditioning adds sample complexity/overfit in the current Joint,
while old context count is confounded with pooled task diversity.  A new
same-task0, same-object, high-context prospective curve is therefore required
to answer the actual question without cross-task or cross-object extrapolation.

## Artifact locations

- fixed subset manifest and checkpoints:
  `old_joint_context_count_diagnostic_20260831/`
- Joint-NoVisual checkpoints and per-root predictions:
  `task0_joint_novisual_diagnostic_20260831/`
- top-level control summary: `TASK0_JOINT_NOVISUAL_DIAGNOSTIC.csv`
