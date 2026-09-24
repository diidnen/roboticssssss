# Final Residual-Utility Forensic Report

**直白结论：没有发现 code、merge、fold 或 normalization bug。PhysicsOnly-WM 实际改变了 38 个选力，但 38 个变化全部保持原 outcome（38 个 success→success，0 个跨越成功边界），所以 SR/under-force 不变；其中 22 次升力、16 次降力使 mean force 改变。Current-WM 则造成 4 个 success→failure，只救回 1 个 failure→success，因此 SR 从 95.14% 降到 93.06%，under-force 从 3.47% 升到 5.56%。最终分类是 `MULTIPLE_FACTORS`。**

## 1. Implementation and alignment

Independent code recomputed reward, Direct utility, residual target, corrected utility, and lower-force tie-broken argmax without calling the original planner helper. All four planners matched the main implementation on 144/144 contexts. All 720 branch IDs were unique and aligned to the same task/root/context/friction/force/repeat/outcome and frozen OOF shard; grouped-root fold membership passed. Task1 retains its already-known 140/180 reconstructed-label caveat, but the 40 direct comparisons have zero mapping mismatches.

A separate correction to the experiment description: the frozen implementation uses **pointwise MSE**, not Huber. This is an objective-description mismatch in the prompt, not a runtime code bug.

## 2. Did the residual really use WM features?

Correct-vs-Zero selected-force agreement was 66.67%; Correct-vs-Shuffle agreement was 43.75%. Therefore `WM_FEATURE_NOT_MATERIALLY_USED` is not supported by the preregistered diagnostic. Output variance and controller metrics for all three paths are in `WM_FEATURE_USAGE_DIAGNOSTIC.csv`.

## 3. Predicted versus real H8 trajectory

Pooled SR: Direct 95.14%, Current predicted WM 93.06%, PhysicsOnly predicted WM 95.14%, RealTrajectory residual 95.83%. Real H8 gains only one episode over Direct, has lower realized utility (0.1335 vs 0.1374), and does not beat Direct-Only Residual on SR. That is not a clear independent trajectory gain and does not isolate World Model prediction error. The diagnostic uses exactly the same 52D summary, architecture, MSE target, folds, epochs, and seeds; it is nondeployable.

## 4. Oracle headroom

The same-branch oracle analysis upper bound reaches 98.61% SR, a 3.47 percentage-point ceiling above Direct, but it uses the scored validation outcomes and is not deployable. The stricter leave-one-repeat-out estimate never uses the scored repeat's own outcome and is reported separately; stochastic repeat disagreement prevents treating it as perfect knowledge.

## 5. Pointwise regression versus argmax

The frozen loss is branch-level MSE, whereas control depends only on within-context candidate ordering and top-1 argmax. Moreover, 18/360 force cells (5.0%) have opposite outcomes across their two repeats while Direct and both WM feature vectors are exactly identical across those repeats. Those rows impose conflicting realized-reward residual targets on the same input. The measured relationship is:

| method | residual MAE | pairwise order | top-1 force | utility regret | low-error/wrong-argmax contexts |
|---|---:|---:|---:|---:|---:|
| Current-WM Residual | 0.2094 | 0.776 | 0.465 | 0.1112 | 32 |
| Direct Utility | 0.0940 | 0.806 | 0.486 | 0.0842 | 17 |
| Direct-Only Residual | 0.1056 | 0.834 | 0.562 | 0.0667 | 19 |
| PhysicsOnly-WM Residual | 0.2368 | 0.764 | 0.486 | 0.0919 | 25 |
| RealTrajectory Residual | 0.1764 | 0.744 | 0.396 | 0.0881 | 33 |

## 6. Why Current-WM is worse

Current-WM more often creates harmful argmax flips than useful rescues. The 4 exact success→failure contexts are listed below and in the full CSV; the per-force correction signs show whether low-force utility was raised or high-force utility was suppressed.

- task1, `pv_train_t1_r03_s5103_low_mu0.252239`, repeat 1: Direct 5.7847 N (success) → Current 4.1312 N (failure); low-minus-high correction +0.4281.
- task1, `pv_train_t1_r03_s5103_low_mu0.252239`, repeat 2: Direct 5.7847 N (success) → Current 4.1312 N (failure); low-minus-high correction +0.4281.
- task1, `pv_train_t1_r04_s5104_low_mu0.223399`, repeat 1: Direct 5.8630 N (success) → Current 4.1210 N (failure); low-minus-high correction +0.2141.
- task1, `pv_train_t1_r04_s5104_low_mu0.223399`, repeat 2: Direct 5.8630 N (success) → Current 4.1210 N (failure); low-minus-high correction +0.2141.

## 7. Direct answers

1. **PhysicsOnly did not select identical forces.** It changed 38/144 decisions.
2. None of its 38 changes crossed the empirical outcome boundary: all were success→success. Mean force changed because 22 decisions moved higher and 16 moved lower.
3. Current-WM increased under-force because its corrected utility flipped more previously successful decisions into failures than it rescued.
4. WM features were materially used under Correct/Zero/Shuffle comparison.
5. Real H8 raises SR by one episode (95.83% vs 95.14%) but lowers realized utility and only matches Direct-Only Residual; this is not a clear independent H8 benefit.
6. A validation-informed perfect same-branch residual has at most 3.47 percentage points of SR headroom here; it is an analysis bound, not deployable performance.
7. The actual pointwise MSE objective is not isolated as the dominant cause: Current-WM branch regression itself is already much worse, so its bad argmax cannot be attributed only to pointwise-versus-ranking mismatch.
8. No code/merge/fold/normalization bug was found.
9. Evidence-based classification: `MULTIPLE_FACTORS` with factors RESIDUAL_LEARNING_IS_BOTTLENECK, DIRECT_ALREADY_CONTAINS_MOST_WM_INFORMATION.

## Scope

This forensic uses only the existing pooled TRAIN-root OOF population and frozen checkpoints. It does not read untouched TEST, add Probe, change the World Model, or tune any hyperparameter.
