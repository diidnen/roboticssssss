# FINAL_EXPERIMENT_ROLLOUT_BUDGET

Status: **ACTIVEFORCING_FULL_CLAIM_EXPERIMENT_DESIGN_READY**

Counts are planned simulator branches, not completed results. A branch is one fresh reset-to-end rollout for one task/root/physics/force/repeat/method cell. Offline fitting and calibration do not multiply simulator branches.

| Block | Minimum design | New branches | Reusable evidence | Lane | Gate |
|---|---|---:|---|---|---|
| E0 | 30 commanded-force interface points plus archive audit | 30 | 720 archive rows, hashes, parity | Parallel CPU + simulator sweep | Force range/resolution certificate |
| E1 | 3 causal controls × 4 tasks × 6 roots × 3 friction × 2 repeats | 432 | 720 active-query branches if semantic parity passes | ISAAC/π0 simulator | Query-control parity |
| E2 | Formal mass-query identification | 108 | P4-B roots 6100–6105/6200–6201 as development evidence | Parallel CPU/GPU + simulator | Mass and joint identifiability gates |
| E4 | 2 candidate screens × 6 roots × 3 friction × 5 force × 2 repeats × 2 policies | 720 | Current four-task screens | ISAAC/π0 simulator | Reach, low/high contrast, delayed failure, mass sensitivity |
| E5 | 4 tasks × 6 roots × 3 friction × 5 force × 2 repeats × 6 arms | 4,320 | 720 archive rows for OOF only, not fresh E2E | ISAAC/π0 simulator | Fresh reset-to-end; frozen `rho_sel` |
| E6 | Long-task qualification/frontier + 5 query policies | 540 | Active query semantics only after parity | ISAAC/π0 simulator | DEV re-query gate and repair |
| E7 | Matched-K planners, K sensitivity, exact off-grid execution | 540 | Existing models for offline diagnosis only | CPU/GPU + simulator | Interface certificate and interpolation repair |
| E8a | 108 mass ID traces + mass force/adaptation arms | 1,188 | Existing mass hooks and query audit | CPU/GPU + ISAAC/π0 | Mass-sensitive task and low/high success |
| E8b | 6 roots × 3 friction × 3 mass × 5 force × 2 repeats × 6 arms | 3,240 | None; new factorial required | ISAAC/π0 simulator | 3×3 identifiability |
| External baseline | Fidelity audit / port candidate | 180 | FORTE logic only; no surrogate claim | CPU + ISAAC/π0 | Admit to TEST only if faithful |
| **Total minimum new** |  | **11,298** | Protected 720 friction branches plus existing mass audit |  |  |

## This-week critical path

### Parallel offline GPU

1. Freeze Direct checkpoint, point/posterior interfaces, planner budgets, and calibration code on TRAIN/DEV.
2. Diagnose the failed continuous gate; make only TRAIN/DEV repairs.
3. Train friction, mass, and joint estimators for Vision / Physical / Vision+Physical / SysID comparisons. The old Joint architecture remains excluded.
4. Select `rho_sel` on TRAIN/DEV only; freeze `rho_env` as evaluation-only and hash both in the protocol.
5. Generate offline predictions for selector, point/posterior, re-query, K, and sample-budget ablations.

### Parallel CPU

1. Build paired rollout manifests, root-heldout splits, hashes, and all metric/cluster-bootstrap code.
2. Audit the external FORTE-Reactive mapping. If slip/hardware semantics cannot be mapped faithfully, write the omission report and do not create a surrogate row.
3. Build the commanded→measured force calibration report: monotonicity, tracking error, effective resolution, dead-zone, saturation, and safe range.

### ISAAC/π0 simulator

1. Run E0 interface sweep and E1 query causal controls.
2. Screen LIBERO-10 task 7 and the book/caddy or controlled TurnAccelRoute reserve.
3. Run mass qualification on a rigid dynamic task; require tested low-force failure and tested reliable high-force success.
4. Collect fresh E5 E2E, then E6/E7 targeted branches, then E8a/E8b.
5. Execute planner-selected off-grid forces exactly; never use nearest-grid replay.

### Locked TEST

Start only after task IDs, checkpoints, thresholds, candidate sets, manifests, and controller checksums are frozen and hashed. Read TEST once. Do not use TEST frontier/outcomes to tune `rho_sel`, task membership, planner, query policy, or baseline labels.

## Duplication controls

- Reuse the 720 friction branches for archive-compatible OOF only; do not relabel them as fresh E2E or exact frozen-grid data.
- Share one query trace across estimator variants, point/posterior inference, and selectors.
- Share contexts across selector, query-policy, and planner arms; only the policy arm changes.
- Use targeted collection for long horizon, second-query evidence, off-grid validation, mass, joint physics, and fresh E2E.
- Keep all sibling branches together within a physical-root split.

The previous single-worker rate was approximately 0.035754 branches/s, so 11,298 branches is about 87.7 hours before reruns. Parallel simulator workers are required; reserve at least 25% for qualification failures, telemetry repair, and reruns.
