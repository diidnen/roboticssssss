# Frozen table-friction sweep report

Official online `pi0_libero` on LIBERO Goal task 5 (`push the plate to the front of the stove`) was evaluated over the frozen 5-root × 3-level grid. All 13 authorized new cells ran sequentially before the 15:00 UTC deadline; two verified init-0 pilot cells were reused.

| Sliding friction | Native success | Rate (Wilson 95%) |
|---:|---:|---:|
| 0.6 | 5/5 | 100% (56.6%–100.0%) |
| 1.2 | 0/5 | 0% (0.0%–43.4%) |
| 2.0 | 0/5 | 0% (0.0%–43.4%) |

The observed ordering is monotonic non-increasing (`1.00 ≥ 0.00 ≥ 0.00`), consistent with the preregistered hypothesis. Fresh baseline roots were 4/4 successful; with reused init 0 the baseline is 5/5. Higher-friction failures all exhausted the native 310-step limit. Fresh state traces show much smaller plate planar displacement on failures than on their successful baseline counterparts; this is descriptive rather than a causal contact-force mechanism.

Validity: every new receipt identifies task 5, its intended supplied-init SHA-256, `pi0_libero`, the required checkpoint-tree digest, online requests/actions, and exact full-vector friction readback. Reused init-0 μ=2.0 is accepted because its authoritative receipt records `before=[0.6,0.005,0.0001]` and `readback=[2.0,0.005,0.0001]`. The older pilot protocol line saying `1.0` is stale and was not propagated. Reused and new cells remain RNG-state incomparable because the stock server exposes no seed API.

Claim limit: this is a five-fixed-root, one-property pilot. It supports only a controlled association between increased table sliding friction and reduced native feasibility under this frozen setup. It does not establish population reliability, force adaptation, or a causal mechanism beyond the declared intervention.
