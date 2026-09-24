# Point-WM Seed Audit

## Direct answer

seed0: Direct 91.67%, Point-WM 88.89%, Δ = -2.78 pp
seed1: Direct 93.75%, Point-WM 95.83%, Δ = +2.08 pp
seed2: Direct 93.75%, Point-WM 95.14%, Δ = +1.39 pp

唯一 negative seed 是 **seed0**：SR 下降 **2.78 pp** （132 → 128 / 144 successes）。它仍有 88.89% SR，因此是有限下降，不是 collapse。

## Exact pooled metrics

| seed | method | SR | under-force | mean force | realized utility | GT agreement |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Direct | 91.67% | 6.94% | 4.160102 N | 0.072748 | 73.61% |
| 0 | Point-WM | 88.89% | 9.72% | 4.135210 N | 0.040577 | 52.78% |
| 1 | Direct | 93.75% | 4.86% | 4.220901 N | 0.085412 | 66.67% |
| 1 | Point-WM | 95.83% | 2.78% | 4.294250 N | 0.092702 | 52.78% |
| 2 | Direct | 93.75% | 4.86% | 4.199908 N | 0.086878 | 80.56% |
| 2 | Point-WM | 95.14% | 3.47% | 4.151070 N | 0.115006 | 50.00% |

## Ensemble definition

Point-WM 95.14% 是 **prediction/utility ensemble 后重新决策**：先对三个 seed 的每个 branch score 取均值，再在每个 `(context, repeat)` 的 frozen force candidates 中 argmax。它不是 seed metric mean。
Point-WM 的三-seed SR mean 是 93.29%，而 prediction-level ensemble SR 是 95.14%。
Direct 的三-seed SR mean 是 93.06%，而 prediction-level ensemble SR 是 93.75%。

Source: direct reconstruction from the 9 atomic-complete `fold{0,1,2}_seed{0,1,2}.npz` branch-score shards. Reconciliation max absolute error versus `NORMAL_POOLED_PROBE_WM_TABLE.csv` = 8.33e-17.
