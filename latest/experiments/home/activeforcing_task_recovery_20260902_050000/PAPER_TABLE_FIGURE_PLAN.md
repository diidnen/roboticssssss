# PAPER_TABLE_FIGURE_PLAN

Status: **ACTIVEFORCING_FULL_CLAIM_EXPERIMENT_DESIGN_READY**

| Artifact | Required content | Source / gate |
|---|---|---|
| Table 1 | Frozen protocol, π0/controller invariance, archive and split summary | E0 hash manifest |
| Table 2 | Active Query vs NoQuery-Prior, Sham, Query-Ignored | E1 causal controls |
| Table 3 | Friction, mass, joint identification/calibration | E2 root-heldout |
| Table 4 | Point physics vs posterior marginalization | E2/E5 matched ablation |
| Table 5 | Full-Task vs Local-Lift supervision | E3 only if labels are non-degenerate |
| Table 6 | Shared vs task-specific feasibility transfer | E4 |
| Table 7 | Fresh reset-to-end E2E by task and physics cell | E5 locked TEST |
| Table 8 | Expected-Utility vs Minimum-Reliable selector | E5 same Direct/belief/candidates |
| Table 9 | One/Always Two/Raw Gate/Decision-Aware/Oracle Requery | E6 locked TEST |
| Table 10 | Long-horizon delayed-failure qualification and checkpoints | E4/E6 |
| Table 11 | Grid and five continuous planners at matched K | E7 exact off-grid |
| Table 12 | Mass query identifiability and estimator variants | E2/E8a |
| Table 13 | Mass adaptation and fresh E2E | E8a |
| Table 14 | 3×3 friction×mass identifiability and decisions | E8b |
| Table 15 | Shared transfer across friction, mass, and joint tasks | E4/E8b |
| Table 16 | Faithful external baseline or documented omission | Baseline fidelity audit |
| Fig. 1 | π0 → query → belief → selector → unchanged controller | E0/E1 |
| Fig. 2 | Query causal decomposition | E1 |
| Fig. 3 | Calibration/reliability diagrams | E2 |
| Fig. 4 | Point vs posterior force-choice/frontier | E2/E5 |
| Fig. 5 | Full-task/stage-aware delayed-failure timeline | E3/E6 |
| Fig. 6 | Shared-transfer matrix | E4 |
| Fig. 7 | Selector tradeoff: SR, force, under/excess, fallback | E5 |
| Fig. 8 | Re-query Pareto: reliability vs interactions/latency | E6 |
| Fig. 9 | Continuous planner regret and K/sample sensitivity | E7 |
| Fig. 10 | Commanded→measured force and certified resolution | E0/E7 appendix |
| Fig. 11 | Mass response and low/high qualification | E8a |
| Fig. 12 | 3×3 force-choice and cross-confusion | E8b |

Do not plot Local-Lift superiority from the 720 archive, empirical test frontiers as runtime inputs, or off-grid results replayed on the nearest grid point.
