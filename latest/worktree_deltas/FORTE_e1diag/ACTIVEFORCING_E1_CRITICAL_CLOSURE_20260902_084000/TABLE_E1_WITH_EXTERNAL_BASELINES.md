# E1 with faithful external-baseline slots

The first five methods use the authoritative 144-episode, 72-context, 24-root grouped-root-OOF development archive. FORTE and Tabero are intentionally NA because a faithful same-tuple rollout is unavailable; missing values are not zero. `mean_selected_force_N` is the commanded/setpoint value. Measured trajectory force and peak force are separate columns.

| method | status | n_episodes | full_task_SR | mean_selected_force_N | mean_measured_force_N | mean_peak_measured_force_N | max_peak_measured_force_N | under_force_rate | mean_excess_force_N | mean_realized_utility | queries_per_episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi0 Default | NOT_ESTIMABLE_NO_EXACT_DEFAULT_FORCE_BRANCH | — | — | — | — | — | — | — | — | — | 0.0000 |
| Fixed-Max | ESTIMABLE_EXACT_ARCHIVED_BRANCHES | 144.0000 | 0.9861 | 4.8329 | 8.1819 | 32.6373 | 39.5956 | 0.0000 | 1.0210 | 0.0189 | 0.0000 |
| Query-Ignored | ESTIMABLE_EXACT_ARCHIVED_BRANCHES | 144.0000 | 0.9375 | 4.4667 | 7.4555 | 32.1290 | 39.3358 | 0.0486 | 0.6704 | 0.0418 | 0.0000 |
| ActiveForcing Utility | ESTIMABLE_EXACT_ARCHIVED_BRANCHES | 144.0000 | 0.9306 | 4.0628 | 7.0717 | 31.7419 | 39.3994 | 0.0556 | 0.2553 | 0.1056 | 1.0000 |
| GT-Friction Utility | ESTIMABLE_EXACT_ARCHIVED_BRANCHES | 144.0000 | 0.9514 | 4.0327 | 7.1450 | 31.7242 | 39.3994 | 0.0347 | 0.2190 | 0.1374 | — |
| FORTE | FAITHFUL_REPRODUCTION_BLOCKED | — | — | — | — | — | — | — | — | — | — |
| Tabero | FAITHFUL_REPRODUCTION_BLOCKED | — | — | — | — | — | — | — | — | — | 0.0000 |

Evaluation metric for estimable methods: `U_eval = y*(1-F/Fmax) + (1-y)*(-1)`, with task Fmax 0:5, 1:6, 5:5, 6:4 N. ActiveForcing alone uses Expected Utility at runtime. FORTE and Tabero retain their own runtime objectives.

Force caveat: the archive's `measured_force_mean_N` averages a transient trajectory that includes branch-hold loading. It is a measured physical telemetry aggregate, not controller setpoint. Failed-episode tracking MAE contains post-contact-loss zero-force tails. Cross-method reporting must not substitute commanded force where measured force is absent.
