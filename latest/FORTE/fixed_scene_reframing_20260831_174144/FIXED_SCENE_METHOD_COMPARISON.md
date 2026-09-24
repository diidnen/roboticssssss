# Fixed-Scene Method Comparison

Scope: retrospective fixed-scene held-out-friction DEV labels only; no formal TEST and no simulator closed loop. The final primary metric is full-task success rate at the selected force. Training losses are diagnostics.

method,contexts,full_task_DEV_SR,fallback_rate,mean_selected_force_N,threshold
ActiveForcing-Direct,24,0.75,0.0,3.73841504729612,0.5
ActiveForcing-Joint,24,0.7083333333333334,0.0,3.711799115077755,0.5

ActiveForcing-Imagination: NOT ESTIMABLE — exact archived implementation has no common-720 fixed-scene adapter; no substitute method was introduced.

Direct and Joint use the same 0.5 decision threshold in this diagnostic. `rho_frontier=0.8` is not used for online selection.
