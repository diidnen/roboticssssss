# GNP-style force-scale mapping

## Scope guard

This study does **not** claim to reproduce GNP. “GNP-style” labels an analytical design principle and a local project ablation: force cost is referenced to one predefined physical action range instead of each task’s candidate-grid maximum.

## Method mapping

| Dimension | Current ActiveForcing | GNP-style principle used here | This study’s implementation |
|---|---|---|---|
| Force support | Five context-specific candidates inside task-specific supports | One predefined physical force action range | Same original five candidates; no action-space expansion |
| Cost normalization | `F/Fmax_task`, where Fmax is the task/archive endpoint | Global action-space scale | `F/8` or `(F-3)/(8-3)` |
| Success model | Frozen Direct probability | Not changed | Identical frozen `p_D(success)` |
| Failure term | `-1` | Fixed across tasks | Preserved for B/C; E uses a fixed global `-8` only as an ablation |
| Selection | Argmax with low-force tie break | Same decision protocol | Identical argmax/tie break |

## Local provenance for the global scale

The local matched-dataset source uses a shared force support `[3,4,5,6,8]` N (`/home/exouser/Tabero/analysis/p5s0a_true_matched_dataset.py:61`). A controller handoff source executes 8 N (`/home/exouser/Tabero/analysis/p6g1r1_controller_grasp_vla_handoff.py:47`). The P7-B source also identifies grip force as the runtime action dimension (`/home/exouser/Tabero/analysis/p7b_gnp_physical_belief_force_planning.py:300`). These establish an existing project-wide action support; they do not establish a certified safety envelope.

## Analytical equivalence

With `Fref=8` and failure penalty `-8`, ablation E is:

`U_E = p(8-F) + (1-p)(-8) = 8 * U_B`.

It must therefore select exactly the same candidate as B. Its identical result is an algebraic check, not independent empirical evidence.

