# Residual Decision Delta Report

PhysicsOnly-WM did **not** choose exactly the same force as DirectUtility: it changed 38/144 decisions (26.39%). Its aggregate SR and under-force nevertheless stayed identical because none of those changes crossed the empirical success/failure boundary: 0 success→failure and 0 failure→success. Success→success and failure→failure force substitutions changed mean force without changing either count.

Current-WM changed 44/144 decisions. It produced 4 success→failure transitions and only 1 failure→success transitions, which directly explains the lower SR and higher under-force.

## Exact counts

| method | changed | changed % | higher | lower | S→S | S→F | F→S | F→F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct-Only Residual | 36 | 25.00% | 10 | 26 | 35 | 0 | 1 | 0 |
| Current-WM Residual | 44 | 30.56% | 22 | 22 | 39 | 4 | 1 | 0 |
| PhysicsOnly-WM Residual | 38 | 26.39% | 22 | 16 | 38 | 0 | 0 | 0 |

The companion CSV contains all candidate forces, empirical outcomes/F*, base utilities, predicted corrections, corrected utilities, and selected forces for every one of the 144 controller contexts.
