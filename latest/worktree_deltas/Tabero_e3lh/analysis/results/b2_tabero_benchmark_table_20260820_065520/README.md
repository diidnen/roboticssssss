# B2 — Tabero Benchmark Qualification + Baseline Table

Timestamp dir: `b2_tabero_benchmark_table_20260820_065520`
Stage: `final`
METHOD_CHANGE = NONE

This round does **not** design OURS, train networks, modify D2, port DeliGrasp/FORTE, or run Tabero-VTLA.

Question: which official Tabero LIBERO-object tasks have hidden-friction **force decision value**, and how large is the Fixed Robust vs Oracle gap?

## Isolation

- D2 results dir not written
- D2 scripts / thresholds not edited
- D2 process not killed
- Tabero core source not modified
- B2 waited for a free Isaac GPU slot before launching Isaac

## Physics / protocol

- Env: `Isaac-Libero-Franka-Hybrid-Tactile-v0`
- Neutral official instruction only
- Arm: scripted/oracle downstream (T1/P3 constants), not π0
- Hidden physics: **friction only** μ ∈ {0.20, 0.50, 1.00}
- Phase A forces: 3, 4, 5, 6, 8 N; N=5 then expand transitions
- Primary metric: **full task success rate** (pick/lift/transport/place)

## Unrun items

- Tabero-VTLA Neutral: NOT_RUN reason: wait until positive task set is frozen; JAX/OpenPI+Isaac GPU; D2 isolation
- DeliGrasp port: NOT_RUN reason: determine positive task set first
- FORTE-inspired: NOT_RUN reason: port after DeliGrasp
- OURS / belief / probe optimization: NOT_RUN reason: out of scope
