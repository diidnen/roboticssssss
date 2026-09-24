# Gate S1 — Safety Substrate Qualification

**Verdict:** `S1_NEGATIVE_FIXED_ROBUST_FORCE_EXISTS`  
**METHOD_CHANGE:** `NONE`  
**Date:** 2026-08-18  
**Tabero:** `3bda8114c07584d3d53ae43fd555ca6606c76274` (git clean)

## Question

Does Tabero’s reproducible cream-cheese task have decision value for safe force, i.e. does uncertainty over plausible physics change the optimal force action?

Not: “does force control/sensing exist?” (already known from Gate S0).

## Answer

No. Every tested force target `{0.5, 1, 2, 4, 6, 8, 12}` N lifts cream cheese at mass `{0.05, 0.10, 0.20}` kg. Pareto-best force is always 0.5 N. VoPI on lift success is 0.

Force sensing is real and even monotonic, but **physics does not change which action should be taken**.

## Layout

| file | contents |
|---|---|
| `FINAL_VERDICT.json` | machine-readable verdict |
| `ENV_PROVENANCE.json` | commit / stack / GPU |
| `SAFETY_SEMANTICS_AUDIT.md` | S1.1 rigid vs deformable, gentleness, damage |
| `FORCE_CALIBRATION.csv` | per-trial hold/lift summaries |
| `FORCE_CALIBRATION_SUMMARY.json` | nominal-mass calibration aggregates |
| `FORCE_SWEEP_RESULTS.csv` | per-step telemetry (12600 rows) |
| `PHYSICS_SWEEP_RESULTS.csv` | same trials, mass×force matrix |
| `ACTION_DISAGREEMENT_ANALYSIS.md` | Q, F*(φ), VoPI, belief signal |
| `ANALYSIS_SUMMARY.json` | numeric digest |
| `representative_plots/` | calibration and heatmaps |
| `scripts/gate_s1_oracle_force_sweep.py` | oracle protocol (analysis-only, no Tabero source edits) |
| `logs/` | Isaac / oracle logs |

## Protocol (oracle / scripted)

Env: `Isaac-Libero-Franka-Hybrid-Tactile-v0`, `libero_object` task 1.

Phases: approach (open) → descend → close `d_pred` 0.04→0 → hold target squeeze → lift +8 cm.

Physics: runtime `set_masses` / `set_inertias` on `cream_cheese_1`. Nominal mass 0.10 kg.

## Not run

- Milk task 7: not needed (task 1 matrix complete).
- Friction variation: `NOT_RUN` — mass already answered decision value in the negative; extreme friction would risk manufacturing a positive.
- Learned policy / π0 / probing / VoPI with arbitrary weights: forbidden this gate.
- Custom damage model / `F_safe` threshold: forbidden.

## Plots

- `representative_plots/force_calibration.png`
- `representative_plots/measured_squeeze_heatmap.png`
- `representative_plots/lift_success_heatmap.png`
- `representative_plots/squeeze_by_mass.png`
