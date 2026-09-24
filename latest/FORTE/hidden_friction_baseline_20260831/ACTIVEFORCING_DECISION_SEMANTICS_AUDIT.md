# ActiveForcing Decision Semantics Audit

## Pre-TEST amendment status

The previous ambiguity is resolved before TEST by the explicit user method
freeze: `ACTIVEFORCING_DECISION_SEMANTICS_AMBIGUOUS` is marked
`RESOLVED_BY_PRETEST_USER_FREEZE`. The historical findings below are retained
unchanged; the authoritative current contract is in
`ACTIVEFORCING_DIRECT_PRETEST_FREEZE.md` and
`ACTIVEFORCING_DIRECT_SELECTOR_CONTRACT.md`.

Audit time: 2026-08-31T12:45:17Z  
Verdict: **ACTIVEFORCING_DECISION_SEMANTICS_AMBIGUOUS**  
Gate 1: **BLOCKED**  
Formal TEST status: **TEST NOT OPENED**

## Answer

The historical record does not support silently choosing either AFI η=0.9 or Direct ρ=0.8 as the authoritative current ActiveForcing selector.

- The only recovered, actually executed end-to-end decision chain is **Active Friction Imagination (AFI)**: P4-B → exact `FRICTION_GRU.pt` → posterior hypotheses → deterministic simulator imagination → minimum force whose imagined mean success is at least **η=0.9** → real scripted downstream.
- A later method-development report explicitly pivots the proposed method to probe-conditioned **full-task feasibility**. Its frozen executable/offline controller rule is minimum force with TRAIN-calibrated `p_success >= 0.5`, not 0.8, and the report explicitly says no simulator E2E was launched.
- The 2026-08-31 Hidden-Friction implementation audit freezes `ActiveForcing-Direct` and describes a `p_success >= ρ=0.8` decision, but its own selected task0 gate says Probe was `NOT_REACHED`; no historical P4-B → exact GRU → Direct-0.8 → task-execution artifact establishes that transition.

Thus AFI has executable provenance but conflicts with the later formal method definition. Direct is the later formal method definition, but its online threshold and closed-loop execution provenance are internally unresolved. Selecting either would be a scientific method choice, not an infrastructure repair.

## Evidence chronology

| Date | Artifact | Recorded semantics | Execution strength |
|---|---|---|---|
| 2026-08-28 | `/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py` and `analysis/results/active_friction_imagination_e2e_20260828_211434/PROTOCOL.json` | `choose_from_curves(..., threshold=0.9)`; minimum imagined-success force | Complete physical probe, exact estimator, decision, and real scripted branch were executed |
| 2026-08-30 | `/home/exouser/Tabero/analysis/results/full_task_feasibility_20260830_012830/FULL_TASK_FEASIBILITY_PROTOCOL.json` | minimum tested force with TRAIN-isotonic calibrated `p_success >= 0.5` | Frozen offline feasibility protocol; explicitly forbids E2E/new simulator collection |
| 2026-08-30 | `/home/exouser/Tabero/analysis/results/active_probe_necessity_20260830_063435/FINAL_REPORT.md` | pivot to probe → friction belief → full-task feasibility → minimum sufficient force; frozen threshold 0.5 | DEV-only offline replay; no simulator E2E |
| 2026-08-31 | `FRICTION_BASELINE_IMPLEMENTATION_AUDIT.md` | `ActiveForcing-Direct`, minimum calibrated probability ≥0.8 | Formal freeze statement, but task0 gate failed and Probe was `NOT_REACHED`; no complete executable handoff |

## Why verdict A is not supported

`ACTIVEFORCING_IS_HISTORICAL_AFI_ETA09` would ignore the documented later pivot and the authoritative 2026-08-31 freeze naming Direct as the proposed method. AFI is the strongest executed ancestor, but its imagination selector was also reported as failing real-branch agreement on part of its pilot. No amendment says the later Direct freeze was withdrawn in favor of AFI.

## Why verdict B is not supported

`ACTIVEFORCING_IS_FROZEN_DIRECT_RHO08` requires explicit historical protocol plus executable provenance. The recovered Direct protocol uses 0.5, while the 0.8 rule appears only in the later baseline audit. No executed artifact closes P4-B → exact GRU → Direct-0.8 → task execution, and no versioned migration explains the 0.5-to-0.8 change.

## Frontier rho is separate

The empirical frontier definition remains unambiguous: force grid 3.00–5.00 N, five repeats per cell, and `F*0.8` is the minimum force with at least 4/5 full-task successes. That **frontier ρ=0.8** does not establish an **online method threshold ρ=0.8**. It is valid for the two quantities to differ.

## Scientific stop rule

Per the frozen instruction, decision-semantic ambiguity blocks scientific rollout. No selector was assembled from components, no DEV decision outcome was generated, and no TEST root was accessed.
