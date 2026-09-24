# Final claim-to-evidence map — Utility protocol

| Allowed claim | Required evidence | Status now | Forbidden overclaim |
|---|---|---|---|
| Frozen π0 with setpoint-only physical adaptation | E0 hashes and matched controller audit | Partial/reusable | method changes nominal motion or controller law |
| Active query contains hidden-physics information | E2 root-heldout identification plus E1 Query-Ignored control | Partial | query benefit proven without physical-state control |
| Selects a low-cost force by maximizing expected full-task Utility | E1/E5 Full Utility versus Success-Only and Fixed-Max | Rerun required | guarantees minimum feasible/reliable force |
| Marginalizes expected task Utility over physical belief | E5/E7 Point versus Posterior, identical tuples/checkpoints/candidates | Rerun required | posterior gain inferred from different data or K |
| Re-query occurs when epistemic uncertainty changes the induced Utility-optimal decision | E6 required five-arm comparison | Rerun required | threshold-crossing disagreement called final decision-aware method |
| Continuous posterior-aware planning improves the success–force tradeoff | E7 matched-K and exact off-grid simulator execution | Rerun required | nearest-grid replay called continuous execution |
| Full-task supervision captures delayed failures | E3 non-degenerate stage labels | Missing | claim from all-positive LocalLift archive |
| Shared Direct transfers across tasks | E4 task/root-heldout matched Utility evaluation | Partial development evidence | clean semantic zero-shot without supporting tasks |
| Mass-aware Utility adaptation | E8a mass belief plus fresh full-task decisions | Missing final | mass query identifiability alone proves adaptation |
| Joint friction×mass posterior adaptation | E8b 3×3 factorial and one-axis ablations | Missing | old Joint neural architecture is joint-physics evidence |

## Canonical wording

- Main selector: “ActiveForcing selects a low-cost force by maximizing expected full-task utility.”
- Posterior: “The planner marginalizes expected task utility over the inferred physical belief.”
- Re-query: “The method triggers additional physical interaction when epistemic uncertainty changes the induced utility-optimal force decision.”
- Tradeoff: “The method improves or characterizes the success–force tradeoff under a frozen utility.”
- Descriptive only: “just-enough force,” and only when empirical force results support it.

Remove or flag: “guarantees the minimum reliable force,” “absolute minimum feasible force,” “rho-selected final controller,” and any sentence that conflates an evaluation frontier with the runtime objective.

No Abstract, Contributions, or headline result may be populated from `MIN_RELIABLE_RHO` rows. Hard-rho can appear only in an ablation table with `DIAGNOSTIC_ONLY` labeling.
