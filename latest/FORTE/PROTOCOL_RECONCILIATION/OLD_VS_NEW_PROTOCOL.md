# Old versus authoritative protocol

| Topic | Revoked / legacy | Authoritative V2 Utility |
|---|---|---|
| Runtime selector | `min F` subject to predicted success `>=rho_sel` | `argmax_F E_z[U(F|z,x)]` |
| Runtime rho | Required/frozen | None |
| Force cost | Indirectly handled by choosing the first passing force | Explicitly encoded in `R_succ(F)` |
| Failure cost | Reliability threshold/fallback | `R_fail=-1` in expected Utility |
| Re-query disagreement | Different threshold-crossing force | Different induced Utility-optimal force |
| Posterior | Average reliability then cross rho | Marginalize Utility over belief, then argmax |
| Continuous planner | Minimum candidate crossing rho | Utility argmax at matched K |
| Proposal guidance | May locate a `p>=0.8` boundary | Must be selector-independent or Utility-guided; no hidden runtime rho |
| Empirical frontier | Could be confused with selector | Evaluation-only label for under/excess/regret |
| Main claim | Minimum reliable / minimum feasible force | Low-cost, utility-optimal, posterior-aware force selection |
| Hard-rho results | Candidate final method | Diagnostic-only ablation |

The old `MIN_RELIABLE_RHO` rule is not deleted from the record. It is retained as an ablation and must carry a legacy protocol tag. It may not define the final controller or contribute rows to the final-method aggregate.

The prior closure `activeforcing_final_closure_20260902_034923` is internally inconsistent: its Markdown says expected-utility selection, while its JSON says “minimum candidate with ensemble mean p_success >= 0.5.” Treat the selector-dependent tables from that closure as requiring provenance-level review; do not cite them as final evidence merely because the Markdown uses Utility language.
