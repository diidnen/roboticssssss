# Rerun required list

Priority is determined by claim dependency and reuse, not by starting new jobs immediately. The A100 was 97% utilized at audit time; no heavy job is launched by this reconciliation.

## P0 — protocol/software freeze before any locked TEST

- Add V2 manifest fields to every future run: `protocol_version`, `final_force_selector`, utility source/hash, Direct/belief hashes, candidate generator, K, posterior samples, fallback, force bounds, task/root split, and TEST-access flag.
- Implement one shared Utility selector used by E5, E6, E7, E8a, and E8b. It must expose per-candidate `p_success`, `R_succ`, `R_fail`, `U`, posterior samples, `U_bar`, and selected argmax.
- Add a preflight that rejects `MIN_RELIABLE_RHO`, `rho_sel`, or `minimum passing` when `split=LOCKED_TEST` and `protocol_version=ACTIVEFORCING_FULL_CLAIM_V2_UTILITY`.

## P1 — E5 corrected Utility

Current smoke uses minimum `p>=0.5`; it cannot become final evidence.

1. DEV-only corrected smoke using the same π0 checkpoint 49999, P4-B, Direct checkpoints, candidates, task tuple, state parity, and setpoint interface.
2. Verify the decision JSON contains expected Utility for every candidate and no runtime rho.
3. Run the matched fresh reset-to-end E5 matrix only after the smoke passes and a genuinely fresh root manifest is sealed.
4. Required arms: Fixed-Max, Success-Only Direct, Full Expected-Utility Direct, Point Utility, Posterior Utility, Query-Ignored, Active Query, and GT diagnostic as budget permits.

## P1 — E6 corrected re-query

- Reuse compatible belief checkpoints if their TRAIN-only/root-heldout provenance passes.
- For member `m`, compute `argmax_F U(F|z_m,x)`.
- Define consensus/disagreement over those force choices.
- Rerun One Query, Always Two, Raw Uncertainty, Decision-Aware Utility Consensus, and Oracle diagnostic on identical tuples.
- The existing rho sweep and threshold-induced decisions remain diagnostic only.

## P1 — E7 corrected continuous planning

- Reuse frozen Direct probability checkpoints and raw branch archive.
- Replace every `min p>=rho` decision with `argmax_F E_z[U(F|z)]`.
- Compare Frozen Grid, Dense Reference, Uniform, Stratified, and Proposal/Posterior-Aware at matched K.
- Proposal guidance must not secretly reintroduce a hard probability boundary.
- Execute selected off-grid forces exactly in simulator validation; no nearest-grid replay.

## P2 — required ablations and causal control

- Utility versus Success-Only versus Fixed-Max on identical episode tuples.
- Point versus Posterior using identical Direct checkpoint, query, candidates, utility, and tuples.
- Query-Ignored versus Active Query with identical P4-B physical transition.

## P2 — mass and joint physics

- Do not repeat completed mass P4-B development or task0 qualification.
- After qualification, train/evaluate mass belief and run Utility-based mass force selection on fresh splits.
- Run the required 3×3 friction×mass factorial with explicit joint belief and one-axis ablations. The old Joint neural architecture remains excluded.

## Recompute, not recollect

Selector-only comparisons on stored Direct probabilities and stored outcomes should be CPU recomputations. Recollect only where the selected force itself changes the physical trajectory and no exact branch exists—especially fresh E5 and off-grid E7 validation.

## Not rerun

- 720 friction branches.
- Frozen π0 checkpoint/server validation already covered, except a minimal V2 preflight.
- Mass query traces and mass task0 qualification already produced, if their manifests pass.
- Belief ensemble training when checkpoint provenance and feature contract are compatible.
