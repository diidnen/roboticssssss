# Corrected locked-TEST plan

## Gate 0 — quarantine observed sets

Task0 root-scaling TEST roots 5174–5183 are observed: 10 contexts and 450 branches were collected, QA-passed, committed, and evaluated. They are not sealed for V2 Utility. All E5 smoke roots 7100 are engineering smoke, not locked TEST. Mass roots 6200/6201 are development heldout, not final locked TEST.

Reserve new, non-overlapping roots for V2. The root manifest must be written and hashed before any outcome is opened.

## Gate 1 — immutable method freeze

Freeze and hash:

1. `protocol_version=ACTIVEFORCING_FULL_CLAIM_V2_UTILITY`.
2. Utility equation, task `Fmax`, tie rule, and realized-Utility metric.
3. π0 policy config/checkpoint/norm stats.
4. Direct architecture/checkpoints/calibration/normalization.
5. Belief architecture/checkpoints/calibration and posterior sampling seed/count.
6. P4-B query and Query-Ignored implementation.
7. Re-query arms and Utility-consensus rule.
8. Candidate generator, force bounds, K, dense-reference budget, off-grid interface.
9. Fallback semantics.
10. Task list, physics levels, root IDs, repeats, sibling grouping, metrics, bootstrap seed.

Any change after this gate increments the protocol version and requires a new unobserved TEST set.

## Gate 2 — automated preflight

Reject a final run if:

- runtime config contains `rho_sel`, `MIN_RELIABLE_RHO`, `minimum_passing`, or equivalent;
- candidate selection is not Utility argmax;
- E6 disagreement is not member-wise Utility-decision disagreement;
- E7 off-grid values are rounded/replayed on a grid;
- train/dev/test roots overlap;
- any checkpoint/normalizer was fit on TEST;
- result directory lacks the V2 protocol manifest and hashes.

## Gate 3 — DEV-only engineering validation

Run the corrected Utility E5 smoke on DEV-only roots. Validate candidate-level Utility logs, state parity, query control, setpoint propagation, full-task termination, metrics, and failure taxonomy. CPU-rescore E6/E7 first; use simulator only for decisions whose physical trajectory is not already represented.

## Gate 4 — one-pass locked TEST order

1. E0 interface/invariance audit.
2. E1/E2 identification and Query-Ignored control.
3. E3/E4 only after their DEV gates are non-degenerate.
4. E5 full matched Utility E2E and required ablations.
5. E6 five-arm re-query.
6. E7 matched-K continuous exact-setpoint validation.
7. E8a mass, then E8b joint factorial after qualification gates.

Do not promote one stage by inspecting TEST from a later stage. Failed hypotheses are reported as complete negative results; TEST does not trigger tuning.

## Output isolation

- Final directory prefix: `AF_V2_UTILITY_*`.
- Legacy directory prefix: `AF_LEGACY_MIN_RELIABLE_*`.
- Each CSV row includes `protocol_version`, `selector`, `split`, `test_manifest_sha256`, and `evidence_role`.
- Aggregation scripts fail closed on mixed protocol versions or selectors.
- Paper tables read only a frozen allow-list of V2 result manifests.
