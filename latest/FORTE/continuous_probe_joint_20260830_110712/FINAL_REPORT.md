# STATUS

CONTINUOUS_FEASIBILITY_NOT_YET_VALIDATED

# THREE FINAL QUESTIONS

1. **CAN WE PREDICT A REPEATED, FINE-GRAINED REAL FORCE FRONTIER? — NO, not reliably with the frozen backend gate.** The repeated real frontier itself was identifiable in 8/9 contexts, but the selected GT Feasibility-only backend failed off-grid probability MAE (0.289 > 0.15) and under-force (0.125 > 0.10).
2. **DOES JOINT HELP MORE THAN DIRECT FEASIBILITY AT UNSEEN 0.25N POINTS? — NO.** Joint's relative off-grid MAE improvement was 0.76%, below 15%, and its frontier-MAE improvement was 0.018N, below 0.10N.
3. **DOES ACTIVE PROBING PRODUCE A CLEARER CONTINUOUS CONTROL DECISION? — NOT EVALUATED.** The frozen GT backend gate failed, so Probe and strict No-Probe were not run, exactly as preregistered.

# SINGLE SCIENTIFIC GOAL

Validate a repeated, fine-grained real force frontier; compare strict-preprobe continuous Feasibility-only and Joint backends under GT friction; and, only after a GT pass, compare frozen Probe with strict No-Probe.

# CLAIM BOUNDARY

This experiment does **not** test cross-object force transfer, cross-object generalization, unseen-task generalization, or universal physics transfer. The claim is conditioned on the current object/task context: hidden friction may change its reliable minimum force, and active probing may resolve that instance-level ambiguity.

# CURRENT VLA ROLE

Frozen π0 provides task understanding, standardized staging, and nominal H8 manipulation motion. It predicts neither friction nor grip force and was not retrained or modified.

# WHY CONTINUOUS FORCE NOW

The authoritative discrete forensic showed that one-rollout 0.5N boundaries changed under corrected replay. This run therefore targets P_real(success | context, F) and defines F*_rho as the lowest tested force with at least 4 successes in 5 valid repeats (rho=0.80).

# REAL REPEATED SUCCESS CURVES

`REAL_CONTINUOUS_SUCCESS_CURVES.csv` reports every count, empirical probability, raw 5-outcome vector, and Wilson 95% interval. Collection completeness was 135/135 expected branches; scientific failures were never retried.

# REAL FINE FRONTIER

Valid repeated fine frontiers: 8/9 (88.9%). Each reported F*_rho is a 0.25N-grid estimate, not exact continuous truth. Unstable contexts remain explicitly labeled rather than isotonic-forced.

# GT CONTINUOUS FEASIBILITY-ONLY

Off-grid MAE=0.289; frontier MAE=0.125N; under-force=0.125; excess=0.094N.

# GT CONTINUOUS JOINT

Off-grid MAE=0.286; frontier MAE=0.107N; under-force=0.125; excess=0.107N.

# DOES JOINT IMPROVE CONTINUOUS INTERPOLATION?

NO. The answer follows the frozen five-part Joint gate, not a post-hoc preference.

# OFF-GRID 0.25N RESULT

Feasibility-only MAE=0.289; Joint MAE=0.286 across the nine unique preregistered midpoint queries.

# FRONTIER MAE

Selected backend `FEASIBILITY_ONLY`: 0.125N.

# UNDER-FORCE

Selected GT backend: 0.125.

# EXCESS FORCE

Selected GT backend mean excess: 0.094N.

# GT CONTINUOUS GATE

FAIL. Coverage=0.889 (threshold ≥0.80), frontier MAE=0.125N (≤0.25N), under-force=0.125 (≤0.10), off-grid MAE=0.289 (≤0.15), and selected-backend monotonicity=1.000. Probe/No-Probe was not reached, as preregistered.

# PROBE CONTINUOUS RESULT

NOT_REACHED.

# STRICT NO-PROBE CONTINUOUS RESULT

NOT_REACHED.

# DOES PROBE CHANGE THE CONTINUOUS ACTION?

NOT_EVALUATED.

# DOES PROBE CHANGE IT IN THE CORRECT DIRECTION?

NOT_EVALUATED.

# QUANTIZATION-UNMASKING

NOT_REACHED.

# JOINT VS FEASIBILITY-ONLY

| Backend | Off-grid probability MAE | Frontier MAE (N) | Under-force | Mean excess (N) | Monotonicity |
|---|---:|---:|---:|---:|---:|
| FEASIBILITY_ONLY | 0.289 | 0.125 | 0.125 | 0.094 | 1.000 |
| JOINT | 0.286 | 0.107 | 0.125 | 0.107 | 0.889 |


# PRIMARY_CLASSIFICATION

CONTINUOUS_FEASIBILITY_NOT_YET_VALIDATED

# WHAT IS NOW PROVEN

The repeated 5-rollout curves establish a usable 0.25N-grid F*₀.₈ in 8/9 current DEV object/task contexts; one context remained `REAL_FRONTIER_UNSTABLE`. They also show why single-rollout labels are inadequate near the boundary. The frozen GT comparison establishes that neither backend satisfies the preregistered continuous feasibility gate, and that Joint has no independent continuous-interpolation advantage. No Probe claim is made because the GT gate was not passed.

# WHAT IS STILL NOT PROVEN

- no cross-object claim
- no unseen-task generalization claim
- no original TEST
- no fresh roots or final E2E
- no when-to-probe agent yet
- no universal physics simulator or transfer claim

# METHOD IMPLICATION

No method promotion; retain the earliest failed gate and do not advance to TEST/E2E.

# NEXT_METHOD

repair only the earliest failed repeated-frontier, continuous-backend, or probe-estimation link. For this classification, the earliest failed link is continuous probability calibration/interpolation under GT friction; Probe estimation is not implicated by this run.

---

Protocol SHA256: `634432bb0324c7247a49e8d9f1aa44b1b0c0f078f6d569a82481ba81b4612d79`. Exact low-row tables and CSV evidence are used instead of decorative charts; stochastic counts and intervals remain authoritative.
