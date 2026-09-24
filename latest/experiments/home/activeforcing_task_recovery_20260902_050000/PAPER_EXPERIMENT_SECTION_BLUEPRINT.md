# PAPER_EXPERIMENT_SECTION_BLUEPRINT

Status: **ACTIVEFORCING_FULL_CLAIM_EXPERIMENT_DESIGN_READY**

## 1. Method and invariance

Define frozen π0, unchanged nominal motion, unchanged low-level controller law, and setpoint-only adaptation. Introduce belief (b_\phi(z\mid D_q)) over (z=(\mu,m)), marginalized feasibility \(\bar p(F\mid D_q,x)\), the minimum-reliable rule, and separate evaluation-only `rho_env`.

## 2. Benchmark and splits

Describe T0 alphabet soup, T1 cream cheese, T5 tomato sauce, and T6 butter as the 720-row hidden-friction core: 72 contexts, 24 physical root families, five force strata, and two repeats. Distinguish grouped-root OOF from fresh locked reset-to-end E2E. Add long-horizon, mass-sensitive, joint, and secondary breadth roles only after qualification.

## 3. Query causal design

Compare NoQuery-Prior, Sham/Duration-Matched, Query-Ignored, and Active Physical Query to separate information gain from state change and timing. For E6 compare One Query, Always Two Queries, Raw Uncertainty Gate, Decision-Aware Consensus, and Oracle Requery diagnostic.

## 4. Physical identification and supervision

Compare Vision, Physical, Vision+Physical, SysID, and GT. Report friction/mass/joint identification and calibration. Compare point physics with posterior-aware marginalization. Compare Full-Task with Local-Lift only on non-degenerate stage-labeled data; report the current archive’s all-positive Local-Lift result as a negative boundary.

## 5. Main E2E force selection

Use fresh matched resets and the same Direct checkpoint, belief, candidate set, and test contexts for Expected-Utility versus Minimum-Reliable selection. Report Full SR, mean/realized force, under-force, excess-force, fallback, latency, and failure stage. Include Fixed Low, Fixed Robust, GT, and any faithfully audited external baseline.

## 6. Continuous planning

Compare Frozen Grid, Dense Deterministic Reference, Uniform Continuous, Stratified Continuous, and Proposal/Posterior-Aware Continuous at matched K. Report K, posterior sample budget, latency, selected force, Full SR, under-force, excess-force, and frontier regret. Off-grid float commands are executed exactly, never nearest-grid replayed.

## 7. Mass and joint physics

Mass is a required final experimental axis. Close M1 mass query identifiability, M2 mass-sensitive task qualification with tested low-force failure and tested reliable high-force success, M3 mass identification (Vision / Physical / Vision+Physical / SysID / GT), M4 full-task mass adaptation, and M5 fresh reset-to-end E2E. Follow with the 3×3 friction×mass factorial, reporting friction MAE, mass MAE, cross-confusion, joint force-choice accuracy, and downstream SR. The old neural Joint architecture is not used.

## 8. Limits and negative results

State which gates are complete, partial, blocked, or negative. Do not call the current 720 archive fresh E2E, exact frozen-grid, mass, joint, or semantic zero-shot evidence. If a repaired DEV hypothesis remains false, mark it `COMPLETE_NEGATIVE` and remove the unsupported claim.
