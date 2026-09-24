# Historical ActiveForcing Probe Recovery

## Verdict

`PREVIOUSLY_WORKING_PROBE_RECOVERED`

The strongest previous implementation is the task0 Active Friction Imagination pilot at `/home/exouser/Tabero/analysis/results/active_friction_imagination_e2e_20260828_211434`. It is not merely code: it contains three completed 215-step P4-B trajectories, three numeric friction estimates from the exact frozen `FRICTION_GRU.pt`, three force decisions, state hashes, real downstream outcomes, and a `PASS` result. Its sensing prefix is the best provenance match to the frozen ActiveForcing method. Its historical downstream was scripted and is not reusable as formal frozen-π0 evidence.

The previous blocker was based on the now-superseded assumption that ActiveForcing had to be inserted after a π0 grasp. Historical protocol and executed artifacts instead establish a root-start physical query: reset root, approach, descend, close, hold, probe, infer physics, then decide. The sensing handoff is therefore defined. Exact original-root restore after sensing remains an evaluation-infrastructure gap.

## Search scope

The search covered readable files beneath `/home/exouser/FORTE`, `/home/exouser/Tabero`, and `/home/exouser/*`, including `analysis`, `analysis/results`, scripts, logs, run directories, checkpoints, outputs, archives, and temporary experiment directories. Content searches included P4-B/P4B/P4_B, P5-S0-C/P5S0C, P7-B, `FRICTION_GRU.pt`, probe/friction/force/frontier/rho/slip/reset/restore/snapshot/preprobe, task0 object aliases, friction values, and scientific verdict terms. Git log, grep, reflog, repository status, and accessible shell history were also checked.

The FORTE git repository contains the base code history but most experiment artifacts are untracked timestamped directories. The Tabero repository has only sparse commits/reflog relative to the experiment volume. No useful matching shell-history command was found. Consequently, executable scripts frozen into result directories, timestamped logs, numeric CSV/JSON outputs, checkpoint hashes, and protocol source hashes are the primary evidence.

## Evidence grading

- A: executable script, actual run outputs, and checkpoint/source provenance where applicable.
- B: executable script plus report/log evidence, but an incomplete runtime link for the claimed method.
- C: code only.
- D: documentation claim only.

Candidates were ranked by execution evidence and semantic match, not by prediction accuracy. The complete index is `HISTORICAL_PROBE_RUN_INDEX.csv`.

## Strongest executed runs

### 1. Active Friction Imagination E2E — selected

- Run: `/home/exouser/Tabero/analysis/results/active_friction_imagination_e2e_20260828_211434`
- Entry: `/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py`
- Task/root: task0, root seed 8100, three independently reset friction conditions.
- Actual evidence: `QUERY_RESULTS.csv`, `ACTION_RESULTS.csv`, `REAL_BRANCH_RESULTS.csv`, `PILOT_RESULT.json`, probe telemetry, worker log.
- Probe evidence: 3/3 qualified, exactly 215 steps each.
- Numeric outputs:
  - μ=0.2840209 → μ̂=0.3244166, σ=0.0482415, selected 4.5 N.
  - μ=0.5265900 → μ̂=0.5174406, σ=0.0494582, selected 3.0 N.
  - μ=0.9857358 → μ̂=0.9192193, σ=0.0538962, selected 3.0 N.
- Pilot result: `PASS`, estimated-selection real success 3/3, force match with GT-imagination 2/3.
- Important limitation: force selection used posterior-aware deterministic imagination with a 0.9 success threshold, and real execution used the scripted downstream branch from post-query state.

### 2. P5-S0-C Paired Boundary Probe Value

- Run: `/home/exouser/Tabero/analysis/results/p5s0c_paired_boundary_probe_value_20260824_000542`
- Entry: `/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py`
- Actual evidence: 144 completed and qualified root-context probes and 576 matched scripted branches.
- Provenance value: freezes P4-B source hash `a1566334...`, the exact 46-channel feature order, TRAIN-only normalization, root split, and state parity.
- Limitation: it does not load the explicit friction GRU; its model adjudication is the older Q2F family.

### 3. P4-B Task0 Contact-Conditioned Probe

- Run: `/home/exouser/Tabero/analysis/results/p4_contact_conditioned_probe_20260822_184213`
- Frozen script: `scripts/p4_collect_probe.py`.
- Actual evidence: 15 task0 trials across five seeds and μ∈{0.2,0.5,1.0}; 15/15 completed with zero task0 probe failures.
- Provenance value: this is the original root-start implementation and raw trajectory source for the later estimator.
- Limitation: the run is probe-only; the overall P4 verdict rejects cross-task generality because task2 was unsafe, but task0 itself is explicitly clean.

## Original scientific definition

The earliest executed predecessor, `/home/exouser/Tabero/analysis/results/p3_probe_belief_decision_20260819_104513`, calls the method a “pre-lift probe + discrete Bayesian force decision” and selects the minimum force whose posterior expected success is at least 0.8. It ran the physical query before downstream lift and reported 0/60 probe failures. P4-B later standardized that concept into the full root-start contact-frame sequence. This is strong provenance that the intended causal ordering was physical sensing before task execution, not π0 grasp followed by an adapter probe.

## Current non-TEST reproduction

The already-running authoritative TRAIN collector generated a fresh task0 P4-B trajectory for root 5126, μ=0.9382536. It contains exactly 215 rows. The unmodified historical `OnlineInference.sequence_array`, frozen P5-S0-C normalization, and exact `FRICTION_GRU.pt` loaded successfully and produced a 215×46 tensor, μ̂=0.9045776, σ=0.0531502.

This is a current execution check of probe sensing plus estimator only. No new Isaac process was started because the GPU already hosted the authoritative TRAIN collector and π0 feature server. No fresh force-decision or downstream result is claimed.

## What is and is not recovered

| Component | Finding |
|---|---|
| P4-B sensing procedure | Recovered and currently executing unchanged |
| 215×46 feature contract | Recovered and reproduced |
| TRAIN-only normalization | Recovered with hash |
| `FRICTION_GRU.pt` | Recovered with checkpoint hash and reproduced inference |
| Numeric friction output | Historical and fresh non-TEST evidence present |
| Historical AFI force decision | Recovered; exact rule is η=0.9 imagination, not Direct ρ=0.8 |
| Exact original-root reset after probe | Not found in the strongest historical E2E |
| Historical scripted downstream | Diagnostic only; not formal frozen-π0 evidence |
| TEST 5174–5179 | Not opened |

## Scientific-use restriction

The recovered sensing prefix can be reused unchanged. The historical downstream cannot be relabeled as frozen π0. Replacing only the downstream after an independently validated original-root restore is an evaluation-infrastructure substitution; it must not alter probe stages, features, normalization, estimator, or selected Direct rule.
