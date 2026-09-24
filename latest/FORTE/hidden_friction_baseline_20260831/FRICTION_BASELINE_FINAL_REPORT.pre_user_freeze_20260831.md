# Hidden-Friction Baseline — Formal frozen π0 run report

## FINAL STATUS

**NOT_ESTIMABLE_PHASE0_BLOCKER_ACTIVEFORCING_PROBE_HANDOFF_UNDEFINED**. No scientific interpretation of ActiveForcing, π0-Neutral, Fixed-Max, Tabero-Oracle-Language, or the privileged GT-slip simulator is estimable. Phase A found that the frozen ActiveForcing estimator is defined on a complete 215-step scripted P4-B approach/grasp/probe sequence, while the formal experiment requires it to begin after a frozen π0-reached canonical grasp. The only runtime estimator integration also uses scripted downstream. Repairing that mismatch requires changing the scientific method, so the prescribed hard stop was applied before DEV or TEST execution.

## FROZEN PI0 DOWNSTREAM

- Component action runner: **FOUND**, but formal matched integration: **BLOCKED**.
- Task0 E2E action path: `/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py`.
- Mid-task continuation found: `/home/exouser/Tabero/analysis/p6g1r1_controller_grasp_vla_handoff.py::run_vla_full`; it does not restore upstream temporal context and is entered from fixed staging.
- Checkpoint: `/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999`.
- Prompt: `pick up the alphabet soup and place it in the basket` for π0-Neutral.
- The running instrumented server is action-producing; it is not merely a feature service.
- Exact evidence and hashes: `FROZEN_PI0_DOWNSTREAM_IMPLEMENTATION_AUDIT.md`.

## CANONICAL SNAPSHOTS

- Planned: 18.
- Valid: 0.
- Failed: 0.
- Not attempted because of the Phase-A hard stop: 18.

Zero is not interpreted as reach failure. `CANONICAL_SNAPSHOT_MANIFEST.json` and `CANONICAL_SNAPSHOT_QA.md` contain no fabricated snapshot paths or hashes.

## ROLLOUT COUNTS

- Frontier planned/completed: **810 / 0**.
- Method planned/completed: **720 / 0**.
- DEV microtests completed: **0**.
- TEST rollouts started: **no**.

## PRIMARY RESULTS

- π0-Neutral SR: **not estimable**.
- ActiveForcing-Direct SR: **not estimable**.
- Paired gain: **not estimable**.
- Per-μ breakdown: **not estimable**.

No blank metric was converted to zero.

## PRIVILEGED RESULTS

- Tabero-Oracle-Language: **not estimable**.
- Privileged Simulator (FORTE-inspired, GT-slip): **not estimable**.
- GT-Physics Oracle: **not estimable**.

The simulator surrogate is never labeled as an official FORTE reproduction.

## FRONTIER

All 162 root×μ×force cells remain planned with 0/5 completed. No existing scripted frontier row was imported or renamed. Per-μ F* and root heterogeneity are therefore not estimable.

## QA

- Frozen bundle checksum before work: pass for all 11 original files.
- Planned count audit: pass; the source manifest contains exactly 720 rows and 40 rows per each of 18 root×μ cases.
- Frontier arithmetic: pass; 6×3×9×5 = 810.
- Pre-execution root leakage audit: pass; only 5174–5179 are TEST, and the active TRAIN collector was confined to 5112–5173 at audit time.
- Scientific-facing method labels: pass; π0-Default/Tabero-Neutral map to one π0-Neutral row, and the legacy reactive slot maps to `Privileged Simulator (FORTE-inspired, GT-slip)`.
- Downstream semantic match: **not passed / blocker**.
- Snapshot pairing and replay equivalence: not run.
- JSON/CSV parse and SHA-256 checks: recorded after generation.

## SCIENTIFIC VERDICT

**No scientific verdict is permitted.** The experiment is not estimable under the current frozen interfaces. In particular, this run does not establish whether ActiveForcing helps, hurts, matches Fixed-Max, anticipates slip better than the privileged reactive surrogate, or closes the gap to GT physics.

## Required resolution before resumption

The freeze owner must supply an already-authoritative mapping for one of the following without using TEST outcomes:

1. how the frozen 215-step P4-B estimator input is constructed from a π0-reached canonical pre-probe state; or
2. a pre-existing frozen probe-only estimator/checkpoint whose training and normalization match the post-grasp query.

Choosing, padding, relabeling, or retraining that interface here would violate the current prohibition on scientific method changes. Once resolved outside this frozen TEST run, a new amendment must be hashed before any DEV microtest or TEST snapshot capture.
