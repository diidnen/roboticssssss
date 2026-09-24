# New Task-Form Data Audit

Audit time: 2026-09-13 UTC  
Task: `dump_bin_bigbin`  
Primary source: `/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments/af_dump_maxf8_20260913`

## Current coverage

| Evidence | Physical roots | Frictions / contexts | VLA policy seeds | Force/method coverage | Status |
|---|---:|---|---|---|---|
| Original root-local feasibility corpus | 200002 | TRAIN .30-.85 by .05 (12); VAL .325/.475/.625/.775 (4) | 30200002 | 8 forces: .5, 1, 2, 3, 4, 5, 6, 8 N; 128 labels | Complete, but split by friction within one root. |
| MaxF8 added full groups | 200002 | TRAIN .425/.55/.575/.70/.725/.85 (6); VAL .475/.775 (2) | 40200002 | Same 8 forces; 64 labels | Complete. |
| MaxF8 added query-only groups | 200002 | 12 TRAIN + 4 VAL settings | 30200002 | Probe/query only; no downstream labels | Complete; do not count as feasibility labels. |
| MaxF8 diagnostic comparison | 200002 | .375/.525/.675/.825 x 2 motions = 8 contexts | 50200002, 60200002 | AF, Fixed-1, Fixed-3, Fixed-5, Fixed-6, Fixed-8; 48 rollouts | Complete but development/diagnostic, not pristine TEST. |
| Stage-I motion-context study | 200002 | Separate 32-motion population | Separate study seeds | Six force branches per motion | Running; analytically separate from efficacy evaluation. |
| Nominal Frozen VLA on this task | none | none | none | none | Missing. |

The frozen maxF8 model used 192 downstream feasibility labels (128 reused + 64 added) and 35 deduplicated belief queries. All downstream labels come from one physical root. The existing diagnostic result is AF 4/8 and Fixed-8 4/8, with two AF-only and two Fixed-only successes; equal aggregates do not establish equal reliability.

## Probe, state, and pairing inventory

- Physical P4/query evidence and post-query saved states exist for all full feasibility groups and all eight diagnostic contexts.
- The original root-local collection used preforked memory states and locked the first pi0 chunk across force branches.
- The maxF8 diagnostic confirmation records fresh policy inference per branch. It preserves the same declared context/seed but is not a replay-locked common trajectory after force-induced observations diverge.
- Measured bilateral squeeze is available. Fixed-force commanded setpoints are available. AF selection curves and checkpoints are frozen.
- No native-gripper nominal rollout exists for `dump_bin_bigbin`.

## Gap against the original protocol

| Requirement from original protocol | Current new-task state | Gap |
|---|---|---|
| Feasibility split by complete physical roots, 4/1/1 | One root, split mainly by friction/query setting | Not reproduced. Retain this as a disclosed limitation of the training corpus rather than collecting more feasibility data. |
| Three friction strata represented in every split | Many frictions exist, but only on root 200002 | Coverage across friction is adequate for the retained within-root model; physical-root generalization will be tested only in the formal main evaluation. |
| Matched force branches within each context | Present for all 24 full groups on root 200002 | Complete for the retained training corpus; no additional feasibility branches are required. |
| Main evaluation roots separate from feasibility roots | The formal evaluation will intentionally reuse root 200002 | Original root-disjointness is not reproduced; test only on eight prospectively locked, unseen VLA paths. |
| Methods paired on the same physical context | Existing diagnostic methods share declared context but 502/602 are burned | Use fresh path seeds on root 200002 and lock the common post-query handoff and initial policy chunk. |
| No mandatory held-out policy-seed split | Current training data used the already locked 30200002/40200002 scheme | Keep the frozen model and do not claim cross-seed generalization. |
| Nominal Frozen VLA | Missing | Implement and pass one development smoke before full evaluation. |

## Reuse decision

Use all 192 downstream feasibility labels and all 35 deduplicated belief queries from root 200002 exactly as already split and used by the frozen `models_v4` ensemble. The checkpoint-selection records show that test labels were neither used nor accessed before lock, and the reload/runtime qualification passed.

No feasibility relabeling, added training roots, retraining, or post-hoc model selection is permitted. The resulting evidence should be described as **single-root-trained ActiveForcing evaluated on held-out VLA paths within the same physical root**, not as a reproduction of the original feasibility corpus's 4/1/1 root split or a test of root generalization.

## What is specifically missing

The remaining work is one native-gripper nominal smoke and a fresh 1-root x 8-path x 3-friction x 3-method main evaluation: 24 test contexts and 72 formal rollouts, for 73 new rollouts total. The eight formal path seeds are prospectively fixed as 80200002-80200009; a targeted pre-freeze search found no prior use of those seeds. The 502/602 diagnostics and Stage-I motions must not enter the formal success-rate denominator. The larger held-out-path test supports a meaningful same-root efficacy comparison; it does not test generalization to unseen physical roots.
