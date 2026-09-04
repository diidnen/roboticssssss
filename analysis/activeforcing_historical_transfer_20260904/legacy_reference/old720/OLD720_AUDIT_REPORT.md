# OLD720 testing-protocol audit (2026-09-04)

## Bottom line

The historical 720 archive is real and is not a two-context dataset. It contains **720 branch samples, 4 tasks, 72 friction-conditioned contexts, 24 task-specific root families, and 72 operational context tuples**. Each context has five continuous candidate-force cells and two repeats. The old data therefore has meaningful context diversity.

However, the old 720 archive is not directly final-protocol-compatible. All 72 contexts are explicitly built from the strict P4-B preprobe/probe topology. Their absolute-Newton candidate force and corrected physical telemetry are reusable, and final lift+hold can be recomputed from raw traces, but strict final no-probe admission is zero contexts.

## Provenance and grain

- Dataset: `/home/exouser/FORTE/gnp_style_continuous_20260830_125107/CONTINUOUS_TRAIN_SUCCESS_DATA.csv`; SHA-256 `80bb811e79796a9c292517ae37d2e224710a228ab8afd1e5be38ab9c2e5857c9`.
- Generation/validation: `/home/exouser/FORTE/gnp_style_continuous_collect.py` and `/home/exouser/FORTE/gnp_style_continuous.py`; collection manifest reports 720/720 valid, 575 old full-task successes and 145 failures.
- One row is one `(task, root, context, requested_force_N, repeat)` physical branch. Shape is 24 roots × 3 friction contexts × 5 force cells × 2 repeats = 720.
- The GNP posterior training population was actually 288 old coarse branches + these 720 continuous branches = 1008 feasibility outcomes. The Joint physical auxiliary used the 720 corrected branches and 576 adjacent-force pairs.

## Task breakdown

[
  {
    "task": 0,
    "samples": 180,
    "unique_contexts": 18,
    "unique_roots": 6,
    "unique_tuples": 18,
    "old_success": 138,
    "old_failure": 42,
    "recomputed_final_success": 174,
    "recomputed_final_failure": 6,
    "unique_force_candidates": 90,
    "force_min_N": 3.067545649434074,
    "force_max_N": 4.9913273990690445,
    "force_support_from_protocol_N": [
      3.0,
      5.0
    ]
  },
  {
    "task": 1,
    "samples": 180,
    "unique_contexts": 18,
    "unique_roots": 6,
    "unique_tuples": 18,
    "old_success": 121,
    "old_failure": 59,
    "recomputed_final_success": 140,
    "recomputed_final_failure": 40,
    "unique_force_candidates": 90,
    "force_min_N": 4.022641498213649,
    "force_max_N": 5.984855347684767,
    "force_support_from_protocol_N": [
      4.0,
      6.0
    ]
  },
  {
    "task": 5,
    "samples": 180,
    "unique_contexts": 18,
    "unique_roots": 6,
    "unique_tuples": 18,
    "old_success": 142,
    "old_failure": 38,
    "recomputed_final_success": 180,
    "recomputed_final_failure": 0,
    "unique_force_candidates": 90,
    "force_min_N": 3.0062501232194427,
    "force_max_N": 4.993325242802623,
    "force_support_from_protocol_N": [
      3.0,
      5.0
    ]
  },
  {
    "task": 6,
    "samples": 180,
    "unique_contexts": 18,
    "unique_roots": 6,
    "unique_tuples": 18,
    "old_success": 174,
    "old_failure": 6,
    "recomputed_final_success": 176,
    "recomputed_final_failure": 4,
    "unique_force_candidates": 90,
    "force_min_N": 3.001736578401635,
    "force_max_N": 3.9951533437943336,
    "force_support_from_protocol_N": [
      3.0,
      4.0
    ]
  }
]

## Split and evaluation recovered

The GNP checkpoint phase was TRAIN-only: 72 train contexts/24 roots; DEV and TEST were false in the checkpoint manifest. A separate nine-context, 135-branch DEV benchmark was loaded once after all six checkpoints froze. The later archive Direct evaluation is a three-fold **root-held-out grouped OOF within seen tasks** over the same 24 root families; it is not task-OOD and it is not an untouched original TEST claim. The old Direct selector used `U=p*(Fmax-F)/Fmax+(1-p)*(-1)` with lower-force tie break; the GNP continuous analysis itself used rho=0.80 threshold selection and did not reach Expected Utility.

## Label and current-mix findings

Old `full_task_success_y` is a full-task terminal success label, not the current lift+30 bilateral-hold label. Independent row-level recomputation gives 670 final-style positives and 50 negatives; the cross-tab and mismatch count are in `OLD_SUCCESS_SEMANTICS_AUDIT.json`.

The current final dataset mix is **not valid for formal training/evaluation**. It contains old 720 + new true-force rows, but `58` old rows disagree with row-wise raw-trace recomputation because `final_no_probe_prepare_dataset.py` caches a label by context and applies the first branch label to every force candidate. In addition, the current split is context-level and has sibling root-family overlap across train/dev/test, while old formal OOF was root-family-level. The 720:11 source imbalance further makes the 11 new rows only a smoke supplement, not a balanced final protocol.

## Posterior and root7703 diagnosis

The current posterior has stored test BCE/NLL `0.3478141129016876`, ECE `0.06519648499915892`, and AUROC `0.8602739726027397`. These are diagnostic only: calibration is not valid for a formal claim because the current labels/split are invalid. Monotonicity is not valid: the frozen audit reports 2/74 contexts with adjacent decreases and 772 total decreases.

Root7703 is absent from old720. It is a separate integration/debug context. The 4N trace succeeds and the 4.09N trace fails despite identical frozen arm hashes and matched handoff, but the relevant realized force means are approximately 5.902074944474941N versus 5.723216153731727N. The second 4N directory is a byte-identical duplicate, so repeatability is not established. The correct diagnosis is unresolved execution-exposure/stochasticity, not a controller regression and not enough evidence for posterior-only attribution.

## Recommendation

Do not collect more contexts yet. First repair the row-level label construction, discard or explicitly quarantine the P4-B structural context reuse for a strict final claim, reconstruct the old root-family OOF split, and retrain/evaluate with a clean seen-task/held-out-root-context protocol. Keep root7703 only as integration smoke; do not use it as the final generalization test.

The detailed JSON/CSV evidence is in this directory. The required final evaluation is **seen tasks, held-out root/context**, with no task-OOD claim.
