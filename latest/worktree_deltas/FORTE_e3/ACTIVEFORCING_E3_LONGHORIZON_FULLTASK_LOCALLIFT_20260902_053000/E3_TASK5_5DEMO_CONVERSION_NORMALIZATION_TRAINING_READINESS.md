# E3 task5 five-demo conversion, normalization, and LoRA readiness

Status: **PREPARED_5_OF_5_REPLAY_GATES_PASS_DOWNSTREAM_NOT_RUN**. Assembly, conversion, normalization, training, and post-onboarding nominal DEV have not started.

The deterministic TRAIN-only set remains IDs `1,2,11,12,19` (901 steps). All five IDs independently passed the real-tactile 7dpf replay gate. This satisfies the replay-input prerequisite only; assembly and every downstream stage remain unexecuted. No TEST data, ActiveForcing outcome, force-selector target, Utility/Fmax signal, synthetic tactile, or masked tactile is admitted.

## Strict stage gates

1. `assemble_convert_validate_task5_5demo_strict.sh` requires explicit `E3_CONVERSION_GO=YES`, all five independent replay gate JSONs, exact Tabero commit `f202819...`, and absent output paths. It assembles source-normalized copies, invokes the strict 13D converter, and requires dataset QA PASS for exactly five episodes and IDs `1,2,11,12,19`.
2. `prepare_task5_5demo_norm_stats.sh` requires explicit `E3_NORM_STATS_GO=YES`, dataset QA PASS, exact OpenPI commit/config hash, and an absent asset directory. It runs on CPU and rejects normalization unless the resulting keys/dimensions are exactly `state:7`, `actions:13`, and `tactile_prefix:396`.
3. `launch_task5_5demo_lora_training.sh` requires explicit `E3_PI0_TRAINING_GO=YES`, dataset QA PASS, valid norm stats, exact OpenPI/config/transform hashes, all three frozen base-checkpoint metadata hashes, an absent experiment directory, and a fresh GPU gate of utilization below 30% with more than 28 GiB free. A coordinator GO and protected-process audit remain mandatory.

All three scripts passed shell syntax checks. Their unset-token tests exited 2 before any output creation. Earlier incomplete-state checks stopped at the first missing independent replay gate as designed. With all five gates accepted, assembly is input-ready but remains forbidden until separately authorized. The CPU config/transform preflight passed.

## Exact LoRA initialization and supervision boundary

- OpenPI isolated commit: `31049447d685cb36ddaeddda4f1d62fec0bc6392`.
- Config: `pi0_lora_tacfield_e3_task5_5demo_7dpf`.
- Base params: authoritative tactile checkpoint 49999 at `/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params`.
- Model architecture equals `pi0_lora_tacfield_tabero`; the only model-config differences are `tactile_loss_weight: 0.1 -> 0.0` and `padding_loss_weight: 1.0 -> 0.0`.
- Effective action dimension is 13 and tactile/force dimension is 6, so the loss supervises only the first seven nominal-control dimensions. Real force and marker streams remain deployment-compatible observations; force outputs are not onboarding targets.
- The real-marker input path was unit-compared to `TaberoTacFieldInputs` and was byte/array identical for every transformed field.
- Fixed training config: seed `20260902`, batch 8, two workers, 1000 steps, save/keep every 250, WandB off, overwrite off, resume off.

The runtime loader points only to checkpoint 49999. Preflight checks `_CHECKPOINT_METADATA`, `params/_METADATA`, and `params/manifest.ocdbt`; OpenPI's `_load_weights_and_validate` additionally enforces parameter-tree shape/dtype compatibility at initialization.

## Freeze and evaluation rule

If training completes, freeze and hash the terminal step-999 checkpoint, exact config, and normalization assets. Evaluate only nominal semantic FullTask success on preregistered DEV roots with robust fixed force. Do not choose among step 250/500/750/999 using DEV, force efficiency, LocalLift-vs-FullTask, ActiveForcing, or TEST outcomes. Only after nominal DEV capability is adequate may the same frozen checkpoint be shared across every later force method.

This readiness does not complete E3. IDs `1,2,11,12,19` are accepted. Assembly, normalization, LoRA training, and nominal DEV remain `NOT_RUN` and require separate explicit authorization and their own fail-closed gates.
