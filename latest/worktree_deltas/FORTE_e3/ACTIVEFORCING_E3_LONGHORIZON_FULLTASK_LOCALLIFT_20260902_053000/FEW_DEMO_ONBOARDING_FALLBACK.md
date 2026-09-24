# Few-demo benchmark-onboarding fallback

This path is active because task5 failed its held-out-root transfer gate and all three frozen reserve candidates failed frozen-π0 nominal qualification. The repository search found the project-local Tabero-VTLA LoRA training entrypoint (`scripts/train.py`) and the exact `pi0_lora_tacfield_tabero` data/model configuration in `src/openpi/training/config.py`. It did **not** find a prior 5/10/20-demo long-horizon onboarding run, so no such historical success is claimed.

If activated:

1. Use only TRAIN/DEV demonstrations for task5; TEST roots remain untouched. The exact project-native source and first-five selection are frozen in `E3_TASK5_5DEMO_ONBOARDING_PLAN.json`.
2. Start at exactly 5 demonstrations. First replay the selected Isaac-native assembled episodes through the existing tactile `7dpf` path, then run the strict Tabero dataset transform so visual, tactile, force-history, marker, language, action-normalization, and chunk semantics match the authoritative checkpoint. Upstream 7D LIBERO data cannot be fed directly or padded with fake tactile fields.
3. Create a new timestamped onboarding-only OpenPI LoRA config initialized from checkpoint 49999. Do not overwrite `pi0_lora_tacfield_tabero`, its norm stats, or its checkpoint filenames.
4. Evaluate only nominal semantic task success on DEV with a robust fixed force. If inadequate, expand to 10 demonstrations, and only then to 20.
5. Never choose demo count using ActiveForcing, LocalLift-vs-FullTask, force efficiency, or final TEST performance.
6. Freeze the resulting task-onboarded checkpoint, preprocessing assets, normalization, and hash manifest.
7. Run every E3 force baseline—LocalLift, FullTask, fixed force, and ActiveForcing—with that identical frozen checkpoint.

Paper disclosure: the VLA received few-demo benchmark-task onboarding and was subsequently frozen for all force-adaptation experiments. This is not ActiveForcing training.

Current execution state: source acquisition and schema/lineage QA are complete; the first one-demo tactile replay smoke is preregistered but resource-gated. No replay, conversion, training, or π0 mutation has occurred yet.
