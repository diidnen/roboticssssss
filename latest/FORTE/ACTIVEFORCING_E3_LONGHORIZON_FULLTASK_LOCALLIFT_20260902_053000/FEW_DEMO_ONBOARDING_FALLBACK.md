# Few-demo benchmark-onboarding fallback

This path is dormant unless every existing long-horizon candidate fails frozen-π0 nominal qualification. The repository search found the project-local Tabero-VTLA LoRA training entrypoint (`scripts/train.py`) and the exact `pi0_lora_tacfield_tabero` data/model configuration in `src/openpi/training/config.py`. It did **not** find a prior local 5/10/20-demo long-horizon onboarding run, so no such historical success is claimed.

If activated:

1. Collect only TRAIN/DEV demonstrations for one task; TEST roots remain untouched.
2. Start at exactly 5 demonstrations. Convert them with the existing Tabero dataset transform so visual, tactile, force-history, language, action-normalization, and chunk semantics match the authoritative checkpoint.
3. Create a new timestamped onboarding-only OpenPI LoRA config initialized from checkpoint 49999. Do not overwrite `pi0_lora_tacfield_tabero`, its norm stats, or its checkpoint filenames.
4. Evaluate only nominal semantic task success on DEV with a robust fixed force. If inadequate, expand to 10 demonstrations, and only then to 20.
5. Never choose demo count using ActiveForcing, LocalLift-vs-FullTask, force efficiency, or final TEST performance.
6. Freeze the resulting task-onboarded checkpoint, preprocessing assets, normalization, and hash manifest.
7. Run every E3 force baseline—LocalLift, FullTask, fixed force, and ActiveForcing—with that identical frozen checkpoint.

Paper disclosure: the VLA received few-demo benchmark-task onboarding and was subsequently frozen for all force-adaptation experiments. This is not ActiveForcing training.
