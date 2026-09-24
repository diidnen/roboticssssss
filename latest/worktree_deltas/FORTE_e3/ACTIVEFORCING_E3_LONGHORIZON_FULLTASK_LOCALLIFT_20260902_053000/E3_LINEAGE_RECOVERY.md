# E3 lineage recovery

Recorded: 2026-09-03 UTC.

## Recovery result

The E3 five-demo TRAIN input lineage is recoverable and passes the available
read-only checks. The accepted demos are exactly `1, 2, 11, 12, 19`; they
were not recollected. The assembled 7DPF dataset, normalization artifact,
training configuration, and frozen base-checkpoint metadata are referenced by
the strict readiness manifest. No TEST data, force outcome, or ActiveForcing
label is in the onboarding input.

## Resolved lineage issues

- `PATH_ALIAS`: the valid normalization is under
  `TASK5_ONBOARDING_7DPF_ASSETS_20260902_113000/${CONFIG}/.../norm_stats.json`;
  the launcher now resolves this exact path.
- `STALE_MANIFEST`: the five-demo manifest is the deterministic set
  `[1, 2, 11, 12, 19]`, with the independent replay gates and source IDs
  retained.
- `NORM_VERSION`: normalization SHA-256 is
  `492a36faac4809acfe98098f67b56738f89a5600b997f1ea8d72d8bcded8ae32` and
  the required dimensions are state 7, actions 13, tactile prefix 396.
- `SERVER_CHECKPOINT_MISMATCH`: the protected authoritative π0 server was
  checked against the frozen server artifact hash
  `56851e5d3fdf034eaf8bcdd2350a66187c40a1a4ed86704de85fcdf020797976`.
- No expected hash was edited to make a gate pass. The historical ID1 hash
  drift is preserved in the read-only refreeze artifact rather than silently
  replaced.

## Checkpoint status

The onboarding checkpoint
`/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_CHECKPOINTS_20260902_113000`
does not yet exist. Therefore no onboarding-trained checkpoint can be
claimed, and no π0 freeze or downstream E3 controller evaluation has started.

## Validation

- five-demo replay/input QA: PASS
- CPU config/transform/preflight: PASS
- strict LoRA preflight: `E3_TASK5_5DEMO_LORA_PREFLIGHT_PASS`
- shell syntax checks: PASS
- protected-process/GPU admission: pending; the user-started Mass workload
  remains protected and currently occupies the available GPU window.

This file records lineage recovery, not scientific E3 completion. Training,
nominal DEV capability, frozen π0 creation, matched LocalLift/FullTask models,
and real controller evaluation remain pending.
