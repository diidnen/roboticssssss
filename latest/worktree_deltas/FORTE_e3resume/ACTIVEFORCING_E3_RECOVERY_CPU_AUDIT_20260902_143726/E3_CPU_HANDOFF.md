# E3 CPU-only recovery handoff

Recorded: `2026-09-02T14:37:26Z`

Status: `CPU_AUDIT_COMPLETE; EXECUTION_FAIL_CLOSED`.

This handoff records read-only findings only. This recovery lane did not execute or launch assembly, conversion, normalization, training, candidate serving, DEV, Isaac, or any GPU command. MASS, TEST outcomes, and Utility were not modified.

## Acceptance findings

Demo19's committed independent gate declares `DEMO19_REAL_TACTILE_7DPF_GATE_PASS`. Its live HDF5, producer-QA file, and all four media files match the hashes recorded by the gate:

- source demo: `19`, TRAIN, `libero_10/task5`
- steps/actions: `165`, `[165,13]`
- HDF5 SHA-256: `ca36d95a379be437843424ef7c8a2d5d9638d6733e1761076fb1fa9f25d19b85`
- producer-QA SHA-256: `83f5e62744d57d09520ea55f032102c85d783bfbd1940db6c121f17a1a37319f`
- independent-gate SHA-256: `7539e974592177c0861408fa14d86edee3e854de2ca74d67bd1852d797ad241d`
- real force and marker-motion checks pass; all four media streams are frame-aligned and decode `165/165`

All five gate JSONs declare PASS for frozen TRAIN IDs `[1,2,11,12,19]`. Live HDF5 and media hashes match all five gates. The producer-QA hash chain matches for IDs `2,11,12,19`, but not for ID1:

- ID1 gate-frozen producer-QA SHA-256: `1ff2851139caadf3bc46f29b3be4697613280ef0f9d132a7a64394965e73af79`
- live ID1 producer-QA SHA-256: `2f4236637035cb4ac1fbd5a06f55f6cd74f6d21daf4f65c8f779e3c97092e418`
- the live ID1 HDF5 and four media hashes still match its gate
- the producer-QA file was modified after the independent gate was recorded

Therefore the correct recovery verdict is: `5_OF_5_GATES_DECLARE_PASS; DEMO19_CHAIN_REVALIDATED; ID1_PRODUCER_QA_HASH_DRIFT_UNRESOLVED`. Do not represent the aggregate 5/5 evidence chain as fully immutable until the ID1 drift is reconciled by a new read-only provenance audit or a newly frozen independent gate.

## Observed downstream artifacts

Assembly/conversion artifacts already existed before this recovery audit. They were not launched or modified here.

- assembly manifest SHA-256: `360864ef3b26f6598845dc6ca1dfd0164cb51fc7fb37b695e29b5e7c8b95fabc`
- assembled HDF5 SHA-256: `b687ccb5b2c1d2dd06d86ec4caa9a2b6c9bf51c19a52dfa1d335fa78e5316989`; live hash matches the manifest
- selected IDs: `[1,2,11,12,19]`; total frames/samples: `901`
- dataset QA SHA-256: `bd096994027f21a787f7c7977278fc0fba20123e0698ca8647fd2b1b0435f7a2`
- dataset QA: `status=PASS`, `errors=[]`, five episodes, 901 frames; all five parquet hashes match

A normalization file also already existed and was inspected read-only:

- actual path: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_ASSETS_20260902_113000/pi0_lora_tacfield_e3_task5_5demo_7dpf/activeforcing_e3_task5_5demo_7dpf/norm_stats.json`
- SHA-256: `492a36faac4809acfe98098f67b56738f89a5600b997f1ea8d72d8bcded8ae32`
- envelope/key/dimension/finite check: PASS for `state:7`, `actions:13`, `tactile_prefix:396`, each with `mean/std/q01/q99`

The frozen scripts and prior handoff instead expect:

`/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_ASSETS_20260902_113000/activeforcing_e3_task5_5demo_7dpf/norm_stats.json`

That path does not exist. The correct OpenPI location includes both the config-name directory and the dataset-repo directory. Consequently:

- `prepare_task5_5demo_norm_stats.sh` would reject the existing asset root before it could recover, and its postcondition checks the wrong path.
- `launch_task5_5demo_lora_training.sh` checks the same wrong path and would fail closed before its GPU check or training launch.
- the candidate-lock command in the prior handoff passes the same wrong norm path.

No training checkpoint experiment directory or candidate lock existed at audit time. Training and every DEV stage are therefore `NOT_RUN/BLOCKED`.

## Frozen command interfaces — do not execute

These are recorded for downstream repair and authorization only.

Assembly interface (outputs now exist, so the no-overwrite gate must prevent reuse):

```bash
cd /home/exouser/FORTE_e3
E3_CONVERSION_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/assemble_convert_validate_task5_5demo_strict.sh
```

Normalization interface (must not be rerun against the existing asset root):

```bash
cd /home/exouser/FORTE_e3
E3_NORM_STATS_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/prepare_task5_5demo_norm_stats.sh
```

Training interface (blocked until the norm-path repair is committed/frozen and separately authorized):

```bash
cd /home/exouser/FORTE_e3
E3_PI0_TRAINING_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/launch_task5_5demo_lora_training.sh
```

Candidate lock must use the actual nested norm path shown above. Candidate serving and the five nominal DEV cells remain downstream of a completed terminal step-999 checkpoint, candidate lock, and fresh coordinator gates. Their CPU analysis interface remains:

```bash
cd /home/exouser/FORTE_e3
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/analyze_task5_post_onboarding_nominal_dev.py "${DEV_ROOT}" /media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_5DEMO_CANDIDATE_LOCK_20260902/CANDIDATE_LOCK.json
```

The DEV gate validator/analyzer also hard-code historical external-E5 PID lineage. That gate-lineage issue requires coordinator-owned engineering repair; it must not be bypassed in this recovery lane.

## CPU validation completed

- shell syntax: PASS for the assembly, norm, training, candidate-server, and DEV-cell launchers
- Python syntax: PASS for all 26 E3 Python files
- LoRA configuration/transform preflight with `JAX_PLATFORMS=cpu` and GPU visibility disabled: `E3_TASK5_5DEMO_LORA_PREFLIGHT_PASS`
- synthetic nominal-DEV analyzer/promotion unit test: `E3_POST_ONBOARDING_ANALYZER_PROMOTION_UNIT_PASS`
- post-onboarding static readiness preflight: `E3_POST_ONBOARDING_READINESS_PREFLIGHT_PASS`
- OpenPI/Tabero commits and frozen config, policy-transform, and base-checkpoint metadata hashes matched their recorded values

## Required downstream actions

1. Reconcile and refreeze the ID1 producer-QA provenance hash without changing replay data.
2. Correct the norm path in the norm postcondition, training preflight, readiness records, and candidate-lock command to include `${CONFIG}/activeforcing_e3_task5_5demo_7dpf/norm_stats.json`.
3. Add a CPU-only acceptance record for the existing norm artifact instead of overwriting or recomputing it.
4. Commit/freeze the currently dirty E3 source worktree before any execution authorization.
5. Resolve the coordinator gate-lineage hard-code before candidate serving or DEV.

Until these prerequisites are satisfied, remain idle and do not launch any downstream stage.
