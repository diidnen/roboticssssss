# E3 task5 five-demo to nominal-DEV execution handoff

Recorded: `2026-09-02T13:23:01Z`

Status: `FIVE_OF_FIVE_REPLAY_GATES_PASS_DOWNSTREAM_EXECUTION_NOT_RUN`.

This is an execution handoff, not an execution record. No replay, conversion, normalization, training, server, or Isaac job was started while producing it. All demonstrations are TRAIN-only. No TEST outcome, force-selector supervision, ActiveForcing outcome, Utility outcome, synthetic tactile, or masked tactile is admitted. The authoritative Utility remains unchanged at `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`; none of the commands below reads or changes Utility or Fmax.

## Current input gate

The frozen TRAIN IDs `1,2,11,12,19` have all independently passed the real-tactile 13D `7dpf` replay gate. Demo19 passed with HDF5 SHA-256 `ca36d95a379be437843424ef7c8a2d5d9638d6733e1761076fb1fa9f25d19b85`; see `TASK5_PI0_ONBOARDING_DEMO19_RESULT.md` and `TASK5_PI0_ONBOARDING_DEMO19_GATE.json`.

This makes strict assembly input-ready, but does not authorize it. Assembly, normalization, LoRA training, candidate serving, and nominal DEV remain `NOT_RUN` and require separate explicit authorization plus their existing fail-closed gates.

## Frozen implementations and identities

- E3 artifact working directory: `/home/exouser/FORTE_e3`
- Artifact directory: `/home/exouser/FORTE_e3/ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000`
- Isolated Tabero checkout: `/media/volume/newdata/exouser/Tabero_e3lh`, commit `f20281944f9771aa32c65745313a193a10665ad1`
- Isolated OpenPI checkout: `/home/exouser/FORTE/agent_lanes/Tabero_VTLA_e3_onboarding_20260902`, commit `31049447d685cb36ddaeddda4f1d62fec0bc6392`
- OpenPI config: `pi0_lora_tacfield_e3_task5_5demo_7dpf`
- OpenPI config hash: `296b01a51c985583bab688402808296315bd449ddd2aa343f4f5458ba1d1295e`
- OpenPI policy-transform hash: `f5eb0161b831c1f4a65b763f24183f5333a3efe54ae0537088a9450fdd54781a`
- Frozen base checkpoint: `/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999`

## Stage 1: strict five-demo assembly, conversion, and dataset QA

Prerequisites:

- Independent PASS gate JSONs exist for IDs `1,2,11,12,19` under the artifact directory.
- Each replay root contains exactly one validated HDF5 plus four aligned media streams.
- Tabero commit is exactly the frozen commit above.
- Neither the assembled output nor final LeRobot dataset output exists.
- The coordinator explicitly authorizes CPU conversion by setting the script's one-shot token.

Exact command:

```bash
cd /home/exouser/FORTE_e3
E3_CONVERSION_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/assemble_convert_validate_task5_5demo_strict.sh
```

Inputs:

- ID1: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_105400_RETRY1/smoke_demo1`
- IDs2/11/12/19: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_110000/demo<ID>`
- Gate JSONs: `TASK5_PI0_ONBOARDING_ID1_RETRY_GATE.json` and `TASK5_PI0_ONBOARDING_DEMO<ID>_GATE.json`

Expected outputs and pass evidence:

- Assembly root: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_ASSEMBLED_20260902_113000`
- Assembly manifest: `ASSEMBLED_5DEMO_MANIFEST.json`
- LeRobot dataset: `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_LEROBOT_7DPF_V21_20260902_120000/activeforcing_e3_task5_5demo_7dpf`
- Dataset gate: `<dataset>/E3_TASK5_5DEMO_DATASET_QA.json`, requiring `status=PASS`, `errors=[]`, exactly five episodes, IDs `[1,2,11,12,19]`, 13D actions, 7D state, force shape `[8,6]`, marker shape `[9,198,2]`, and three 224x224x3 image features.
- Terminal stdout: `STRICT_5DEMO_ASSEMBLY_CONVERSION_DATASET_GATE_PASS`.

Failure is terminal for this stage; never reuse or overwrite an existing output root.

## Stage 2: CPU normalization statistics

Prerequisites:

- Stage 1 dataset QA is PASS for exactly the frozen five IDs.
- OpenPI commit/config hash matches the frozen identity.
- The assets output root does not exist.
- This command is CPU-only by script (`JAX_PLATFORMS=cpu`).

Exact command:

```bash
cd /home/exouser/FORTE_e3
E3_NORM_STATS_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/prepare_task5_5demo_norm_stats.sh
```

Expected output and pass gate:

- `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_ASSETS_20260902_113000/activeforcing_e3_task5_5demo_7dpf/norm_stats.json`
- The JSON envelope must contain only `norm_stats`; keys must be exactly `state`, `actions`, and `tactile_prefix`; all `mean/std/q01/q99` vectors must have lengths `7/13/396` respectively.
- The script prints the SHA-256 of the accepted file. A missing or malformed file fails closed.

## Stage 3: five-demo nominal-only LoRA training

Prerequisites:

- A new explicit coordinator authorization and protected-process audit are durable before launch. The script token alone is not coordinator authorization.
- Stage 1 dataset QA and Stage 2 norm schema pass.
- Exact OpenPI commit/config/transform and all frozen base-checkpoint metadata hashes pass.
- The checkpoint experiment directory does not exist.
- Dedicated GPU gate at the script's final check: utilization `<30%` and free memory `>28672 MiB`.

Exact command:

```bash
cd /home/exouser/FORTE_e3
E3_PI0_TRAINING_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/launch_task5_5demo_lora_training.sh
```

Frozen training configuration: base checkpoint step `49999`, seed `20260902`, batch size `8`, workers `2`, `1000` steps, log interval `25`, save/keep interval `250`, WandB off, overwrite off, resume off. Only the first seven nominal-control dimensions are supervised; force/tactile output losses remain zero.

Expected experiment root:

`/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_CHECKPOINTS_20260902_113000/pi0_lora_tacfield_e3_task5_5demo_7dpf/task5_5demo_7dpf_20260902_120000`

The preregistered candidate is terminal step `999`; it is not selected by comparing DEV, Utility, ActiveForcing, or TEST outcomes. Training completion alone is not a freeze or an E3 result.

## Stage 4: candidate lock and nominal DEV gate

### 4a. CPU candidate lock

Run only after training naturally completes and terminal step `999` contains `_CHECKPOINT_METADATA`, `params/_METADATA`, and `params/manifest.ocdbt`.

```bash
cd /home/exouser/FORTE_e3
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /media/volume/newdata/exouser/tabero/Tabero-VTLA/.venv/bin/python ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/validate_and_lock_task5_onboarded_candidate.py \
  --openpi /home/exouser/FORTE/agent_lanes/Tabero_VTLA_e3_onboarding_20260902 \
  --experiment-root /media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_CHECKPOINTS_20260902_113000/pi0_lora_tacfield_e3_task5_5demo_7dpf/task5_5demo_7dpf_20260902_120000 \
  --norm-stats /media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_ASSETS_20260902_113000/activeforcing_e3_task5_5demo_7dpf/norm_stats.json \
  --output /media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_5DEMO_CANDIDATE_LOCK_20260902/CANDIDATE_LOCK.json
```

Expected gate: `CANDIDATE_LOCK.json` with `status=CANDIDATE_LOCKED_FOR_NOMINAL_DEV_NOT_FINAL_FREEZE`, `selected_step=999`, full checkpoint-tree hash, norm hash, exact code identities, `force_output_supervision_used=false`, and `test_used=false`. Existing output is never overwritten.

### 4b. Fresh locked candidate server

`SERVER_GATE` and `SERVER_RUN_ROOT` cannot be frozen now: the gate must be scheduler-issued after a final resource/duplicate check and be at most 60 seconds old, while the run root must be new. The scheduler must export absolute paths immediately before execution.

```bash
cd /home/exouser/FORTE_e3
E3_ROOT_GO=YES E3_CORE_COORDINATOR_GO=YES E3_FINAL_GATE_GO=YES \
  bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/launch_task5_onboarded_candidate_server.sh \
  "${SERVER_GATE}" "${SERVER_RUN_ROOT}"
```

The gate operation must be `E3_TASK5_CANDIDATE_SERVER_START`. The launcher rejects any active E5, duplicate server, bound port `18883`, preexisting run root, missing candidate lock, GPU utilization `>=30%`, or free memory `<=28672 MiB`. The run root receives immutable copies `COORDINATOR_FINAL_GATE_V2.json` and `CANDIDATE_LOCK.json`; the server must expose the exact candidate metadata handshake on port `18883`.

### 4c. Five one-root nominal DEV cells

Set `DEV_ROOT` once to a new absolute timestamped output root. For each root, the scheduler must independently set `CELL_GATE_<root>` to a fresh per-cell V2 gate with operation `E3_TASK5_NOMINAL_DEV_CELL` and the matching root seed. Run exactly one cell at a time after a fresh protected-process/resource/duplicate audit.

```bash
cd /home/exouser/FORTE_e3
E3_ROOT_GO=YES E3_CORE_COORDINATOR_GO=YES E3_FINAL_GATE_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/launch_task5_post_onboarding_nominal_dev_cell.sh "${CELL_GATE_7600}" "${DEV_ROOT}" 7600
E3_ROOT_GO=YES E3_CORE_COORDINATOR_GO=YES E3_FINAL_GATE_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/launch_task5_post_onboarding_nominal_dev_cell.sh "${CELL_GATE_7601}" "${DEV_ROOT}" 7601
E3_ROOT_GO=YES E3_CORE_COORDINATOR_GO=YES E3_FINAL_GATE_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/launch_task5_post_onboarding_nominal_dev_cell.sh "${CELL_GATE_7602}" "${DEV_ROOT}" 7602
E3_ROOT_GO=YES E3_CORE_COORDINATOR_GO=YES E3_FINAL_GATE_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/launch_task5_post_onboarding_nominal_dev_cell.sh "${CELL_GATE_7603}" "${DEV_ROOT}" 7603
E3_ROOT_GO=YES E3_CORE_COORDINATOR_GO=YES E3_FINAL_GATE_GO=YES bash ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/launch_task5_post_onboarding_nominal_dev_cell.sh "${CELL_GATE_7604}" "${DEV_ROOT}" 7604
```

Each exact cell is `NOMINAL_DEV_root<seed>_mu0.6_F8N`, task `libero_10/task5`, friction `0.6`, fixed robust engineering setpoint `8 N`, one experiment, and no lighting randomization. The setpoint is not Fmax, Utility, or a safety-limit claim. Each launcher rejects active E5, duplicate cell, missing server port, reused cell output, utilization `>=50%`, or free memory `<=18432 MiB`; it copies the exact gate and candidate lock into the cell before rollout.

### 4d. CPU nominal DEV analysis

Only after all five cells naturally complete:

```bash
cd /home/exouser/FORTE_e3
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/analyze_task5_post_onboarding_nominal_dev.py \
  "${DEV_ROOT}" \
  /media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_5DEMO_CANDIDATE_LOCK_20260902/CANDIDATE_LOCK.json
```

Expected outputs, created with no overwrite:

- `${DEV_ROOT}/E3_TASK5_POST_ONBOARDING_NOMINAL_DEV_GATE.json`
- `${DEV_ROOT}/E3_TASK5_POST_ONBOARDING_NOMINAL_DEV.csv`

PASS requires exactly roots `7600..7604`, one complete episode/step pair per root, distinct root-state hashes, exact candidate/gate fingerprints, empty telemetry errors, official FullTask success `>=3/5`, and query-state reach `>=4/5`. Failure triggers the preregistered nominal-only expansion to 10 demos; it never authorizes checkpoint cherry-picking or TEST/force-method inspection.

### 4e. CPU final freeze after PASS only

`FINAL_FREEZE_JSON` must be a fresh absolute output path chosen before promotion.

```bash
cd /home/exouser/FORTE_e3
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python ACTIVEFORCING_E3_LONGHORIZON_FULLTASK_LOCALLIFT_20260902_053000/promote_task5_onboarded_candidate_after_nominal_dev.py \
  /media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_5DEMO_CANDIDATE_LOCK_20260902/CANDIDATE_LOCK.json \
  "${DEV_ROOT}/E3_TASK5_POST_ONBOARDING_NOMINAL_DEV_GATE.json" \
  "${FINAL_FREEZE_JSON}"
```

Expected gate: `FINAL_FREEZE_JSON` with `status=TASK5_ONBOARDED_PI0_FROZEN_AFTER_NOMINAL_DEV_PASS`. Promotion rehashes the unchanged checkpoint and norm assets and binds the exact passing nominal-DEV gate. The resulting checkpoint/config/norm fingerprint must then be shared by every later force method.

## Critical operational blocker: stale external-E5 PID lineage

Current protected external E5 is PID `2072012`, but the frozen `validate_e3_coordinator_final_gate_v2.py`, gate template, and nominal analyzer still require historical `external_e5_pid=2059639`. This is a **high-severity execution blocker**, not a scientific-protocol change. Future server/cell commands are executable only if the scheduler creates a fresh V2 gate that the existing validator accepts and all live E5 scans are empty. If the scheduler's correct current attestation uses PID `2072012` and the validator rejects it, stop: the core coordinator must separately authorize an engineering-only gate-lineage repair and rerun CPU QA. Do not bypass, hand-edit, or weaken the validator.

## Execution state

- Five-demo replay gate: `5_OF_5_PASS`
- Five-demo assembly: `NOT_RUN_AWAIT_SEPARATE_EXPLICIT_GO`
- Norm stats: `NOT_RUN_BLOCKED_BY_DATASET_GATE`
- LoRA training: `NOT_RUN_BLOCKED_BY_DATASET_AND_NORM_GATES`
- Candidate lock/server/nominal DEV: `NOT_RUN_BLOCKED_BY_TRAINING_AND_FRESH_V2_GATE`
- Overall E3: `NOT_COMPLETE`
