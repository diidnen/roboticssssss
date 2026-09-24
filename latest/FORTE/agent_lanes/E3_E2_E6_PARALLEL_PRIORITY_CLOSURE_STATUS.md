# E3/E2/E6 Parallel Priority Closure

## Coordinator state

- Objective: `E3_E2_E6_CORE_CLAIMS_CLOSED`
- Protected priority: E5 fresh E2E; no interference permitted.
- Paused: Mass, Joint, Boundary, FORTE, Tabero, E7.
- Disk snapshot: `/home/exouser` has approximately 30 GiB free; warning gate 15 GiB, hard stop 10 GiB.
- GPU snapshot at final audit: NVIDIA A100-SXM4-40GB, 31% utilization, 15,446 MiB used / approximately 25.5 GiB free. This capacity remains reserved for the protected E5 campaign; no new GPU workload was allocated by the coordinator.
- E5 process audit at final check: protected E5 is active. Scheduler PID 2251991 is running `core_gpu_scheduler.py`; worker PID 2253306 is running task 6, tuple offset 04. No E5 process, config, checkpoint, scheduler, or accepted tuple was modified.
- E5 latest visible completed shard status: task1 offset04 `5/5 PASS`, task5 offset03 `5/5 PASS`, task6 offset03 `5/5 PASS`.
- Current E5 shard remains in progress: `/media/volume/newdata/exouser/activeforcing_e5_shards_20260902/ACTIVEFORCING_E5_FRESH_UTILITY_E2E_20260903_015454_task6_offset04`; its `FULL_STATUS.json` is not yet complete and the rollout CSV currently has two lines. This is observational only; do not treat it as accepted completion.

## Active lanes

| lane | agent | result directory | state |
|---|---|---|---|
| A — E3 FullTask vs LocalLift | `01a064f4-0f35-7ad0-8f98-45ac036bf38a` | `agent_lanes/E3_FULLTASK_VS_LOCALLIFT` | `SCIENTIFIC_NEGATIVE + BLOCKED` |
| B — E2 Physical Identification final | `01a064f4-0fd1-7ad0-ad44-62326aa789f6` | `agent_lanes/E2_PHYSICAL_IDENTIFICATION_FINAL` | `SCIENTIFIC_NEGATIVE` — final table/report complete |
| C — E6 Decision-aware Re-query | `01a064f4-104a-7792-9fe2-46a84f6d9295` | `agent_lanes/E6_DECISION_AWARE_REQUERY` | `SCIENTIFIC_NEGATIVE` — CPU recompute complete |

## Closure audit

- E3: task5 1–5 N TRAIN is complete (15/15) but has zero FullTask positives; task5 classifier/DEV is correctly stopped. Remaining provenance blockers are server hash mapping and missing onboarding-trained checkpoint evidence. No Isaac collection launched.
- E2: final comparison table/report completed from frozen offline evidence. Physical-history OOF is quantified; visual rows remain non-μ diagnostics; ensemble uncertainty is not calibrated; Phys2Real-style fusion is absent and explicitly marked blocked.
- E6: 144 episodes recomputed; 36/144 disagreement; frontier-harm rates are 7/108 consensus vs 7/36 disagreement (RR 3.00). Signal is descriptively positive but no qualified second-query continuation/post-query posterior exists, so no deployment claim is made.
- E5: no files, processes, checkpoint, scheduler, or campaign configuration modified by this closure. Latest visible task1/task5/task6 shards each report 5/5 PASS.
- Paused lanes remain paused: Mass, Joint, Boundary, FORTE, Tabero, E7.

## Core-claims disposition

`E3_E2_E6_CORE_CLAIMS_CLOSED` by the requested closure rule: E3 has a scientific negative plus a real resource/provenance blocker; E2 has a quantified scientific-negative final; E6 has a quantified scientific-negative final with a validated CPU signal but missing qualified second-query evidence.

## Guardrails

- No agent may kill, pause, reconfigure, retune, or relaunch E5.
- No sealed TEST supervision or outcomes may enter development.
- New Isaac collection requires a fresh disk check and coordinator resource clearance.
- Below 15 GiB free: stop new large collection; below 10 GiB: fail closed on material writes.
