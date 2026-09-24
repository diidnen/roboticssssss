# Protected running experiments — Agent B / E3

Last host-level audit: 2026-09-02 08:39 UTC (read-only `nvidia-smi`, `ps`, and `pgrep`).

## Protected processes

| PID | Role | Command / evidence | Action policy |
|---:|---|---|---|
| 33931 | Shared authoritative pi0 inference server | `visual_pi0_server.py --port 18881 --mode control --policy-config pi0_lora_tacfield_tabero .../49999` | PROTECTED; reuse read-only endpoint only; never kill or overwrite checkpoint |
| 555222 | MASS / joint-physics qualification | `/home/exouser/FORTE_mass/qualify_mass_structure.py --task 3 ... --roots 7042` | PROTECTED; never modify checkout/config/output/process |
| 556562 | E5 fresh Utility E2E | `run_e5_fresh_utility.py --phase full --task 1 --tuple-start 0 --max-tuples 2 ...` | PROTECTED; never modify checkout/config/output/process |
| 559267 | External experiment monitor | Reported by Agent A host audit | PROTECTED; never kill or modify |

GPU snapshot: A100 40 GB; utilization 93%; memory used 21,523 MiB; free 18,918 MiB. Per resource policy, Agent B starts no GPU/Isaac workload while utilization exceeds 90%; only CPU/offline work is allowed.

## Isolation

- FORTE: `/home/exouser/FORTE_e3`, branch `activeforcing-e3-longhorizon`, reused because it is the pre-existing isolated E3 worktree and contains only the prior E3 artifact directory as untracked work.
- Tabero: primary requested `/home/exouser/Tabero_e3lh` creation failed during checkout with `No space left on device` (root filesystem 98% full). The partial directory is untouched and is not registered as a worktree.
- Tabero alternate: `/media/volume/newdata/exouser/Tabero_e3lh`, branch `activeforcing-e3-longhorizon`, created successfully on the volume with sufficient space.
- Shared `/home/exouser/FORTE` and `/home/exouser/Tabero` are read-only evidence sources for Agent B. No shared files, checkpoints, configurations, or processes may be changed.

Before every future GPU/Isaac job Agent B must append a fresh host-level `nvidia-smi`, `ps`, and `pgrep` audit and re-evaluate the throughput gate.

## Fresh gate audit — 2026-09-02 08:46:20 UTC

- GPU: 89% utilization; 21,523 MiB used; 18,918 MiB free.
- Compute PIDs: 33931 (8,640 MiB), 555222 (6,034 MiB), 556562 (6,752 MiB).
- `ps` confirms PID 555222 MASS and PID 556562 E5 are each actively consuming CPU; PID 33931 remains the shared pi0 server.
- PID 559267 from the earlier monitor report was not present in the fresh `ps -fp` output; no signal or action was taken.
- PID 564927 (`python3 scripts/reconstruct_e1_decision_chain.py`) is Agent A's CPU/offline audit and is also PROTECTED.
- Decision: `GPU_LAUNCH_PROHIBITED` because utilization is above the strict `<70%` launch gate. Agent B remains CPU/offline only.

## Fresh gate audit — 2026-09-02 08:48:47 UTC

- GPU: 89% utilization; 21,523 MiB used; 18,918 MiB free.
- Compute PIDs remain 33931 (8,640 MiB), 555222 (6,034 MiB), and 556562 (6,752 MiB).
- MASS and E5 remain alive and CPU-active. Agent A's earlier CPU audit PID 564927 had completed by this check.
- Decision: `GPU_LAUNCH_PROHIBITED`; no E3 Isaac process started.

## Fresh gate audit — 2026-09-02 08:50:05 UTC

- GPU: 93% utilization; 21,523 MiB used; 18,918 MiB free.
- Compute PIDs remain 33931, 555222, and 556562; MASS and E5 are alive and active.
- Decision: `GPU_LAUNCH_PROHIBITED`; the newly frozen task5 support extension remains queued and no E3 Isaac process started.

## Pre-launch gate audit — 2026-09-02 08:56:59 UTC

- GPU: 36% utilization; 15,451 MiB used; 24,990 MiB free.
- MASS PID 555222 has completed; its branch table has 15 data rows. No action was taken on it.
- Protected processes still alive: shared pi0 PID 33931 (8,640 MiB) and E5 PID 556562 (6,752 MiB). E5 has 9/10 data rows and remains protected.
- Gate decision: `PASS_ONE_SMALL_E3_CELL`. Agent B may launch only its own root7400/mu0.6/F6N task5 support-extension cell. E5 progress and combined utilization must be monitored; if throughput degradation exceeds 25–30%, stop only the Agent B process and stagger.

## In-flight audit — 2026-09-02 08:58:55 UTC

- Agent B's only owned GPU process is PID 575336: isolated `run_e3_reserve_qualification_wrapper.py`, libero_10/task5, TRAIN root7400, mu=0.6, F6N.
- A new independent MASS process appeared after the launch gate: PID 575814, `/home/exouser/FORTE_mass/qualify_mass_structure.py --task 6 ...M2_STRUCTURED_TASK6_ROOT7043`. It is immediately classified `PROTECTED`; Agent B will not signal, modify, or inspect its outputs beyond read-only throughput checks.
- Shared pi0 PID 33931 remains `PROTECTED`. The earlier E5 wrapper PID 556562 was absent from `ps`; no action was taken and its outputs remain protected.
- GPU snapshot: 88% utilization, 21,312 MiB used, 19,129 MiB free. Compute allocations: pi0 8,640 MiB, Agent B 5,826 MiB, new MASS 6,748 MiB.
- Decision: continue only the already-running F6 cell while measuring episode wall time/instability. Do not launch F7/F8 without a fresh post-F6 gate. If slowdown exceeds 25–30% or instability appears, stop only PID 575336 safely and stagger.

## Post-F6 / pre-F7 gate audit — 2026-09-02 09:00:35 UTC

- F6 PID 575336 exited naturally with exit code 0. The episode file has exactly one data row, `root_state_hash=a26a845...bb6c816`, query/reset validity passed, and the analyzer reports no schema/root/unexpected-force errors.
- F6 outcome: pick=1, lift=1, transport=0, place=0, FullTask=0; 500 steps / 25.0 simulated seconds; measured mean force 0.594 N and peak 9.051 N. This is a second `LOCAL_SUCCESS_DOWNSTREAM_FAILURE`, not FullTask success and not evidence for any Fmax.
- The analyzer sees six TRAIN rows across F1..F6, common root hash, regime counts `{LOCAL_FAILURE:4, LOCAL_SUCCESS_DOWNSTREAM_FAILURE:2, FULL_TASK_SUCCESS:0}`; F7/F8 remain missing.
- Fresh `nvidia-smi`, `ps`, and bracketed `pgrep` show only shared pi0 PID 33931 on GPU. F6 and the newly appeared MASS PID 575814 both exited naturally; E5 PID 556562 is absent. No signal was sent to any process.
- GPU: 0% utilization, 8,660 MiB used, 31,781 MiB free.
- Throughput check: F6 completed in about 131 s wall time; the earlier isolated 8N qualification directory spanned about 134 s, so there is no evidence of a >25-30% slowdown or instability.
- Gate decision: `PASS_ONE_SMALL_E3_CELL`. Launch only F7 under the frozen TRAIN root7400/mu=0.6 plan. F8 requires another fresh post-F7 audit and is forbidden if F7 yields FullTask success.

## In-flight F7 audit — 2026-09-02 09:01:50 UTC

- Agent B owns only PID 578809 (root7400/mu=0.6/F7N).
- Two independent workloads appeared after the pre-F7 gate and are immediately `PROTECTED`: E5 PID 578179 (`run_e5_fresh_utility.py --phase full --task 5 ...tuple01`) and MASS PID 579378 (`qualify_mass_structure.py --task 0 ...ROOT7044`). Shared pi0 PID 33931 remains `PROTECTED`.
- GPU snapshot: 58% utilization; 28,945 MiB used; 11,496 MiB free. Allocations: pi0 8,640 MiB; E5 6,748 MiB; Agent B F7 5,858 MiB; MASS 7,560 MiB.
- Decision: finish only the already-running F7 while actively checking wall-time against F6's ~131 s baseline. A 30% ceiling is ~170 s; if F7 crosses that threshold or shows instability, stop only PID 578809 safely. No F8 may launch until all three protected workloads and the resource gate are freshly audited after F7.

## Post-F7 audit — 2026-09-02 09:03 UTC

- Agent B PID 578809 exited naturally with code 0. F7 is a valid FullTask success: pick/lift/transport/place/official all 1, 202 steps / 10.1 simulated seconds, common TRAIN root hash `a26a845...bb6c816`, mean measured force 4.671 N, peak 14.397 N.
- The frozen TRAIN analyzer reports `gate_pass=true`, 7/7 required rows through the stopping force, no errors, all query states valid, stable root hash, and regime counts 4 Local failures / 2 Local-success-downstream-failures / 1 FullTask success.
- Per the precommitted stopping rule, F8 is `NOT_RUN_STOP_RULE`; it must not be launched later.
- E5 PID 578179 and MASS PID 579378 remain independent `PROTECTED` workloads. Root observed GPU utilization returning to 87% while they were active, so the six-cell DEV confirmation is `RESOURCE_GATED` and no DEV cell has been launched.

## DEV resource monitor — 2026-09-02 09:09:03 UTC

- Fresh read-only audit: GPU 88% utilization, 23,052 MiB used, 17,389 MiB free.
- Compute processes: shared pi0 PID 33931 (8,640 MiB), protected E5 PID 578179 (6,752 MiB), and protected MASS PID 579378 (7,560 MiB).
- Decision: `DEV_LAUNCH_PROHIBITED`; no Agent B GPU process is running and no DEV cell has started.

## Pre-DEV-root7500-F4 gate — 2026-09-02 09:19:24 UTC

- Fresh `nvidia-smi`, `ps`, and `pgrep` audit: protected E5 PID 578179 exited naturally; no action was taken. Shared pi0 PID 33931 and MASS PID 579378 remain `PROTECTED`.
- GPU: 37% utilization, 16,263 MiB used, 24,179 MiB free. Allocations: pi0 8,640 MiB and MASS 7,560 MiB.
- Concurrent-work gate passes (`util <50%`, free memory >18 GB). Root coordination explicitly authorizes only the first frozen DEV cell, root7500/F4 at mu=0.6.
- Decision: `PASS_ONE_DEV_CELL`. Agent B will launch one cell only, monitor MASS throughput, then analyze schema/root/label validity before any later cell.

## DEV root7500/F4 engineering launch failure — 2026-09-02 09:19:48 UTC

- The first DEV launch exited during Isaac extension import after about 8 seconds with `AttributeError: module 'warp.types' has no attribute 'array'`.
- No cell directory, episode CSV, step CSV, policy chunk, outcome, or root hash was produced. This is `ENGINEERING_FAILURE_NO_SCIENTIFIC_OUTCOME`.
- Root cause: the new DEV launcher referenced an external Warp 1.10 staging path instead of the Isaac-bundled `omni.warp.core-1.8.2+lx64` path already validated by the TRAIN extension launcher. The stack also fell through to shared `tac_manip` after Isaac startup failed; no shared source was modified.
- Engineering-only fix: align Warp, asset variables, EULA variables, and output-directory creation exactly with the validated TRAIN launcher. Task, root, force, friction, pi0 checkpoint, controller, label semantics, and evaluator remain unchanged.
- Retry policy: rerun only the identical frozen root7500/F4 cell after a new resource audit. Do not count this startup failure as a branch or change the six-cell plan.

## Pre-retry DEV-root7500-F4 gate — 2026-09-02 09:20:44 UTC

- The earlier MASS PID 579378 exited naturally; no action was taken.
- A new independent current-task E3 workload is immediately `PROTECTED`: parent PID 586330 (`current4task_low_force_e3.py --collect-task 0`) and GPU worker PID 586370 (`current4task_low_force_e3.py --worker`, 7,560 MiB). Agent B owns neither process and will not signal or modify them.
- Shared pi0 PID 33931 remains protected. Agent B currently owns no GPU process.
- GPU: 40% utilization, 16,261 MiB used, 24,181 MiB free. With one protected Isaac worker, this meets the explicit concurrent-work condition (`util <50%`, free >18 GB).
- The prior F6/F7 cells completed in about 131 s or less alongside other protected work, providing no evidence of >25-30% throughput degradation. Retry authorization remains exactly one identical root7500/F4 cell; monitor the protected worker and stop only Agent B's PID if slowdown/instability appears.

## In-flight DEV-root7500-F4 retry — 2026-09-02 09:21:52 UTC

- Engineering fix passed Isaac startup. Agent B owns only PID 589196.
- Two more independent workloads appeared after the retry gate and are immediately `PROTECTED`: E5 PID 588147 (`run_e5_fresh_utility.py --task 6 ...tuple01`) and MASS PID 588271 (`qualify_mass_structure.py --task 7 ...ROOT7045`). Current-task worker PID 586370 and pi0 PID 33931 remain protected.
- GPU rose to 99% utilization with 35,026 MiB used and 5,415 MiB free. Per policy, Agent B will start nothing else. The running cell is monitored against the ~131 s observed wall baseline and ~170 s 30%-slowdown ceiling; on excess delay or instability, stop only PID 589196 safely.

## Post-DEV-root7500-F4 / pre-F6 audit — 2026-09-02 09:24:52 UTC

- Owned PID 589196 exited naturally with code 0 and finalized telemetry at 09:23:25. Startup began at 09:21:16, for about 129 s wall time; this did not exceed the ~170 s ceiling. No signal was sent.
- Valid F4 result: root hash `a7e8dec7e85187d880d50d67e1040c74e7e825e2b28ed8d60ac55135db7f51a9`, query reached, pick=1, lift=0, FullTask=0, regime `LOCAL_FAILURE`, 500 steps.
- Incremental analyzer initially exposed two engineering-only bugs on the one-cell partial dataset (missing-root `defaultdict` mutation and lowercase JSON booleans). Both were fixed without changing data, labels, gate, or plan. Final analyzer reports zero data errors and the expected five missing fixed cells.
- Prior protected workers exited naturally. New independent protected workers: MASS PID 590632 (`task7 ...ROOT7045_RERUN`) and E5 PID 592443 (`task6 ...tuple01`), plus shared pi0 PID 33931.
- GPU: 90% utilization, 22,269 MiB used, 18,172 MiB free. Decision: `DEV_F6_LAUNCH_PROHIBITED`; despite F4 validity, no next Agent B process may start until a fresh gate passes.

## Pre-DEV-root7500-F6 gate — 2026-09-02 09:42:53 UTC

- Protected E5 PID 592443 exited naturally; no action was taken. Shared pi0 PID 33931 and MASS PID 590632 remain protected.
- GPU: 46% utilization, 15,483 MiB used, 24,958 MiB free. This meets the explicit concurrent-work gate (`util <50%`, free >18 GB).
- Incremental DEV analyzer confirms root7500/F4 has query validity, correct force/friction telemetry, root hash `a7e8dec7...f51a9`, no schema errors, and `LOCAL_FAILURE` label. Exactly five frozen cells remain.
- Root coordination authorizes only root7500/F6. Decision: `PASS_ONE_DEV_CELL`; monitor MASS throughput and analyze/hash-check before F7.

## DEV-root7500-F6 launcher race — 2026-09-02 09:43 UTC

- Agent B's returned launcher invocation exited 4 because the F6 cell directory appeared concurrently. Read-only inspection found exactly one F6 launcher PID 598822 and one child PID 598825 running the exact frozen isolated command; there is no duplicate F6 process.
- Root confirmed it did not launch PID 598822/598825. Because process ownership is not unambiguous, both are classified `PROTECTED_PENDING_OWNERSHIP`; Agent B will not signal them.
- The cell will be accepted only if it exits naturally and the standard analyzer passes schema, query, force/friction, and same-root hash checks. The exit4 race is engineering-only and does not count as an outcome or authorize a plan change.

## Post-DEV-root7500-F6 / pre-F7 gate — 2026-09-02 09:46:01 UTC

- F6 PIDs 598822/598825 exited naturally and produced one episode plus step/raw telemetry. The analyzer accepted the cell with zero errors.
- Valid F6 result: same root7500 hash `a7e8dec7...f51a9`, query reached, force/friction telemetry 6N/0.6, pick=1, lift=0, FullTask=0, regime `LOCAL_FAILURE`, 500 steps.
- Fresh audit: GPU 35% utilization, 15,483 MiB used, 24,958 MiB free. Protected pi0 PID 33931 and MASS PID 590632 remain; no Agent B process runs.
- Numeric concurrent gate passes; root coordination authorizes exactly root7500/F7. No root7501 cell may start before F7 completion, analyzer, and another fresh gate.

## F7 already-running detection — 2026-09-02 09:46:37 UTC

- Before Agent B invoked any F7 launcher in this step, the immediate audit found launcher PID 600712 / child PID 600715 already running the exact isolated root7500/F7 command.
- Ownership is not unambiguous, so both PIDs are `PROTECTED_PENDING_OWNERSHIP`. Agent B did not launch a duplicate and will not signal them.
- GPU at detection: 89% utilization, 21,347 MiB used, 19,094 MiB free. Protected pi0 PID 33931 and MASS PID 590632 also remain active.
- Accept F7 only after natural exit and full analyzer/root-hash QA. Start no root7501 cell meanwhile.

## Task5 DEV gate failure and partial root7501 evidence — 2026-09-02 09:51 UTC

- Root7500/F7 PIDs 600712/600715 exited naturally and finalized telemetry at 09:48:22 (~142 s wall, below the ~170 s ceiling). Analyzer: same root7500 hash, query valid, correct 7N/0.6 telemetry, lift=0, FullTask=0, `LOCAL_FAILURE`.
- Root7500 fixed anchors F4/F6/F7 are all `LOCAL_FAILURE`; therefore the frozen requirement of all three regimes independently per DEV root is mathematically failed. No adaptive force insertion is allowed.
- An exact frozen root7501/F4 process was already running as PIDs 602265/602268 before Agent B launched a new cell; ownership remained ambiguous, so it was protected and allowed to exit naturally. It yielded a valid distinct root hash `220bd2ff...0f2b65`, query=1, lift=0, FullTask=0, `LOCAL_FAILURE`.
- Root7501/F6 and F7 are `NOT_RUN_DUE_FROZEN_GATE_IMPOSSIBLE_AND_THROUGHPUT_CONSERVATION`.
- New independent MASS PID 603571 (`task0 ...ALT_ROOT7046`) and E5 PID 600247 (`task1 tuple02`) are `PROTECTED`. Root reported GPU ~93% with both active; reserve qualification launch is prohibited until a fresh gate.

## Pre-reserve-task2 nominal gate — 2026-09-02 09:59:17 UTC

- Protected E5 PID 600247 exited naturally; no action was taken. Shared pi0 PID 33931 and MASS PID 603571 remain protected.
- GPU: 27% utilization, 16,262 MiB used, 24,179 MiB free. This passes the explicit concurrent-work gate (`util <50%`, free >18 GB).
- Read-only `ps`/process search found no existing reserve task2 qualification or conflicting isolated E3 output cell.
- Per the pre-frozen reserve order and root coordination, authorize exactly one nominal-only `libero_10/task2` robust 8N episode at TRAIN root7600, mu=0.6. This is not a Utility/Fmax/TEST experiment. No task8/g3 job may start until task2 exits and is analyzed.

## Concurrent pending-ownership E3 processes — 2026-09-02 10:00:34 UTC

- Exact reserve task2 process PID 607692 (parent launcher 607689) is running the authorized nominal-only 8N command. The unified Agent B session initiated the task2 command, but due the repeated launcher/orchestrator race pattern root requires `PROTECTED_PENDING_OWNERSHIP`; no signal will be sent.
- An exact root7501/F6 task5 process PID 607586 (parent launcher 607583) also appeared although root7501/F6 had been stopped after the task5 gate became impossible. Root did not launch it and Agent B did not invoke it after the stop. It is `PROTECTED_PENDING_OWNERSHIP`; no signal or duplicate.
- If root7501/F6 exits naturally with valid telemetry, preserve it as `POST_GATE_PRESTARTED_EXTRA_EVIDENCE_EXCLUDED_FROM_CONFIRMATORY_GATE`. It cannot rescue the already-failed root7500 per-root criterion. Root7501/F7 remains prohibited.
- Protected MASS PID 603571 and pi0 PID 33931 remain active. GPU: 88% utilization, 26,042 MiB used, 14,399 MiB free. No further launch of any kind is permitted until all running E3 cells are analyzed and a fresh gate passes.

## Post-concurrency validation — 2026-09-02 10:03 UTC

- Reserve task2 PID 607692 exited with code 137 during concurrent load. Its timestamped `t2` directory contains no episode, step, action-chunk, or root-hash artifact. This attempt is `RESOURCE_ABORT_NO_SCIENTIFIC_OUTCOME`; the empty directory is preserved and must never be reused.
- Root7501/F6 PIDs 607583/607586 exited naturally. The isolated analyzer accepted one episode/step pair with root hash `220bd2ff0b9aa6b50034b23844b2c0a12f7dea6dfa02eb4b3477b351620f2b65`, query reached, pick=1, lift=0, FullTask=0, 500 steps, and `LOCAL_FAILURE`. Episode CSV SHA-256 is `2c27e5ed982fe73010b226085edf5cd77154aac47e667bb5b90e82c7a565a648`; step CSV SHA-256 is `6a1e82c6c1649bc00e041bbddfcb5bdfdaf9cc2134ecb8b89f447921ec23dd82`.
- Because this cell appeared after root7500 had already made the frozen per-root gate impossible, root7501/F6 is `POST_GATE_PRESTARTED_EXTRA_EVIDENCE_EXCLUDED_FROM_CONFIRMATORY_GATE`. It cannot alter `TASK5_DEV_FAIL_NO_ROBUST_REGIME_TRANSFER`.
- Five DEV cells are now observable and all five are `LOCAL_FAILURE`; root7501/F7 remains `NOT_RUN_DUE_FROZEN_GATE_IMPOSSIBLE_AND_THROUGHPUT_CONSERVATION`. The analyzer's sole incompleteness is that intentionally missing fixed cell.

## Retry gate closed by unrequested root7501/F7 — 2026-09-02 10:04:35 UTC

- Fresh GPU audit: 91% utilization, 22,126 MiB used, 18,315 MiB free. Shared pi0 PID 33931 and MASS PID 603571 remain `PROTECTED`.
- A new exact isolated root7501/F7 process PID 609394 (launcher parent 609391) was detected, started at 10:02:34. Root did not authorize it and Agent B did not invoke it after the frozen stop decision. Ownership is ambiguous; it is `PROTECTED_PENDING_OWNERSHIP` and will receive no signal or duplicate.
- Because this is another post-gate prestarted cell, any naturally finalized valid result can only be preserved as `POST_GATE_PRESTARTED_EXTRA_EVIDENCE_EXCLUDED_FROM_CONFIRMATORY_GATE`; it cannot alter the already-failed root7500 criterion.
- Reserve task2 retry is not launched. Resource gate is `CLOSED_UTIL_91_PERCENT_AND_PENDING_F7`; wait for natural exit, validate F7 separately, then require a new GPU/process audit and a new timestamped task2 output root.

## Post-root7501/F7 validation — 2026-09-02 10:05:38 UTC

- PID 609394 exited naturally. The analyzer validates root7501/F7 with the same root hash `220bd2ff0b9aa6b50034b23844b2c0a12f7dea6dfa02eb4b3477b351620f2b65`, query reached, pick=1, lift=1, transport/place/FullTask=0, 500 steps, measured mean 0.173 N and peak 6.956 N. Episode CSV SHA-256 is `a4aff4e58bceeac04e14bc62a6ffa070abf82820a60e6b4671cfd19e6429dbfa`; step CSV SHA-256 is `f17c43839a7633a1cafeaef44a2073e0b1090fcae3e027673327b2d0138668d6`.
- The six-row analyzer now reports no errors: five `LOCAL_FAILURE`, one `LOCAL_SUCCESS_DOWNSTREAM_FAILURE`, zero FullTask successes, stable within-root hashes, and distinct root hashes. All six exact cells belong to the frozen six-cell DEV plan and enter the confirmatory decision; none is excluded.
- This extra branch demonstrates a descriptive LocalLift/FullTask label divergence but does not cure the zero-positive FullTask target or the failed root7500 per-root regime gate. No powered classifier or robust-transfer claim is made.

## Reserve retry gate remains closed — 2026-09-02 10:06:30 UTC

- Fresh `nvidia-smi`, `ps`, and process-pattern audit found no running or queued reserve task2/task8 launcher. A new E5 PID 611219 (`run_e5_fresh_utility.py --task 5 --tuple-start 2`) is immediately `PROTECTED`; pi0 PID 33931 and MASS PID 603571 remain protected.
- GPU: 90% utilization, 23,048 MiB used, 17,393 MiB free. Decision: `RESERVE_TASK2_RETRY_PROHIBITED`; no Agent B GPU process was started.
- Continue CPU/offline provenance updates and wait for a fresh coordinated gate. The retry must use a new timestamped root and exactly task2; no task8 launcher is queued.

## Root-confirmed closed gate — 2026-09-02 10:09-10:10 UTC

- Repeated read-only snapshots showed 87-94% utilization, 23,052 MiB used, and 17,389 MiB free. Protected compute allocations remain pi0 PID33931 (8,640 MiB), MASS PID603571 (7,560 MiB), and E5 PID611219 (6,752 MiB).
- Root explicitly set the gate to `CLOSED` and prohibited task2 retry or any GPU job until an explicit GO. Agent B launched nothing.
- Offline reserve plan/launcher/wrapper/analyzer compile and syntax QA passed; hashes and retry constraints are recorded in `E3_RESERVE_TASK2_RETRY_READINESS.md`.

## Approved reserve task2 retry launch — 2026-09-02 10:16:09 UTC

- Root sent explicit GO for exactly one task2 retry. Immediate audit: GPU 34% utilization, 15,452 MiB used, 24,989 MiB free; protected pi0 PID33931 and E5 PID611219 active; MASS PID603571 absent after natural exit. Exact reserve/task2/task8 duplicate scan was empty.
- Launched one foreground frozen `libero_10/task2` TRAIN nominal qualification at root7600, mu=0.6, diagnostic setpoint 8 N. New output: `/media/volume/newdata/exouser/activeforcing_e3/RESERVE_LONGHORIZON_QUALIFICATION_RETRY_20260902_101609/t2`.
- Owned processes: launcher PID616019 and Python PID616040, unified session73714. Immediate GPU snapshot was 33% utilization with 23,743 MiB free. No task8 or other Agent B GPU job was launched.
- Protected pi0/E5 will not be signaled or modified. Accept task2 only after natural completion plus episode/step/root/goal analyzer QA.

## Reserve task2 valid completion — 2026-09-02 10:22 UTC

- An external MASS task8 PID616079 appeared immediately after the approved task2 launch and was protected. It exited naturally; no signal was sent. Another MASS task9 PID619215 then appeared at 10:20 and is `PROTECTED`. E5 PID611219 and pi0 PID33931 remain protected. Agent B launched nothing else.
- Owned task2 PID616040 completed naturally after one 800-step/80-chunk episode and wrote complete telemetry. This is not exit137 and not a resource abort.
- Analyzer verdict: `VALID_NOMINAL_QUALIFICATION_FAILURE_PI0_NO_GRASP`. Root hash `b9006260dcdf73bc08cff649a65b18be9b1d1860593e0dd4ad36c0a81ab55760`; grasp/lift/transport/place/turn/official FullTask all zero; goal vectors stayed `[0,0]`; maximum object XY displacement 0.000171 m; measured mean/peak force 0/0 N.
- The contact/proximity flag was positive on 54/800 rows while measured squeeze remained zero throughout, so this is not interpreted as an established-contact controller result. The primary failure is upstream frozen-pi0 nominal execution.
- Task8 is prepared offline as the next frozen reserve candidate. GPU gate remains closed by protected pi0/E5/MASS task9, and no task8 launch is authorized without explicit root GO.

## Unrequested frozen task8 already-running detection — 2026-09-02 10:24:09 UTC

- Agent B did not invoke task8 and did not perform a task8 prelaunch audit because the active instruction was offline-only and wait for explicit GO. The last actual prelaunch audit was 10:16:09 for task2 and found no task2/task8 process.
- Launcher PID621733 (PPID466936) appeared at 10:23:33 with child PID621736 running the exact frozen isolated task8 command at root7600, friction0.6, diagnostic8N, output `/media/volume/newdata/exouser/activeforcing_e3/RESERVE_LONGHORIZON_QUALIFICATION_RETRY_20260902_102500_t8/t8`. Ownership is ambiguous; both are `PROTECTED_PENDING_OWNERSHIP`.
- Immediate audit: GPU 90% utilization, 20,872 MiB used, 19,570 MiB free; protected pi0 PID33931 and MASS task9 PID619215 active. E5 PID611219 had exited naturally. No signal or duplicate was issued.
- Root directed that this exact single frozen task8 be allowed to finish naturally because the numeric gate had passed immediately before it appeared. Accept it only after complete telemetry/analyzer/hash QA. No other job may launch.

## Protected pending-ownership task8 valid completion — 2026-09-02 10:28 UTC

- PID621736 exited naturally and wrote one episode, 800 step rows, and 80 raw chunks. Analyzer/schema/hash checks pass; ownership remains `EXTERNAL_OR_QUEUED`, and Agent B makes no launch claim.
- Result: root hash `81fd20a39b0059dbe5276cc6726fd9c87447080f2ec6cc16bf4bc53a31cc5528`; all three goal flags stayed `[0,0,0]`; grasp/lift/transport/place/turn/official FullTask all zero; maximum monitored-object XY displacement 0.000118 m; measured mean/peak0/0 N.
- Verdict: `VALID_NOMINAL_QUALIFICATION_FAILURE_PI0_NO_GRASP`. Contact/proximity flag was positive on 18/800 rows but measured squeeze was zero throughout, so no established-contact controller claim is made.
- MASS task9 PID619215 and pi0 PID33931 remain protected at completion. No further Agent B job was launched. G3 is prepared offline only and awaits explicit root GO plus fresh resource audit.

## Unrequested frozen g3 already-running detection — 2026-09-02 10:29:04 UTC

- Launcher PID624466 (PPID466936) and child PID624472 appeared at 10:28:07 running the exact frozen isolated `libero_goal/task3` nominal cell, output `/media/volume/newdata/exouser/activeforcing_e3/RESERVE_LONGHORIZON_QUALIFICATION_RETRY_20260902_102800_g3/g3`. Agent B did not invoke it.
- Root's immediately preceding audit found 41% utilization, about22GB free, protected pi0 plus MASS task9, and no duplicate g3. Root instructed natural completion/validation only.
- Agent B's immediate check at 10:29:04 found 95% utilization and 19,424 MiB free with pi0 PID33931, MASS PID619215, and g3 PID624472. G3 is `PROTECTED_PENDING_OWNERSHIP_EXTERNAL_OR_QUEUED`; no signal, duplicate, or further job will occur.
# 2026-09-02T10:47:26Z — Agent B task5 ID1 real-tactile replay prelaunch audit

- Host GPU: 35% utilization, 15,614 MiB used, 24,827 MiB free.
- Protected PID 33931: authoritative `pi0_lora_tacfield_tabero/49999` server on port 18881.
- Protected PID 627135: E5 task6 fresh Utility tuple02 Isaac rollout.
- Prior MASS task9 had exited naturally. No matching `replay_demos_with_camera.py`, `Replay-Camera-Tactile`, task5 7dpf, or ID1 smoke process existed; the `pgrep` match was the audit shell itself.
- Numeric gate passed (`util < 50%`, free memory >18 GiB) with one protected Isaac job. Authorization is limited to exactly one ID1 smoke. No other selected demo may start without separate root GO.
- Agent B will use only a fresh output directory and will not signal/modify either protected process.

## 2026-09-02T10:48:50Z–10:50:00Z — owned ID1 smoke

- Agent B owned PID 646854 only. It exited naturally after one successful 187-step replay; no signal was sent.
- Protected PID 33931 remained untouched. Protected E5 PID 627135 exited naturally after the launch.
- Output HDF5 is valid 13D/tactile-schema evidence, but the overall smoke failed closed because the historical hardcoded `/usr/bin/ffmpeg` was absent and no MP4s were produced.

## 2026-09-02T10:51:17Z — externally queued duplicate

- PID 649947 -> 649969, cwd `/media/volume/newdata/exouser/Tabero_e3lh`, output `TASK5_ONBOARDING_REPLAY5_20260902_104200/smoke_demo1` appeared without an Agent B launch.
- Classified `PROTECTED_PENDING_OWNERSHIP_EXTERNAL_OR_QUEUED`; no duplicate was launched and no signal or code modification occurred while it ran.
- It exited naturally with the identical HDF5 and identical missing-media failure. Only after exit was the authorized offline ffmpeg patch copied into the isolated checkout.

## 2026-09-02T10:54:50Z — patched ID1 retry

- External/queued PID 654230 -> 654254 started from the patched isolated checkout with explicit verified `TABERO_FFMPEG_BIN` and a fresh output root.
- Classified `PROTECTED_PENDING_OWNERSHIP_EXTERNAL_OR_QUEUED`; Agent B launched no duplicate and sent no signal.
- Protected E5 PID 655134 task5 tuple03 started after the replay; utilization rose to 99%. Both jobs were left untouched.
- Retry exited naturally and passed the full data gate: one successful 187-step 13D episode plus four independently decoded, 187-frame media streams. No additional demo was launched.

## 2026-09-02T11:00Z — remaining-four batch preparation, GPU gate closed

- Coordinator snapshot: GPU utilization `85%`; protected π0 PID `33931`, MASS PID `647777`, and E5 PID `655134`.
- No Agent B GPU job was launched. The next authorized shape is exactly one TRAIN-only batch containing source demo IDs `[2,11,12,19]`; ID1 is excluded and its accepted output remains immutable.
- Offline preparation created an immutable 714-step source subset, one fail-closed launcher, and strict raw-to-source identity/schema/media validation. Launch still requires explicit coordinator GO and a fresh `nvidia-smi`/`ps`/`pgrep` audit.

## 2026-09-02T11:15Z — remaining-four GO superseded by fresh closed gate

- Coordinator GO was based on 38% utilization and protected π0 PID `33931` plus MASS PID `647777`; the duplicate scan was empty.
- Agent B's immediate host audit found a newly auto-started protected E5 PID `672168` (`run_e5_fresh_utility.py --task 6`, tuple03). The first full GPU sample was 47% utilization with 24,066 MiB free; the launcher's own final gate sample was 89% utilization with 17,198 MiB free.
- The launcher exited fail-closed before creating the output root or starting Isaac. There is no Agent B replay PID and no partial batch data. A new explicit GO and fresh audit are required.

## 2026-09-02T11:35:06Z — pre-outcome activation of preregistered single-demo fallback

- Host audit found core GPU scheduler PID `679695` had externally started launcher PID `687488` and task5 TRAIN demo2 replay PID `687494` in the isolated Tabero checkout. Agent B did not launch or signal it.
- Before inspecting any demo2 output/outcome, the coordinator activated the five-isolated-job fallback that was already declared in `TASK5_PI0_ONBOARDING_REAL_REPLAY_READINESS.md` before the ID1 result.
- Demo2 is `PROTECTED_PENDING_OWNERSHIP_EXTERNAL_OR_QUEUED`. It may count only after the unchanged per-demo success/schema/identity/media gate. No successor may start automatically; IDs11/12/19 each require fresh audit and explicit GO.

## 2026-09-02T11:37Z — protected external demo2 natural completion

- Scheduler PID `679695`, launcher PID `687488`, and replay PID `687494` exited naturally. No signal was sent.
- Independent fail-closed validation passed: exact source demo2 initial state, semantic success, 180 finite 13D action rows, real nonzero force and marker motion, four fully decoded 180-frame media streams, and no failure record.
- HDF5 SHA-256 is `e9bd7466baa3c1e2dfccc039ebaa63b7cc7ab48d2ba0e49263c354ac6c222a50`. No ID11 or other Agent B GPU job was launched.

## 2026-09-02 11:50:53 UTC — GPU gate remains closed during CPU-only readiness work

- Fresh host GPU snapshot: utilization `60%`, memory used `24,063 MiB`, memory free `16,378 MiB`.
- Protected π0 server PID `33931` remains active with authoritative checkpoint 49999.
- Protected MASS PID `647777` remains active (`collect_mass_structured_formal.py --task 2`).
- Protected E5 PID `697207` remains active (`run_e5_fresh_utility.py --phase full --task 0 --tuple-start 5 --max-tuples 2`).
- No E3 replay, conversion, normalization, or training job was launched. The prior three-way-load replay slowdown and current free-memory/utilization state keep the gate closed; ID11 remains pending explicit coordinator GO after a new audit.

## 2026-09-02 12:13–12:18 UTC — protected external/pending-owner demo11 validation

- Coordinator classified the completed demo11 output as external/pending-owner under the active serial fallback. Agent B did not launch or signal it and launched no successor.
- Output `/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_REPLAY5_20260902_110000/demo11` independently passed exact source identity, semantic success, finite `[200,13]` actions, real nonzero force/marker, and four fully decoded aligned media gates. No nonempty failure record exists.
- HDF5 SHA-256 is `6f0e122d60ae05ea56a20da8d0dbebfca48b567fc6359ac73d7aa64e574d2baa`; producer QA SHA-256 is `0f0781c52fa5a13e71506fddfb4726c9aab3220b027e8d55e0978cd93a8dfa72`.
- The replay log ends with a weak-reference exception during Isaac teardown after complete outputs. The gate does not claim a clean exit; it accepts data integrity based on independent HDF5/media decoding and producer-QA checks.
- Protected π0 PID `33931`, MASS PID `647777`, and E5 PID `1781117` remain out of scope and untouched. No ID12, ID19, conversion, normalization, training, or other GPU job was launched.

## 2026-09-02 13:04:04 UTC — CAM successor and external E5 remain protected

- MASS owner launched the valid CAM successor PID `2076670`, command `collect_mass_structured_formal.py --task 2`, output `M3_TASK2_FORMAL_STRUCTURED_20260902_130600_CAM`. It is `PROTECTED` and outside Agent B ownership.
- External E5 task5 PID `2072012` remains active and `PROTECTED`; authoritative π0 server PID `33931` remains `PROTECTED`.
- Fresh read-only GPU audit reported `88%` utilization, `23,055 MiB / 40,960 MiB` used. The only listed GPU allocations were PID `33931` (`8,640 MiB`), PID `2072012` (`6,752 MiB`), and PID `2076670` (`7,564 MiB`).
- The E3 gate remains `CLOSED`. Demo12/demo19, conversion, normalization, training, and every other new GPU workload remain frozen until the scheduler/coordinator issues an explicit fresh `FINAL_GATE_PASS` after its required checks.
- Agent B launched and signaled nothing; the checks in this entry were read-only.
