# Current running process audit

Authoritative host snapshot: `2026-09-02T13:38:14Z`.

- GPU 0: NVIDIA A100-SXM4-40GB, `23,181/40,960 MiB` used, `17,260 MiB` free, `89%` GPU utilization. The preceding 13:35 sample was 95%, so the card remains heavily active.
- Per-process GPU utilization is not exposed by the available `nvidia-smi` query, so only GPU memory is assigned to individual PIDs.
- `tmux ls`: no socket. `screen -ls`: no sockets.
- No process was killed, signaled, restarted, or modified by this audit.

| PID | Role | Start | CWD | GPU memory | State | Current work | Protected |
|---:|---|---|---|---:|---|---|---|
| 33931 | frozen pi0 server | 04:11:17Z | `/home/exouser/FORTE` | 8640 MiB | Ssl | config `pi0_lora_tacfield_tabero`, checkpoint 49999, port 18881 | yes |
| 2080255 | Mass Isaac collector | 13:08:06Z | `/home/exouser/Tabero` | 7564 MiB | Rsl+ | task2, formal resume, roots 8100-8103/8200-8201, repeat 2 | yes |
| 2096169 | E5/E3 core scheduler | 13:26:03Z | `/home/exouser/FORTE` | 0 | Ssl+ | wait/orchestration; parent of PID 2099454 | yes |
| 2099454 | E5 Isaac rollout worker | 13:27:54Z | `/home/exouser/Tabero` | 6880 MiB | Rl+ | task1 tuple root01/seed7201/LOW, offset03 | yes |

## Process-specific truth

### PID 2080255 — Mass

- This is data collection, not model training.
- Formal target is 6 roots x 3 mass bands x 5 forces x 2 repeats = 180 branch rows.
- At the final snapshot, the output CSV contained 70 data rows. The last flushed row was TRAIN root8102/LOW; the exact in-memory cell cannot be determined without instrumenting the process.
- Protocol file still says `RUNNING`; no `MASS_FINAL_REPORT`, `TABLE_MASS`, or mass E2E closure exists. Process presence proves only that collection is active.

### PID 2099454 — E5

- This is a single-tuple reset-to-end worker under V2 Expected Utility.
- At the snapshot it had persisted two of five rollout rows. `query_valid=0` is visible on the first partial ActiveForcing row; no shard-level verdict is assigned while the worker is active.
- It is **not** included in the canonical 20/60 accepted count. Acceptance requires natural exit plus `FULL_STATUS` and `SHARD_QA` PASS, followed by coverage re-audit.

### PID 2096169 — scheduler

- It is not a scientific result and consumes no GPU memory.
- Scheduler log declares `never signal; MASS read-only priority` and alternates gated E3/E5 work.
- It discovered/accepted demo12 QA and then launched the current E5 tuple. It must not be confused with an E5 rollout or a completed gate.

### PID 33931 — pi0 server

- This is the shared frozen backend. It is not training and not an experiment completion marker.
- Its presence is a runtime dependency for E5/E3/other native pi0 rollouts.

## Negative findings

- No E7 exact-float worker.
- No Boundary preflight/collection worker.
- No E3 replay or LoRA training worker at the final snapshot.
- No joint-physics worker.
- No FORTE/Tabero baseline rollout worker.
- Multiple Codex app-server/code-mode-host processes exist, but they are agent infrastructure rather than scientific jobs and do not hold GPU allocations in `nvidia-smi`.
