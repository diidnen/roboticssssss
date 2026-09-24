# Protected Running Experiments

Snapshot: 2026-09-02 08:40 UTC

All processes below are protected and were running before Agent A started. Agent A will not signal, restart, reconfigure, or write into their checkouts/output directories.

| PID | Role | Command / output scope | GPU memory |
|---:|---|---|---:|
| 33931 | shared frozen pi0 control server | `visual_pi0_server.py --port 18881`; checkpoint `.../49999` | 8640 MiB |
| 555222 | MASS / joint-physics Isaac job | `/home/exouser/FORTE_mass/qualify_mass_structure.py --task 3`; output `M2_STRUCTURED_TASK3_ROOT7042` | 6034 MiB |
| 556562 | E5 fresh Utility E2E Isaac job | `run_e5_fresh_utility.py --phase full --task 1`; output `...083500_task1_tuples01_02` | 6752 MiB |
| 559267 | MASS monitor shell | read-only monitor for PID 555222 | none |

GPU snapshot: NVIDIA A100-SXM4-40GB, 92% utilization, 21,523 MiB used, 18,918 MiB free.

Policy consequence: GPU utilization is above 90%, so Agent A is restricted to CPU/offline analysis until a new preflight shows capacity and no material throughput risk.

Preflight commands used: `nvidia-smi`, `ps`, and `pgrep` from the host namespace.
