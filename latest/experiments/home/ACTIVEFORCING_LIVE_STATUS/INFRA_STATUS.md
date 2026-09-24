# ActiveForcing Infrastructure Status

Updated: 2026-09-03T01:39Z.

## Current verdict

`INFRA_RECOVERED_FOR_SAFE_ADMISSION`: PASS.

- Root disk: 30.10 GiB free (`32301867008` bytes; 31G rounded by `df -h`), 49% used; the operator-adjusted admission floor is 10 GiB and is currently PASS. Archive disk holds approximately 28G of reversible historical archives.
- NVIDIA: stable five-sample `nvidia-smi` PASS.
- CUDA: PyTorch allocation/matmul PASS.
- π0: existing authoritative native websocket server PASS; fixed-sample inference PASS.
- E3: ID1 read-only refreeze PASS; norm path root cause repaired in execution scripts; training remains downstream of clean source freeze and fresh scheduler gate.
- Handoff: tmux session `af_e5_resume_20260903e` is active; patched scheduler PID 2237816 runs E5 worker PID 2238217 and fails closed before any new shard if free space drops below 10 GiB.

## Active experiment processes

- Mass: collection complete at 180/180 accepted formal branch rows; no collector remains active.
- π0 server PID 2128032: protected, authoritative checkpoint 49999.
- E5 worker PID 2238217: protected, task1 offset04 in progress; scheduler PID 2237816 is the current fail-closed parent.

No process was killed, no accepted result was overwritten, and no driver reboot/reload was performed.
