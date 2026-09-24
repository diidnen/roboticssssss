# Recovery log

- 2026-09-02T14:55Z: disk recovered to approximately 30G free by reversible archive moves and clean Git worktree removal; see `/home/exouser/DISK_CLEANUP_LOG.md`.
- 2026-09-02T14:55Z: NVIDIA five-sample stability and PyTorch CUDA smoke PASS; no reboot or driver package change.
- 2026-09-02T14:55Z: existing authoritative π0 websocket server reused; fixed-sample inference returned `(50,13)` actions.
- 2026-09-02T14:55Z: E3 ID1 live-QA hash drift reconciled with a new read-only refreeze; norm path resolved to the nested config/dataset-repo location. Training remains fail-closed until source/gate freeze.

- 2026-09-02T14:21Z: live `nvidia-smi` failed with NVIDIA driver communication error; no GPU restart attempted.
- 2026-09-02T14:22Z: confirmed no scientific worker, scheduler, or pi0 process remains visible.
- 2026-09-02T14:22Z: accepted E5 offset03 only after `FULL_STATUS=PASS` and `SHARD_QA=PASS`; coverage is 21/60.
- 2026-09-02T14:22Z: Mass CSV QA found 100 complete-width rows / 10 complete contexts; root8103-mid in-flight material excluded.
- 2026-09-02T14:23Z: created recovery audit, resume points, quarantine manifest, and fail-closed live status.
- 2026-09-02T14:24Z: created nine isolated FORTE recovery worktrees and one CPU coordinator worktree from commit `7f88d01`.
- 2026-09-02T14:25Z: Tabero dedicated worktree creation failed closed because the filesystem is 100% full; no original Tabero worktree was modified.
- 2026-09-02T14:26Z: launched nine CPU/QA recovery agents plus one CPU coordinator through Paseo; all carry `AF_GPU_GATE=BLOCKED_DRIVER` and are forbidden from GPU/TEST/MASS/Utility mutations.
- 2026-09-02T14:28Z: independent host recheck still showed no scientific process and NVIDIA driver unavailable. Coordinator observed a transient external Mass/pi0 relaunch between checks; treated as an unowned race hazard and not as accepted evidence.
- 2026-09-02T14:38Z: CPU recovery lanes completed/idle: E5 QA, Mass atomic snapshot, E6 Utility disagreement recompute, E7 manifests, Boundary preflight manifest, Joint protocol, baseline provenance, and E3/E2 readiness diagnostics. No lane authorized GPU work.
2026-09-02T15:10Z — infrastructure resume handoff

- Exact root free space rechecked at approximately 30.017 GiB after reversible archive cleanup; disk admission PASS.
- NVIDIA A100/driver and CUDA smoke remain PASS; protected Mass and π0 processes unchanged.
- The first nohup watcher exited because `pipefail` interacted with `ps | rg -q`; it was corrected to `pgrep` and relaunched in persistent tmux session `af_resume_handoff_20260902`.
- Mass resume advanced to 150 raw branch rows/15 raw contexts; accepted count remains 100/180 pending independent QA.
- E5 remains fail-closed until Mass completion and fresh token admission.
