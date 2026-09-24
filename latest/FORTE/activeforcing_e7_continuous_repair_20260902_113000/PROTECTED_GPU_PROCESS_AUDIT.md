# E7 Protected GPU/Process Audit

No E7 Isaac or GPU process was started during the offline repair.

## Preflight observations

At 2026-09-02 11:25 UTC, the A100 was at 23,247 MiB allocated and 92% utilization. Protected owners included:

- Mass: PID 647777, `collect_mass_structured_formal.py`, approximately 7,756 MiB.
- E5: PID 672168, `run_e5_fresh_utility.py`, approximately 6,752 MiB.
- Frozen visual pi0 server: PID 33931, approximately 8,640 MiB.

At 2026-09-02 11:44 UTC, the A100 was at 24,063 MiB allocated and 83% utilization. Protected owners included:

- Mass: PID 647777, still active, approximately 7,756 MiB.
- E5: PID 697207, task0 run, approximately 7,568 MiB.
- Frozen visual pi0 server: PID 33931, approximately 8,640 MiB.

At 2026-09-02 12:12 UTC, the A100 was at 30,068 MiB allocated and 74% utilization. Protected owners included:

- Mass: PID 647777, still active, approximately 7,756 MiB.
- E3: PID 1767274, task5 onboarding replay, approximately 5,972 MiB.
- E5: PID 1781117, locked task0 run, approximately 7,560 MiB.
- Frozen visual pi0 server: PID 33931, approximately 8,640 MiB.

No process was signaled, stopped, reprioritized, or modified. Exact-float E7 execution remains queued until a fresh host preflight confirms that protected Isaac workloads are absent and the GPU is no longer saturated.
