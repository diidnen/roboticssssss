# GPU live queue

## Admission state

`RECOVERED_WITH_E5_ACTIVE`: NVIDIA/CUDA and π0 smoke gates PASS. Mass is complete at 180/180; the authoritative π0 server remains protected. E5 is the only active heavy worker under a fresh three-layer admission token, and its scheduler is fail-closed.

## Queue after hardware recovery

1. E5 current canonical missing cell: task5 offset04; promote only after terminal status and independent QA, then continue the remaining 36 tuples.
2. Mass formal collection is complete at 180/180; modeling is admitted and quarantined spillover remains excluded.
3. Complete the E7 exact-float 8-rollout gate after E5 scheduling permits it.
4. Continue E3 only after the repaired lineage gate and a fresh explicit GPU admission window.
5. Boundary preflight/collection, Joint 3x3 collection, and faithful baselines follow their dependency gates.
# 2026-09-03 infrastructure gate and resume

- Root disk: PASS, approximately 30G free.
- NVIDIA/CUDA: PASS; A100 visible and PyTorch allocation PASS.
- π0: existing PID 2128032 protected on port 18881; fixed inference smoke PASS.
- Current allocations: E5 PID 2233493 is active alongside protected π0 PID 2128032; observed combined GPU memory is approximately 9.3 GiB with 31.1 GiB free.
- Admission: scheduler PID 2233086 required fresh GPU samples and the three-layer token grant. It now stops on any missing terminal/failed QA state and does not retry the same tuple.
- E3 training requires a dedicated low-utilization window and the frozen lineage repair; it must not compete with Mass.
