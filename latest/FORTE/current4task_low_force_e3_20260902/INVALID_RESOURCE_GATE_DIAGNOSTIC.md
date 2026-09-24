# INVALID_RESOURCE_GATE_DIAGNOSTIC

Status: **DIAGNOSTIC_NOT_FINAL**

The task0 collector launched only after its recorded resource gate passed at
2026-09-02T09:19:35Z (GPU utilization 41%, 24179 MiB free, protected MASS
visible). Subsequent independent Paseo lanes started E3 and E5 workloads and
the machine reached 99% utilization with less than 8 GiB headroom.

The worker ended with return code `-9` after only two of 36 planned branches.
Its partial telemetry is retained for engineering diagnosis only. It must not
be merged into E3 label-support tables, used for tuning, or counted toward an
E3 completion condition. No MASS process was signalled or modified.
