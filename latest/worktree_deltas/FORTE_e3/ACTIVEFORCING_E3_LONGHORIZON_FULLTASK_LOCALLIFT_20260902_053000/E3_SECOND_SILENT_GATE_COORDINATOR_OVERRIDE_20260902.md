# E3 coordinator final-gate override V2

This amendment is operational only and was recorded before demo12/demo19 outcomes. It does not change the selected demos, pi0 training, nominal DEV roots/threshold, force semantics, labels, Utility, Fmax, or TEST firewall.

Core E5 PID `2059513` exited `-9` during the dual-E5 race. Its partial output is rejected and locked-test coverage remains `17/60`. External E5 PID `2059639` remains protected and running; MASS PID `647777` is healthy; pi0 PID `33931` is protected. GPU utilization was approximately 90% with about 17.2 GiB free. Scheduler PID `2068026` is wait-only.

For every future E3 GPU operation, V1's 120-second template/validator remains preserved as provenance but is no longer sufficient to launch. V2 requires external E5 PID `2059639` to have ended naturally, zero active E5 PIDs, at least 10 seconds of observed silence, an immediate final resource/duplicate recheck, and a fresh explicit coordinator `FINAL_GATE_PASS`. Natural termination by itself is not authorization, and every operation still requires a separate gate.

No demo12/19, conversion, normalization, training, server, nominal DEV cell, or other GPU workload was launched while recording this amendment.

## 2026-09-02 13:04:04 UTC coordination update

The protected workload set has advanced: MASS owner has a valid CAM successor PID `2076670` writing `M3_TASK2_FORMAL_STRUCTURED_20260902_130600_CAM`, external E5 task5 PID `2072012` is still active, and pi0 PID `33931` remains protected. A fresh read-only GPU snapshot showed `88%` utilization and `23,055 MiB / 40,960 MiB` used. This is a closed gate. The V2 requirements above remain necessary but not sufficient: no E3 GPU operation may start until the scheduler/coordinator explicitly issues a fresh `FINAL_GATE_PASS`. No E3 GPU workload was launched or signaled during this update.
