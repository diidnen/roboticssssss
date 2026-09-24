# GPU/NVML diagnostic

The A100 and CUDA runtime are healthy. The failure was a restricted-execution-context NVML problem amplified by a single-shot launcher gate.

- Host `nvidia-smi` repeat: 30/30 successful.
- Tabero policy-environment CUDA test: 10/10 successful.
- Isaac simulator-environment CUDA test: 10/10 successful.
- Frozen policy server `127.0.0.1:18885`: 10/10 metadata probes successful; PID 2868465 and checkpoint SHA256 match the frozen manifest.
- No-step Isaac task-scene startup: PASS, with zero explicit reset calls, zero explicit `env.step` calls, and zero physics outcomes.
- The restricted Codex execution namespace intermittently returned `nvidia-smi` code 9 and hid host `/proc` PIDs and `/dev/nvidia*`; host-level inspection showed all devices and servers normally.

Kernel `dmesg` and complete journal access were unavailable to the user. No Xid or fallen-off-bus evidence was found in accessible logs. Stable CUDA allocation, server inference state, device nodes, and Isaac startup provide strong indirect evidence against a real driver failure.

`INFRASTRUCTURE_FAILURE_CLASS = NVML_ONLY` with the qualifier that the fault was sandbox/device-namespace scoped.
