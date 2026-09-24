# GPU health gate patch

`OLD_LAUNCHER_SHA256 = 7f1cf8101203f5f4be59057bc971477e0149fdb8d307e6e06f8bb72096a3cad8`

`NEW_LAUNCHER_SHA256 = 161437240134bc469dcb192c4e14119a1a094b455c59fdb5f3ab481fe7bd334c`

The launcher now retries the exact NVML query up to ten times. If all NVML attempts fail, it performs a small CUDA allocation/synchronization and checks CUDA free memory. Every launch also verifies that port 18885 returns the exact frozen policy-server metadata, PID, checkpoint SHA256, and online-VLA source. It records the signal source and all attempts in `GPU_HEALTH_GATE_LOG.jsonl` and each job's `GPU_HEALTH_PREFLIGHT.json`.

The gate remains active. A branch is rejected if both NVML and CUDA resource checks fail, if available memory is insufficient, or if the frozen policy server does not match.

`SCIENTIFIC_RUNTIME_CHANGED = NO`

`INFRASTRUCTURE_LAUNCH_CHECK_ONLY = YES`

The root list, method order, execution plan, models, posterior, feasibility, utility, controller, evaluator, force support, and scientific manifest were not changed.
