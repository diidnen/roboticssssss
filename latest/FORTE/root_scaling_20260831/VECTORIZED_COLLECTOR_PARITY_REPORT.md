# Vectorized Isaac microtest and parity decision

Date: 2026-08-31 UTC

## Decision

Vectorized Isaac execution is **not enabled** for the authoritative experiment. No frozen vectorized implementation exists in the current collector. The authoritative worker constructs `AppLauncher(..., num_envs=1)`, creates the environment with `num_envs=1`, and the frozen P5-S0-C runner indexes the single environment directly. Running `num_envs=2` or `4` would therefore be a new implementation, not a controlled execution-equivalent microtest.

The required microtest was intentionally not run against TEST or by modifying the scientific runner. This is a safety rejection, not an outcome claim. No vectorized outcome, GPU safety, or speedup is accepted as evidence.

## Evidence and gate results

| Candidate | Status | Restore/observation/action/outcome parity | GPU safety | Throughput |
|---|---|---|---|---|
| sequential, `num_envs=1` | PASS baseline | Existing authoritative TRAIN rows: required S50 parity PASS | NVIDIA driver query unavailable in this shell; no new claim | 520 completed branches / 14,543.8 logged execution seconds = 0.03575 branches/s (mean reciprocal duration; median branch reciprocal 0.04530/s) |
| single-process vectorized, `num_envs=2` | REJECTED_NOT_RUN | No parity-capable implementation; 0/0 comparisons | Not tested; no safety margin established | N/A; speedup N/A |
| single-process vectorized, `num_envs=4` | REJECTED_NOT_RUN | No parity-capable implementation; 0/0 comparisons | Not tested; no safety margin established | N/A; speedup N/A |

The sequential baseline is taken from the completed TRAIN stage log, including roots `root12`–`root63`; it is an engineering throughput reference only and is not a scientific selection criterion. The final TEST/TRAIN collector remains sequential.

## Required parity fields

Because candidates 2 and 4 were rejected before execution, no values are fabricated for restore state hash, initial observation hash, applied-force trace, π0 action semantics, outcome/failure reason, corrected telemetry, branch length, or final success evaluator. The CSV records these as `NOT_RUN`.

## GPU safety

`nvidia-smi` could not communicate with the NVIDIA driver in the current execution namespace. Consequently, peak VRAM, utilization, Xid, and CUDA/OOM evidence for candidate vectorized workers are unavailable. A candidate without both outcome parity and a measured VRAM safety margin cannot be promoted. No independent Isaac processes were started.

## Freeze consequence

The only safe freeze is `num_envs=1`, `worker_count=1`, using the existing sequential worker and unchanged physics, force, seed, branch, repeat, and success semantics.
