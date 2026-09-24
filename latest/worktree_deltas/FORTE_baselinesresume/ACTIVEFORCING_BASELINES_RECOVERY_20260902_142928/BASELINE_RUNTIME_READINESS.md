# Faithful FORTE/Tabero baseline recovery status

Audit time: `2026-09-02T14:31:57.362485+00:00`  
Lane: `/home/exouser/FORTE_baselinesresume`  
Execution policy: **CPU-only; no simulator, policy-server request, serial connection, or rollout.**

## Decision

- **FORTE:** `IMPLEMENTATION_BLOCKED`. The official CPU model and runtime source are intact, but the benchmark cannot be called FORTE until a six-channel analog-tactile adapter and a Dynamixel-equivalent impedance-position actuator mapping pass fidelity QA.
- **Tabero:** `RUNTIME_COMPONENTS_READY / PAIRED_DATA_NOT_RUN`. The clean source checkout, native 13-D path, frozen checkpoint metadata/norm stats, task datasets, exact snapshots, and no-override task configs are present. No paired result is claimed.
- **Paired manifest:** `144` tuples (`72` contexts, `24` task-specific roots, two repeats; 36 tuples per task) are frozen in `E1_EXTERNAL_BASELINE_PAIRED_TUPLE_MANIFEST.csv`.
- **Scientific result:** `NONE`. Every FORTE/Tabero outcome remains `NA`; oracle-slip, fixed/ladder-force, and unpaired native smoke rows are excluded.

## Faithfulness gates

| Gate | FORTE | Tabero |
|---|---|---|
| Source/CPU contract | PASS | PASS (compile/static contract only) |
| Frozen model/checkpoint provenance | PASS (`9d9ba2449282196db1a85f31e1e41cdca7ccd56522130e2ff3deffd93e8017bc`) | PASS via inherited frozen content digest plus current metadata hashes |
| Exact tuple/reset manifest | Frozen; unusable until adapter exists | Frozen; all snapshot paths and exact-restore hashes present |
| Native sensor/observation semantics | BLOCKED: six analog channels absent | READY: visual/tactile/force history native path present |
| Native actuator semantics | BLOCKED: Dynamixel impedance adapter absent | READY: native 13-D force-position action path present |
| Same-tuple rollout | NOT RUN | NOT RUN |
| Paper/result row | NA | NA |

## Paired-tuple acceptance contract

A future row is admissible only when it uses the manifest's task, `root_seed`, friction, repeat, and exact snapshot hash; starts from a fresh reset/verified restore; reaches an explicit terminal transition; and records commanded left/right force separately from measured contact force, under/excess force, latency, and failure stage. Tabero must execute its learned 13-D outputs without any ActiveForcing force-slot overwrite. FORTE must preserve the native 2 kHz six-channel → 24-feature SVR → 10–50 Hz Welch/variance slip → `-0.006` impedance-position reaction with 0.2 s cooldown.

## Blockers and next legal actions

1. Build and audit the FORTE sensor/actuator simulator adapter. Until both mappings pass, omit FORTE from numerical tables.
2. Restore disk capacity, create a dedicated Tabero baseline checkout, and freeze a lane-local exact-manifest native runner.
3. Only then schedule Tabero same-tuple collection and, after FORTE fidelity passes, FORTE collection. This audit intentionally stops before either simulator action.

The machine's root filesystem was observed at 100% utilization with about 420 MiB free. That is a checkout/output prerequisite blocker, not a scientific result.
