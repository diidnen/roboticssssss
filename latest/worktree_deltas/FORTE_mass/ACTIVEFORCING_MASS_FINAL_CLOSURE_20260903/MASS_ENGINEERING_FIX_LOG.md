# Mass engineering fix log

Status: **COMPLETED: QA, OFFLINE MODELING, MAIN BENCHMARK, AND FRESH E2E**.

## Changes and checks

- Restored the authoritative completed formal Mass source from the frozen resume directory; no raw branch was recollected.
- Added an independent QA implementation that compares the 180-row source against the 100-row accepted snapshot, checks the 80 newly promoted branches, verifies hashes/state parity, and excludes prior quarantine.
- Refused modeling unless `MASS_FINAL_BRANCH_QA.json` is PASS with 180 accepted branches.
- Kept all outputs under the independent `ACTIVEFORCING_MASS_FINAL_CLOSURE_20260903` directory.
- Added a small CUDA-trained identifier with three seeds and explicit CPU-safe fallback; no large Isaac collector or old Joint neural architecture was launched.
- Evaluated compact RGB summaries because all 36 retained query RGB artifacts are present; no visual backbone or TEST tuning was added.
- Implemented Mass-only Direct as `z=mass` with force and fixed polynomial interaction, trained on full-task outcome labels only.
- Used the authoritative Expected Utility formula `p_success*(8-force)+(1-p_success)*(-1)` with lower-force tie-break; hard-ρ minimum-force selection was not used.
- Corrected failure-analysis field references after validating the generated episode-table schema; reran the complete failure report successfully.
- Verified the protected frozen π0 server remains PID 2128032. The active E5 scheduler/worker was not killed, modified, or reused as a Mass result.
- Fresh coordinator carries the scheduler's authoritative HDF5/LIBERO environment explicitly when launched outside that scheduler.
- Fresh task-2 adapter extends the E5 helper's own task dictionaries before calling `configure_task`, fixing the task-map API ordering error.
- Fresh protection scan was extended to the separately named protected E7 Isaac worker after a short overlap was detected; only the agent-owned Fresh PID was terminated, while E7 and π0 remained untouched.
- Fresh protection scan was extended to the E6 worker name and scheduler-compatible marker. Two agent-owned Fresh coordinators were terminated when E6 appeared; protected E6 and π0 were left untouched. The completed Fresh runs were admitted only in clear windows.
- Patched the Fresh runner to keep one Isaac environment alive across the selected root/band sequence, avoiding task re-import churn while preserving reset-per-context semantics.
- Merged four independently completed Fresh bundles (root 8400 LOW/MID/HIGH and root 8401 LOW/MID/HIGH) into 30 rows across 6 matched contexts; no raw Mass recollection occurred.

## Input provenance

- Formal source: `/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME`
- QA manifest: `MASS_ACCEPTED_MANIFEST.json`
- Model metadata: `MASS_MODELING_METADATA.json`
- Runtime used for CUDA training: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`

## Fresh E2E gate and completion

Fresh E2E was started only during admitted GPU windows in which no protected Isaac worker was active. The frozen π0 server was reused read-only through its existing endpoint. The merged protocol is COMPLETED with 30 rows and 6 contexts; checkpoint hash and π0 provenance are recorded in the protocol and report.
