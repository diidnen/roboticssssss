# GPU queue recovery

Snapshot: `2026-09-02T13:38:14Z`, A100-SXM4-40GB. Memory `23,181/40,960 MiB`, free `17,260 MiB`, GPU utilization `89%` (13:35 sample: 95%).

## Already active; do not duplicate

1. **Mass formal task2 collection — PID 2080255, 7,564 MiB.** Protected. Let it finish naturally, then check protocol status and exact 180-row target before any training.
2. **E5 task1 offset03 — PID 2099454, 6,880 MiB.** Child of scheduler PID 2096169. It is an in-flight single tuple and is not part of the 20 accepted tuples until `FULL_STATUS` and `SHARD_QA` pass.
3. **Frozen pi0 server — PID 33931, 8,640 MiB.** Shared protected inference dependency; not itself an experiment result.

No E3, E7, Boundary, model-training, or joint-physics worker was active at the snapshot.

## Recommended queue after the in-flight workers

This is a non-launching priority order:

1. **E3 demo19 replay only.** Shortest dependency unlock; IDs 1/2/11/12 already pass. Requires a fresh protected-process/duplicate scan and final gate. After demo19 passes, assembly/conversion and norm stats are CPU work; LoRA training becomes the next E3 GPU step.
2. **E5 balanced continuation under a corrected arm plan.** Canonical coverage is only 20/60; continue only after confirming the next tuple is missing and no duplicate worker exists. Before declaring final, extend beyond the current five-arm runner to the authoritative comparison set.
3. **E7 frozen 8-rollout exact-float block.** Small, high-information gate. Do not jump directly to the 144 rollout matrix.
4. **E3 1000-step five-demo LoRA training.** Only after demo19, strict conversion and CPU norm-stat QA; then post-training nominal DEV gate.
5. **Boundary minimal force-interface preflight.** This is calibration, not scientific collection. Boundary acquisition remains forbidden until it passes.
6. **Mass downstream estimator/Direct/Utility/E2E.** Only after current formal collection passes QA; CPU stages should precede new simulator work.
7. **E6 true second-query pilot.** Only after a valid continuation contract exists; legacy rho launchers must not be used.
8. **Joint 3x3 friction x mass pilot.** Only after Mass formal closure freezes the mass axis.
9. **Tabero exact paired baseline.** Scientifically useful but behind the core E3/E5/E7 blockers.
10. **FORTE baseline.** Not queue-ready: faithful simulator adapter is missing.

## Gate policy recovered from disk

- A gate requires natural worker exit, at least 10 seconds of silence, an immediate duplicate/resource recheck, and a fresh explicit authorization artifact. A template is never authorization.
- Current scheduler logs say `never signal; MASS read-only priority`. Do not kill Mass or the shared pi0 server to accelerate another lane.
- Historical gate files list old PIDs and old coverage values. Always regenerate the process audit immediately before launch.

## If only three lanes may continue today

1. Finish and QA the already-running **Mass formal collection**.
2. Finish/repair the **E5 V2 campaign**, beginning with the in-flight shard and authoritative coverage/arm reconciliation.
3. Complete **E3 demo19 -> CPU conversion/norm -> onboarding training gate**.

E7 is the first reserve because its 8-rollout block is short, but it does not supersede the work already consuming GPU or E3's one-demo dependency.
