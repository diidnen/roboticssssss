# ActiveForcing CPU-only recovery coordination report

Manifest snapshot: `2026-09-02T14:30:06Z`; final host check: `2026-09-02T14:31:31Z`

Scope: read-only reconciliation of `/home/exouser/ACTIVEFORCING_LIVE_STATUS` and the lane evidence it cites. This report was written only in `/home/exouser/FORTE_recovery_coordinator`. No shared main, Mass, TEST, Utility, GPU, Isaac, pi0, or training state was modified by this coordinator.

## Executive verdict

`AF_GPU_GATE=BLOCKED_DRIVER` remains the controlling admission state. The driver is unstable: `nvidia-smi` failed at `14:30:06Z`, succeeded at `14:31:21Z`, then failed again at `14:31:31Z`. The one successful sample showed 16,230 MiB allocated and 41% utilization, with Mass using 7,564 MiB and pi0 using 8,612 MiB. This is not a healthy two-query recovery and does not reopen admission.

The host is **not fail-closed in practice**:

- Mass Isaac collector PID `2126605` started at `14:21:01Z` and remains live.
- Frozen pi0 inference PID `2128032` started at `14:22:20Z` and remains live.
- E5 wait scheduler PID `2127549` started at `14:21:31Z`; it is armed to execute `core_gpu_scheduler.py` after Mass reports `COMPLETED` and a 10-second wait.
- All three processes have `AF_GPU_GATE` unset. The scheduler command contains no fresh driver-query or `AF_GPU_GATE` check.
- The registered recovery agents do carry `AF_GPU_GATE=BLOCKED_DRIVER`; ten corresponding new app-server runtimes were observed with that value. Older app servers and the detached tmux jobs do not.

This contradicts the live files' claims `pi0.running=false`, `new_gpu_workload_launched=false`, and `NO_SCIENTIFIC_WORKER_VISIBLE`. Those files were written at `14:24:35-36Z`, after the current Mass and pi0 start times, so the discrepancy is not explained by a later launch alone.

No process was stopped or signaled: Mass and pi0 are protected scientific processes, and stopping them was not explicitly authorized. The armed E5 scheduler is therefore an unresolved immediate admission risk.

## Reconciliation rules

1. The blocked-driver gate overrides every older `authorized_to_execute`, `READY_FOR_ISAAC`, template, and scheduler decision.
2. Only atomically complete, independently validated units count. A running process, partial telemetry, or a status value of `RUNNING` does not count.
3. Post-gate Mass output is not promoted by this audit. The accepted recovery boundary remains 10 complete contexts / 100 branch rows until independent QA is performed after the worker stops. The final readback had grown to 11 raw contexts / 110 raw branch rows, demonstrating active post-gate mutation; those extra rows remain unaccepted here.
4. Frozen Utility, TEST outcomes, shared main, and Mass outputs remain read-only.

## Reconciled lane state

| Lane | Accepted truth | Reconciled status | Immediate CPU-safe work | GPU admission dependency |
|---|---|---|---|---|
| E0 / pi0 | checkpoint 49999/config verified | runtime unexpectedly live; not an experiment result | provenance/report only | gate reopened before any restart or use |
| E1 | DEV complete; no sealed final closure | frozen partial | report/table prep only | fresh locked closure only if separately authorized |
| E2 | point physical-history OOF and partial modality evidence | partial / data-blocked | inventory missing folds and freeze matched split contract | only missing preregistered training after review |
| E3 | onboarding replay gates 5/5 PASS | assembly/norm/training/DEV not started | validate the five-demo manifest; prepare assembly/norm plan | explicit scientific authorization; assembly and norm QA first |
| E4 | 108 DEV shards; negative DEV conclusion | done and freeze | reporting only | none unless paper retains a locked confirmation |
| E5 | 21/60 tuples, 105/300 rollouts | incomplete; 39 tuples / 195 current-arm rollouts remain | reconcile missing-tuple and required-arm manifests | healthy driver + pi0 + explicit token + duplicate scan |
| E6 | belief infrastructure and legacy diagnostic only | no qualified scientific second query | specify genuine continuation and Utility-decision contract | qualified second-query evidence, then explicit token |
| E7 | Direct/offline gates PASS; result CSV header only | 0/8 exact-float gate and 0/144 matched real rollouts | validate runner/result schema | live interface parity + 8-rollout PASS before 144 block |
| Mass | 10 complete contexts / 100 branch rows accepted; 11/110 raw at final readback | interrupted/resumed process live; protocol still `RUNNING`; all post-gate additions unaccepted | read-only atomic-boundary/QA plan | healthy driver + explicit token; exclude/revalidate root8103-mid material |
| Boundary | 720 old branches + 0 new | CPU Phase 1 PASS; hypothesis not tested | manifest QA and preflight acceptance criteria | minimal force-interface preflight PASS before acquisition |
| Joint physics | collector/analyzer implementation only; no 3x3 data | dependency-blocked | freeze 3x3 schema and ablation plan | valid Mass closure and frozen mass axis |
| Tabero baseline | executor/checkpoint/smoke only | paired results missing | freeze paired tuple/telemetry contract | core lane priority and explicit token |
| FORTE baseline | reproduction audit only | implementation-blocked | build/audit faithful six-channel/impedance adapter | adapter PASS before any rollout |

## Manifest conflicts resolved

- E5: the old global audit's `20/60` is superseded by the later canonical coverage auditor at `21/60`; the same file says `completion_status=INCOMPLETE`. `status=PASS` means the audit ran, not that the campaign completed.
- E3: the global audit's `4/5` is superseded by the later demo19 gate and E3 status: demos `1,2,11,12,19` pass. `automatic_successor_allowed=false`; assembly, normalization, training, and post-training DEV remain not run.
- Mass: the live count of 100/180 agreed with the `14:30:06Z` CSV boundary (101 branch lines and 11 context lines, each including its header). By `14:31:21Z`, the running worker had appended one more raw context: 111 branch lines and 12 context lines. The protocol remains `RUNNING`; the extra 10 rows are post-gate and are not accepted by this report.
- E7: `0/152` must be represented as two gated stages, `0/8` then `0/144`; an older manifest's `authorized_to_execute=true` is stale under the blocked-driver gate.
- Boundary: `PASS_CPU_PHASE1` is not a collection or scientific PASS. The augmented manifest is `720 old + 0 new`, and training is false.
- Live process status: `RUNNING_WORKERS.csv` and `MASTER_LIVE_STATUS.json` are stale/incorrect relative to the host process table.

## Dependency-aware recovery queue

Nothing below authorizes a launch.

### Admission repair — required before all GPU work

1. Resolve the live-policy breach: obtain explicit authority to stop or otherwise neutralize PID `2127549` before it can auto-launch E5. Separately decide how the protected Mass/pi0 processes should be handled; this coordinator did not signal them.
2. Require a fresh successful device query. A single success should be followed by a 10-second quiet period and a second successful query.
3. Confirm zero duplicate Isaac/pi0/training workers, verify the intended output boundary, and issue a one-shot explicit admission token naming one lane and one command.
4. Clear the storage gate before output-generating work: `/home/exouser` is 100% full with about 419 MiB free. Do not delete evidence to make space.

### CPU queue while the gate remains blocked

1. E5: freeze the 39-tuple missing set and reconcile the current five arms against required Success-Only, same-transition Query-Ignored, Point, Posterior, and re-query comparisons.
2. Mass: prepare an immutable accepted-boundary receipt for the 100 rows; list every post-boundary telemetry file for later quarantine/revalidation.
3. E3: verify five-demo hashes and prepare, but do not run, assembly/normalization. LoRA is not an automatic successor.
4. E7 and Boundary: validate exact-float result schemas and the minimal force-interface acceptance contract.
5. E2/E6/Joint/baselines: complete evidence inventories, data contracts, and implementation QA that do not touch protected outcomes.
6. Prepare paper tables with pending cells explicitly marked `NA`/`GPU_BLOCKED`; do not infer results from offline decisions or smoke tests.

### GPU queue after admission is genuinely reopened

1. Mass completion and E5 continuation are the same priority tier but must be serialized. Mass-first shortens the dependency path to Joint and matches the interrupted atomic resume; E5-first prioritizes the broader submission blocker. The owner must choose the resource order explicitly.
2. E7 exact-float small block: exactly 8 DEV rollouts; proceed to 144 only if its execution audit passes.
3. E3 onboarding training only after five-demo assembly and CPU norm QA, and only with separate scientific authorization.
4. Boundary minimal force-interface preflight; collection remains blocked until it passes.
5. Mass downstream modeling/E2E only after 180-row QA. Any Utility mutation requires separate authority.
6. E6 true second-query pilot only after its continuation contract is qualified.
7. Joint 3x3 pilot only after valid Mass closure freezes the mass axis.
8. Tabero paired baseline after core blockers; FORTE only after its adapter passes.

## Evidence receipt

- Live status hash: `MASTER_LIVE_STATUS.json` SHA-256 `37a39de0e54f3725f1c5f1719de44596661ac7c326eed888f20d383b15ade6a4`.
- Live accepted counts hash: `64e1874ad6abef89a034de7f0ff68d37b0c2d5e32c5df16a411b28e0b9826d50`.
- E5 canonical coverage hash: `c5f89f1350e9baa7307a7f220802ecbf7e4cdec35ecc27d2dd3543b615855cad`.
- E3 status hash: `a0f368eee5861deec9226dc8157afb334a20a2d095b2edea4190d1015003aab8`.
- Mass protocol hash: `ba7d5e0fe068b5948cda4ba025b0f0a9f919b1a71cded826478f08aeac850c2d`.
- E7 result CSV hash: `2c89cf2717dd993dd5133952e2a70945e9c4482d886fb143cfe7b6d6e6eb4333` (header only).
- Boundary augmented manifest hash: `f80d283bdc7f2bcf111b2f346b7724c17eb527a0720d9d9f83e4d9cc268784a4`.
