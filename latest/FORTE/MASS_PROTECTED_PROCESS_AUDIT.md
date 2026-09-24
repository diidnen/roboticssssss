# MASS Protected Process Audit

- Audit timestamp: `2026-09-02T05:30:42Z`
- Protected PID: `465038`
- Parent PID: `4264`
- Working directory: `/home/exouser/Tabero`
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`
- Program: `/home/exouser/FORTE_mass/qualify_mass_tasks.py`
- Arguments: `--task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_TASK2_QUAL_RERUN2 --roots 7020 7021`
- GPU: NVIDIA A100-SXM4-40GB, PID memory observed `7592 MiB`
- System GPU snapshot: utilization `91%`, memory `29974/40960 MiB` (about `10.7 GiB` free)
- Status at audit: running normally; no signal, priority change, environment mutation, checkout change, or file mutation performed.

## Protected scope

The scheduler and E1/E3/E5 workers must not kill, renice, overwrite, reconfigure, reset, clean, or otherwise interfere with this process, `/home/exouser/FORTE_mass`, its output directory, `/home/exouser/Tabero`, its CUDA environment, or shared-memory resources.

GPU-heavy E3/E5 launches remain gated while GPU utilization is above 90% or free GPU memory is below 8 GiB. CPU/offline preparation may continue.

## Successor snapshot — 2026-09-02T05:54:03Z

- Previous protected PID `465038` exited naturally; no scheduler or worker signal was sent to it.
- New protected PID: `491191`
- Parent PID: `4264`
- Working directory: `/home/exouser/Tabero`
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`
- Program: `/home/exouser/FORTE_mass/qualify_mass_tasks.py`
- Arguments: `--task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_TASK2_QUAL_LOWGRID --roots 7020 7021`
- PID GPU memory observed: `7560 MiB`
- System GPU snapshot: utilization `39%`, memory `16261/40960 MiB` (about `24.1 GiB` free)
- Status at audit: running normally and now protected under the same non-interference rules.

## Successor snapshot — 2026-09-02T06:15:51Z

- Previous protected PID `491191` exited naturally; no scheduler or E1/E3/E5 signal was sent to it.
- New protected PID: `504869`
- Parent PID: `4264`
- Working directory: `/home/exouser/Tabero`
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`
- Program: `/home/exouser/FORTE_mass/qualify_mass_tasks.py`
- Arguments: `--task 3 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_TASK3_QUAL_LOWGRID_RERUN --roots 7030 7031`
- PID GPU memory observed during initialization: `3945 MiB`
- System GPU snapshot: utilization `45%`, memory `20250/40960 MiB`.
- Status at audit: running normally and protected under the same non-interference rules.

## Successor snapshot — 2026-09-02T06:43:30Z

- Previous protected PID `517976` exited naturally; no scheduler or E1/E3/E5 signal was sent to it.
- New protected PID: `520169`
- Parent PID: `4264`
- Working directory: `/home/exouser/Tabero`
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`
- Arguments: `--task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK2_RERUN2 --roots 7040 7041`
- System GPU snapshot at detection: utilization `92%`, free memory `16581 MiB`.
- Status at audit: running normally and protected under the same non-interference rules.

## Successor snapshot — 2026-09-02T06:41:11Z

- Previous protected PID `504869` exited naturally; no scheduler or E1/E3/E5 signal was sent to it.
- New protected PID: `517976`
- Parent PID: `4264`
- Working directory: `/home/exouser/Tabero`
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`
- Arguments: `--task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK2_RERUN --roots 7040 7041`
- PID GPU memory observed: `7560 MiB`
- System GPU snapshot: utilization `92%`, memory `23861/40960 MiB`.
- Status at audit: running normally and protected under the same non-interference rules.

## Natural exit snapshot — 2026-09-02T07:25:00Z

- Protected PID `520169` exited naturally after completing its structured task2 process lifetime.
- No scheduler or E1/E3/E5 signal, priority change, environment mutation, checkout change, or output mutation was applied to PID `520169`.
- No MASS successor was visible in the first post-exit process/GPU audit.
- E3 detected the absence at its next cell boundary and stopped fail-closed; its in-flight `TRAIN_root7400_mu1.0_F2N` cell had already completed and was preserved.
- The scheduler continues to watch for a MASS successor before resuming E3.

## Successor snapshot — 2026-09-02T07:27:10Z

- Previous protected PID `520169` had exited naturally; no scheduler or worker signal was sent to it.
- New protected PID: `537528`
- Parent PID: `4264`
- Working directory: `/home/exouser/FORTE_mass`
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`
- Arguments: `--task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK2_ROOT7040 --roots 7040 7040`
- PID GPU memory observed during initialization: `5411 MiB`
- System GPU snapshot at detection: utilization `80%`, memory `21007/40960 MiB` (about `19.0 GiB` free).
- Status at audit: running normally and protected under the same non-interference rules.

## Successor snapshot — 2026-09-02T08:01:20Z

- Protected PID `537528` exited naturally; no scheduler or worker signal was sent to it.
- New protected PID: `546417`
- Parent PID: `4264`
- Working directory: `/home/exouser/FORTE_mass`
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`
- Arguments: `--task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK2_ROOT7041 --roots 7041`
- PID GPU memory observed during initialization: `882 MiB`
- System GPU snapshot at detection: utilization `3%`, memory `9581/40960 MiB` (about `30.1 GiB` free).
- Status at audit: running normally and protected under the same non-interference rules.

## Successor snapshot — 2026-09-02T08:35:20Z

- Protected PID `546417` exited naturally; no scheduler or worker signal was sent to it.
- New protected PID: `555222`
- Parent PID: `4264`
- Working directory: `/home/exouser/Tabero`
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`
- Arguments: `--task 3 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK3_ROOT7042 --roots 7042`
- PID GPU memory observed: `6034 MiB`
- System GPU snapshot: utilization `86%`, memory `21519/40960 MiB` (about `18.5 GiB` free).
- Status at audit: running normally and protected under the same non-interference rules.

## Successor snapshot — 2026-09-02T08:59:30Z

- Protected PID `555222` exited naturally; no scheduler or worker signal was sent to it.
- New protected PID: `575814`
- Parent PID: `4264`
- Working directory: `/home/exouser/Tabero`
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`
- Arguments: `--task 6 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK6_ROOT7043 --roots 7043`
- PID GPU memory observed: `6748 MiB`
- System GPU snapshot: utilization `90%`, memory `21312/40960 MiB` (about `18.7 GiB` free).
- Status at audit: running normally and protected under the same non-interference rules.

## Natural exit snapshot — 2026-09-02T09:03:40Z

- Protected PID `575814` exited naturally; no scheduler or E1/E3/E5 signal was sent to it.
- No MASS successor was visible in the first post-exit process audit.
- E5 used the newly available GPU slot; E3 remained stopped because its MASS-presence guard fails closed while no protected process is visible.

## Successor snapshot — 2026-09-02T09:04:10Z

- Previous protected PID `575814` had exited naturally; no scheduler or worker signal was sent to it.
- New protected PID: `579378`
- Parent PID: `4264`
- Working directory: `/home/exouser/Tabero`
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`
- Arguments: `--task 0 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK0_ROOT7044 --roots 7044`
- PID GPU memory observed: `7560 MiB`
- System GPU snapshot: utilization `99%`, memory `28949/40960 MiB` (about `11.2 GiB` free).
- Status at audit: running normally and protected under the same non-interference rules.

## Successor snapshot — 2026-09-02T09:21:45Z

- Protected PID `579378` exited naturally; no scheduler or E1/E3/E5 signal was sent to it.
- New protected PID: `588271`
- Parent PID: `4264`
- Working directory: `/home/exouser/Tabero`
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`
- Arguments: `--task 7 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK7_ROOT7045 --roots 7045`
- PID GPU memory observed: `6780 MiB`
- System GPU snapshot: utilization `99%`, memory `35735/40960 MiB` (about `4.6 GiB` free).
- Status at audit: running normally and protected under the same non-interference rules. No additional GPU-heavy launch is permitted while the utilization/headroom gate remains closed.

## Protected rerun snapshot — 2026-09-02T09:24:00Z

- Protected PID `588271` exited before completion during a transient four-Isaac oversubscription. No scheduler signal was sent to MASS.
- MASS independently launched a protected rerun successor: PID `590632`, parent PID `4264`.
- Working directory: `/home/exouser/Tabero`.
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`.
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`.
- Arguments: `--task 7 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK7_ROOT7045_RERUN --roots 7045`.
- System GPU snapshot after resource fail-closed cleanup: utilization `36%`, memory `15483/40960 MiB` (about `24.4 GiB` free).
- Status at audit: running normally and protected. The incomplete predecessor is not modified; the MASS-owned rerun directory is untouched.

## Successor snapshot — 2026-09-02T09:49:35Z

- Protected PID `590632` exited naturally after its task7 rerun lifetime; no scheduler or E1/E3/E5 signal was sent to it.
- New protected PID: `603571`, parent PID `4264`.
- Working directory: `/home/exouser/Tabero`.
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`.
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`.
- Arguments: `--task 0 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK0_ALT_ROOT7046 --roots 7046`.
- PID GPU memory observed: `7560 MiB`.
- System GPU snapshot at detection: utilization `94%`, memory `24084/40960 MiB` (about `16.0 GiB` free), with one already-queued E5 shard and one pre-frozen E3 DEV cell active.
- Status at audit: running normally and protected. All additional launches are frozen until utilization drops below the configured gate.

## Successor snapshot — 2026-09-02T10:16:25Z

- Protected PID `603571` exited naturally after its task0 ALT process lifetime; no scheduler or E1/E3/E5 signal was sent to it.
- New protected PID: `616079`, parent PID `4264`.
- Working directory: `/home/exouser`.
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`.
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`.
- Arguments: `--task 8 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK8_ROOT7047 --roots 7047`.
- PID GPU memory observed during initialization: `3959 MiB`.
- System GPU snapshot at detection: utilization `31%`, memory `16699/40960 MiB` (about `23.2 GiB` free). E5 and an E3 reserve retry had already passed their low-load gates before this MASS successor became visible.
- Status at audit: running normally and protected. No further GPU-heavy launch is permitted while the three existing workloads are active.

## Successor snapshot — 2026-09-02T10:21:05Z

- Protected PID `616079` exited naturally after its task8 process lifetime; no scheduler or E1/E3/E5 signal was sent to it.
- New protected PID: `619215`, parent PID `4264`.
- Working directory: `/home/exouser/Tabero`.
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`.
- Program: `/home/exouser/FORTE_mass/qualify_mass_structure.py`.
- Arguments: `--task 9 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK9_ROOT7048 --roots 7048`.
- PID GPU memory observed: `6066 MiB`.
- System GPU snapshot at detection: utilization `90%`, memory `25624/40960 MiB` (about `14.5 GiB` free), with the existing E3 reserve retry and E5 shard already active.
- Status at audit: running normally and protected. No additional GPU-heavy launch is permitted.

## Formal-collection successor snapshot — 2026-09-02T10:48:28Z

- Protected PID `619215` exited naturally; no scheduler or E1/E3/E5 signal was sent to it.
- New protected PID: `645609`, parent PID `4264`.
- Working tree/program family: `/home/exouser/FORTE_mass`.
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`.
- Program: `/home/exouser/FORTE_mass/collect_mass_structured_formal.py`.
- Arguments: `--task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_112000 --train-roots 8100 8101 8102 8103 --test-roots 8200 8201 --repeats 2`.
- PID GPU memory observed: `7560 MiB`.
- System GPU snapshot: utilization `85%`, memory `23212/40960 MiB` (about `17.3 GiB` free), with the existing E5 shard active.
- Status at audit: running normally and protected. E3 replay and training remain resource-gated.

## Formal-collection restart snapshot — 2026-09-02T10:50:42Z

- Protected PID `645609` exited naturally; no scheduler or E1/E3/E5 signal was sent to it.
- New protected PID: `647777`, parent PID `4264`.
- Working directory/program: `collect_mass_structured_formal.py` under the MASS-owned checkout.
- Arguments: `--task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_114500 --train-roots 8100 8101 8102 8103 --test-roots 8200 8201 --repeats 2`.
- PID GPU memory observed: `7692 MiB`.
- System GPU snapshot: utilization `50%`, free memory `24048 MiB`, with no E3 or E5 Isaac worker active at the instant of audit.
- Status at audit: running normally and protected. Only one additional heavy workload is allowed under the scheduler rule.

## MASS-owner optimization transition — 2026-09-02T13:00:54Z

- Protected PID `647777` disappeared after the MASS-owned protocol was updated to
  `INTERRUPTED_RESOURCE_OPTIMIZATION_PARTIAL_NOT_FOR_FINAL_TOTALS` at 68 branches.
- The MASS-owned note states that its own worker was interrupted to remove unused
  camera rendering overhead. The E1/E3/E5 coordinator did not signal, modify, or
  restart the process and makes no natural-exit claim.
- MASS owner then launched successor PID `2075675` with output
  `M3_TASK2_FORMAL_STRUCTURED_20260902_130500_NOCAM`; it was protected on detection.
- PID `2075675` exited immediately. Its own protocol records `status: ERROR` and
  `RuntimeError('A camera was spawned without the --enable_cameras flag...')`.
- No output was deleted or overwritten. Both partial/error directories remain
  MASS-owned and are excluded from E1/E3/E5 evidence.
- At the first post-error audit no MASS successor was visible. Any later successor
  remains protected under the same non-interference rules.

## MASS-owner camera successor — 2026-09-02T13:02:45Z

- New protected PID: `2076670`, parent PID `4264`.
- Working directory: `/home/exouser/Tabero`.
- Python: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python`.
- Program: `collect_mass_structured_formal.py`.
- Output: `/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_130600_CAM`.
- Arguments retain task2, TRAIN roots 8100--8103, TEST roots 8200--8201,
  and two repeats.
- PID GPU memory observed: `7560 MiB`; system snapshot with one pre-existing E5
  worker: utilization `88%`, free memory `17390 MiB`.
- Status at audit: running normally and protected. The coordinator did not launch,
  signal, reconfigure, or modify this successor.

## MASS-owner resume successor — 2026-09-02T13:08:07Z

- PID `2076670` disappeared while its own protocol still contained `RUNNING`;
  the coordinator therefore makes no clean/natural-exit claim and did not modify
  the stale MASS-owned status.
- New protected PID: `2080255`, parent PID `4264`.
- Working directory: `/home/exouser/FORTE_mass`.
- Program: `collect_mass_structured_formal.py`, task2, the same TRAIN/TEST roots
  and repeats, with `--resume-from .../M3_TASK2_FORMAL_STRUCTURED_20260902_114500`.
- New timestamped output:
  `M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME`.
- GPU memory at early initialization was `584 MiB`; it shares the host with one
  pre-existing E5 worker and the frozen pi0 server.
- Status at audit: running and protected. Resume semantics and any branch reuse
  remain solely MASS-owned; E1/E3/E5 neither read nor modify its results.

## Protected concurrency snapshot — 2026-09-02T13:28:55Z

- Protected MASS PID `2080255` remains alive with its unchanged task2 resume
  command and MASS-owned checkout/output.
- After E3 demo12 replay had naturally flushed and passed producer QA, the core
  scheduler admitted one E5 task1 offset03 worker under the frozen repeated gate.
- System snapshot after that E5 worker initialized: GPU utilization `93%`, memory
  used `23177 MiB`, memory free `17264 MiB`.
- Because utilization exceeded 90%, no further E3/E5 heavy launch is permitted
  until the active worker exits and the repeated gate is re-evaluated.
- No signal, priority change, environment mutation, shared-memory deletion, file
  write, or checkout/config change was applied to MASS.
