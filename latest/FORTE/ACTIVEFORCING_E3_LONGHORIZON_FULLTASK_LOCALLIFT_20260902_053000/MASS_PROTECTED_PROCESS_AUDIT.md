# MASS protected process audit

- Audit time: `2026-09-02T05:30:12Z`
- Protected host PID: `465038`
- Command: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python /home/exouser/FORTE_mass/qualify_mass_tasks.py --task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_TASK2_QUAL_RERUN2 --roots 7020 7021`
- Working lane inferred from the absolute command/output paths: `/home/exouser/FORTE_mass`.
- GPU memory at audit: `7592 MiB` for this process.
- Whole-GPU snapshot: `92–95%` utilization, `29974 MiB` used, `10467 MiB` free.

Protection contract: E3 will not kill, renice, signal, alter CUDA visibility, modify the MASS checkout/config/output, delete shared memory, or reuse its output paths. No E3 GPU-heavy process is allowed while the scheduler gate is closed.

## Protected successor observed at 2026-09-02T05:53:20Z

- Original PID `465038` was no longer present; E3 did not signal or alter it.
- New protected MASS PID: `491191`.
- Command: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python /home/exouser/FORTE_mass/qualify_mass_tasks.py --task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_TASK2_QUAL_LOWGRID --roots 7020 7021`.
- GPU memory at observation: `7560 MiB`.
- This successor inherits the full protection contract above. E3 did not launch because E5 PID `488892` and MASS PID `491191` already occupied the GPU.

## Protected successor observed at 2026-09-02T06:15:37Z

- PID `491191` was no longer present; E3 did not signal or alter it.
- New protected MASS PID: `504869`.
- Command: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python /home/exouser/FORTE_mass/qualify_mass_tasks.py --task 3 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_TASK3_QUAL_LOWGRID_RERUN --roots 7030 7031`.
- It was still loading at the first observation (`552 MiB`); E3 remained stopped while E5 PID `501731` occupied the current heavy slot.
- This successor inherits the full protection contract.

## Protected successor observed at 2026-09-02T06:40:13Z

- PID `504869` had exited naturally; E3 did not signal or alter it.
- New protected MASS PID: `517976`.
- Command: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python /home/exouser/FORTE_mass/qualify_mass_structure.py --task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK2_RERUN --roots 7040 7041`.
- The process used `7560 MiB` at the read-only GPU snapshot and ran from the protected `/home/exouser/FORTE_mass` lane.
- E3's prior pilot cell PID `515506` had already exited, and its fail-closed runner did not launch another cell. E3 did not touch the new MASS process and yielded the available experiment slot to E5.
- This successor inherits the full protection contract.

## Protected successor observed at 2026-09-02T06:45:31Z

- PID `517976` had exited naturally; E3 did not signal or alter it.
- New protected MASS PID: `520169`.
- Command: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python /home/exouser/FORTE_mass/qualify_mass_structure.py --task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK2_RERUN2 --roots 7040 7041`.
- GPU memory was `7560 MiB`; with E5 concurrently active, whole-GPU utilization was `92%` and free memory was `16581 MiB`.
- E3 remained paused because the utilization gate was closed. This successor inherits the full protection contract.

## Protected successor observed at 2026-09-02T07:30:10Z

- PID `520169` had exited naturally; E3 did not signal or alter it.
- New protected MASS PID: `537528`.
- Command: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python /home/exouser/FORTE_mass/qualify_mass_structure.py --task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK2_ROOT7040 --roots 7040 7040`.
- GPU memory was `7560 MiB`; with E5 concurrently active, whole-GPU utilization was `90%` and free memory was `16548 MiB`.
- E3 remained paused at 12/15 TRAIN cells. This successor inherits the full protection contract.

## Protected successor observed at 2026-09-02T08:01:19Z

- PID `537528` had exited naturally; E3 did not signal or alter it.
- New protected MASS PID: `546417`.
- Command: `/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python /home/exouser/FORTE_mass/qualify_mass_structure.py --task 2 --out /home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M2_STRUCTURED_TASK2_ROOT7041 --roots 7041`.
- GPU memory was `7560 MiB`; E3 ran only one additional preregistered cell at a time alongside this process.
- This successor inherits the full protection contract.
