# Historical long-horizon task search audit

Search covered `/home/exouser/FORTE`, `/home/exouser/Tabero`, both git histories, LIBERO configs/assets, and prior result plans. The three explicitly recalled tasks are present in project-local configs:

- `libero_10/task3`: black bowl → bottom drawer → close. Two independent final goals.
- `libero_10/task9`: yellow/white mug → microwave → close. Two independent final goals.
- `libero_10/task5`: book → back compartment of caddy.

Additional history-backed candidates are recorded in the CSV. A dedicated `PutIntoAndCloseDrawerEnvCfg` and final `obj_is_into_drawer_and_drawer_is_closed` evaluator also exist in the repository.

No local assembled HDF5/demo file for task3/task5/task9 was found under `/home/exouser` or `/media/volume/newdata/exouser`. Qualification therefore uses default resets first. Missing demos are not silently guessed or replaced. Few-demo onboarding remains a gated fallback only after nominal-capability qualification fails.
