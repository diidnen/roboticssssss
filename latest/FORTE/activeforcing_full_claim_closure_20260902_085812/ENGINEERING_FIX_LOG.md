# Engineering fix log

- Fixed E5 adapter unpacking: the authoritative `run_probe_no_reset` returns `(probe_rows, probe_record)`; post-query state is captured separately.
- Added a full-run-only VLA chunk-budget override (`AF_VLA_MAX_CHUNKS`, default 50) without changing π0, nominal motion, or controller law. The interrupted smoke used the historical 22-chunk default.
- E6/E7 was rerun with single-thread CPU execution after the first offline runner exited after member_0; final run completed with 3 members and 216 planner rows.
- Current task0 full retry was stopped by an Isaac `carb.tasking::Mutex` assertion while other known simulator workers were active. Three arm rows and telemetry were preserved; this is an execution-concurrency failure, not a scientific-method change. The retry remains partial until rerun with an exclusive Isaac worker.
