# Engineering fix log

- Fixed E5 adapter unpacking: the authoritative `run_probe_no_reset` returns `(probe_rows, probe_record)`; post-query state is captured separately.
- Added a full-run-only VLA chunk-budget override (`AF_VLA_MAX_CHUNKS`, default 50) without changing π0, nominal motion, or controller law. The interrupted smoke used the historical 22-chunk default.
- E6/E7 was rerun with single-thread CPU execution after the first offline runner exited after member_0; final run completed with 3 members and 216 planner rows.
