# Current four-task low-force audit

Prepared: `2026-09-02T09:24:48.059710+00:00`

Authoritative archive: `/home/exouser/FORTE/activeforcing_final_closure_20260902_034923/E3_BRANCH_LABEL_AUDIT.csv` (720 branches).

No existing archive cell is scheduled for recollection. Candidate selection uses only the physical lift boundary, full-task outcomes, and the controller-safe nonnegative force domain; Utility and ActiveForcing performance are excluded.

| Task | Existing support (N) | Lift failures | Full-task failures | Full-task successes | Estimated FullTask transition | New pilot forces (N) |
|---:|---:|---:|---:|---:|---|---|
| 0 | 3.068–4.991 | 0 | 42 | 138 | failure edge 3.068–4.423; success edge 3.191–4.423 N across 11 contexts | 1.5, 2.0, 2.25 |
| 1 | 4.023–5.985 | 0 | 59 | 121 | failure edge 4.155–5.454; success edge 4.681–5.960 N across 7 contexts | 2.0, 2.5, 3.0 |
| 5 | 3.006–4.993 | 0 | 38 | 142 | failure edge 3.197–4.814; success edge 3.404–4.814 N across 8 contexts | 1.5, 2.0, 2.25 |
| 6 | 3.002–3.995 | 0 | 6 | 174 | failure edge 3.002–3.395; success edge 3.002–3.129 N across 5 contexts | 1.5, 2.0, 2.25 |

## Resource gate

Each simulator launch requires GPU utilization <=70%, at least 10 GiB free, a visible healthy Mass process, the full E5 campaign completion gate (inter-shard gaps stay reserved), and no conflict with the E3 long-horizon lane. A failed check exits without launching Isaac.

## Telemetry contract

Every collected row is finalized with: `task`, `root_id`, `friction`, `force_N`, `repeat`, `initial_state_hash`, `query_state_hash`, `grasp_success`, `lift_success`, `lift_height_m`, `lift_hold_duration_s`, `transport_retention`, `placement_success`, `full_task_success`, `drop_timestamp_s`, `failure_stage`, `failure_reason`, `commanded_grip_force_N`, `measured_grip_force_N`, `peak_force_N`, `tracking_error_N`.

Status at preparation: `PENDING_RESOURCE_GATE`.
