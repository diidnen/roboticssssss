# E3 task5 Fmax identity-collision audit

Verdict: `PRIOR_FMAX_BOUNDARY_INFERENCE_INVALID_RAW_LABELS_PRESERVED`

The authoritative Utility configuration maps numeric task IDs only within the current E1 four-task archive. Its own protocol says the mapping is “for the current four-task archive.” The E1 fresh runner resolves task5 to `tomato_sauce_1`, with the basket instruction in the `libero_object` family.

E3's candidate uses `libero_10/task5`, resolves to `black_book_1`, and requires placing the book in the back compartment of a caddy. LIBERO task IDs are suite-local. Equality of the integer `5` does not establish equality of semantic task, object, safety range, candidate grid, controller support, or Utility normalization.

Consequences:

- The completed 15-cell 1--5 N table is valid branch-level evidence: 10 LocalLift failures, 5 LocalLift-success/downstream-failure rows, 0 FullTask successes.
- The earlier `FAIL_CLOSED_NO_IN_DOMAIN_FULL_TASK_SUCCESS` interpretation is invalid because no authoritative long-horizon task5 Fmax had been established.
- The 8 N nominal qualification remains qualification-only and cannot define Fmax or enter a Utility table.
- A TRAIN-only same-root 6/7/8 N label-support extension is scientifically allowed because it is motivated by the pre-existing robust qualification, not by ActiveForcing results. It does not use TEST and does not modify the existing Utility hash.
- Task5 DEV, classifier fitting, and any Utility selector evaluation remain prohibited until the TRAIN label-support gate passes and a separate task-specific Fmax protocol is frozen where needed.

Evidence:

- `/home/exouser/FORTE/ACTIVEFORCING_AUTHORITATIVE_PROTOCOL.md`: Fmax mapping explicitly scoped to the current four-task archive.
- `/home/exouser/FORTE/run_activeforcing_fresh_e2e.py`: `OBJECTS[5] = "tomato_sauce_1"`.
- `/media/volume/newdata/exouser/Tabero_e3lh/benchmarks/datasets/libero/config/libero_10.json`: task5 is `black_book_1` -> `desk_caddy_1`.
- E1 Utility config hash remains `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`; it is unchanged.

Original pre-correction report/status hashes are preserved in `E3_TASK5_FMAX_CORRECTION_PROVENANCE.json`.
