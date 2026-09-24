# Mass E2E engineering fix log

- Confirmed root cause: `train_direct.predict(masses, forces)` iterates with `zip`; the old fresh adapter passed a singleton mass array with five forces, producing one probability and selecting only 0.5 N.
- Fixed the adapter to pass `np.full(len(FORCES), mass)` for Active, Prior, and GT-Mass scoring.
- Fixed inherited fresh telemetry metadata by setting `CANDIDATES_PER_TASK=5` for task 2, matching the formal `[0.5, 1.0, 1.5, 2.5, 4.0]` candidate set.
- Added persistence of raw query record and all candidate p/U rows for future attribution.
- No Mass raw recollection, π0 change, controller-law change, Utility-formula change, evaluator change, or TEST tuning.
- Corrected same-six-context Fresh rerun completed on the six canonical contexts with the repaired five-candidate scorer. All 30 corrected rows and six persisted query records passed lineage checks; no protected worker or π0 server was interrupted.
- This attribution pass uses only the corrected rows. The old singleton-candidate rollouts are excluded from all conclusions.
- No additional force-sweep/repeat was launched: without re-query/state snapshots, such a rollout would not be an exact paired replay. Existing corrected Active/GT arms are used as one-repetition paired force evidence.
- Fresh identifier parity audit completed: all 12 required physical features are present, metres/mm fields match the frozen feature contract, and recomputation agrees with persisted predictions to 0.0 kg. The 0.41--0.55 kg output is a real fresh feature-range/calibration shift, not a preprocessing plumbing mismatch.
- A query-state Fixed-4N diagnostic was launched in a separate output directory. It waits for protected Isaac workers to finish naturally before entering the simulator; no signal is sent to protected tasks.
