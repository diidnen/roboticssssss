# Mass E2E lineage audit

## Authoritative inputs

- Formal Mass source: `/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_AND_TASK_BREADTH_EXTENSION_20260902_041500/M3_TASK2_FORMAL_STRUCTURED_20260902_130700_RESUME`
- QA status: `PASS`, 180/180 accepted branches, 18/18 contexts, 3,708 query timesteps.
- Fresh source audited: `fresh_e2e_corrected_final/`, 30 corrected rows over roots 8400 and 8401, six root×mass contexts; this supersedes the old singleton-candidate bundle.
- Formal task: task 2, `salad_dressing_1`, friction fixed at 0.5, mass bands 0.05/0.10/0.20 kg.

## Runtime lineage

| Component | Lineage |
|---|---|
| Frozen π0 | `pi0_lora_tacfield_tabero`, step 49999; checkpoint SHA-256 `0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17`; websocket `127.0.0.1:18881`; replan 10; effective action dim 13 |
| Query | frozen P4B, one query per query arm, post-query state continuation |
| Mass identifier | Physical-History-only, TRAIN roots 8100–8103, seeds 11/23/37, reproducible compact MLP in `mass_modeling_final.py`; no fresh TEST fitting |
| Direct | Mass-only `z=mass` with `[mass, force, mass×force, force²]`, full-task labels, same TRAIN branches, seeds 11/23/37 |
| Utility | `p_success*(8-force)+(1-p_success)*(-1)`, lower-force tie break; config hash `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10` |
| Candidate support | formal Mass grid `[0.5, 1.0, 1.5, 2.5, 4.0] N`; corrected fresh adapter scores all five candidates |
| Controller | inherited P6G1R1 frozen low-level execution and force-slot transform; no controller-law change |
| Evaluator | inherited full-task success, lift, transport retention, placement, drop, termination telemetry; no evaluator rewrite |

## Finding

The original fresh adapter had a deterministic Direct invocation bug: it passed one mass value against five forces to a predictor implemented with `zip(masses, forces)`. Consequently only force 0.5 N was scored and Active, GT-Mass, and Prior all selected 0.5 N. This is an engineering failure, not evidence that GT-Mass cannot select a useful force. The corrected adapter expands mass to one value per candidate and persists query records and candidate p/U rows.

The corrected query-only pass persisted one raw query record per context. Exact equality to formal query states is still not asserted because the fresh and formal telemetry schemas differ; root/query hashes, reset parity, and proxy comparisons are retained.

The corrected same-six-context result is the authoritative fresh E2E result: Active=4/6, GT-Mass=2/6, Fixed-Max=1/6, Prior=0/6, Native=0/6. Attribution is `MASS_E2E_MIXED_FAILURE`: force selection explains the two MID Active-vs-GT wins, while both HIGH contexts fail downstream under the observed force support.
