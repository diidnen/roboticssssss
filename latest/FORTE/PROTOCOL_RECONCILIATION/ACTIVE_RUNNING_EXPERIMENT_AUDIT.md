# Active running experiment audit

Initial audit snapshot: `2026-09-02T04:49:17Z`; final refresh: `2026-09-02T04:53:14Z`. Initial GPU load was 30,753 MiB and 97%; final refresh was 30,755 MiB and 84%. No additional heavy GPU job is authorized because four long-lived GPU processes still occupy the device and no missing run is ready to launch. No tmux or screen session was present.

## Classification summary

| PID | Run | Selector | Class | Disposition |
|---:|---|---|---|---|
| 33931 | frozen π0 server | N/A | A | Continue; shared backend only |
| 37328 | E5 smoke `041520` | min force with `p>=0.5` | B | Let finish; engineering diagnostic only |
| 44384 | mass task0 qualification | N/A | A | Continue; selector-independent qualification |
| 45847 | E5 smoke `043100` | min force with `p>=0.5` | B | Let finish; engineering diagnostic only |
| 288330 | E6/E7 closure | min force with `p>=rho`; DEV rho sweep | C | Completed during audit; reuse belief only after QA |
| 290934 | mass figure renderer | N/A | A | Completed during audit; development figure only |
| 292875 | duplicate E6/E7 closure | min force with `p>=rho`; DEV rho sweep | C | Running at final refresh; no further duplicate launches |
| 295222 | duplicate E6/E7 closure | min force with `p>=rho`; DEV rho sweep | C | Running at final refresh; no further duplicate launches |

The complete required fields are in `ACTIVE_RUNNING_EXPERIMENT_AUDIT.csv`.

## E5 determination

Both E5 processes load `run_two_method_formal.DirectRuntime`. Its `decide()` scores the grid `[3.00, 3.25, ..., 5.00]`, selects the first candidate with ensemble `raw_probability >= 0.5`, and otherwise uses 5 N. Decision JSONs explicitly record `threshold: 0.5`. This is the revoked hard-threshold selector, not expected Utility.

The runs are not killed. They remain useful for reset, query, state parity, π0 connectivity, low-level force setpoint, telemetry, and end-to-end engineering validation. They must never be cited as final E5 controller evidence. Their corrected rerun is scheduled after GPU capacity is available and after the V2 Utility freeze is complete.

## E6/E7 determination

The closure trains three belief members on TRAIN and evaluates DEV without loading original TEST. That part is compatible and reusable. Its planner, however, selects the minimum candidate crossing `rho`, tunes `rho` from `[0.80,0.85,0.90,0.95]`, defines decision disagreement through those decisions, and uses a `p>=0.8` boundary in proposal guidance. All selector-dependent E6/E7 outputs are diagnostic. Final E6/E7 must use `argmax_F E_z[U(F|z)]` and disagreement among member-wise Utility argmax decisions. Two identical CPU closures (PIDs 292875 and 295222) were concurrently active at final refresh; they are short legacy duplicates, so no additional copy should be launched.

## Mass determination

The active mass job is qualification-only: task0, roots 7000/7001, fixed friction 0.5, masses 0.05/0.10/0.20 kg, forces 3/4/5/6/8 N, scripted branch controller, and no learned selector. It is unaffected by the selector correction. It establishes only task/branch suitability, not final mass-aware ActiveForcing performance.

## TEST and safety

None of the active runs loads a locked TEST split: E5 is explicitly `SMOKE`; mass is `QUAL`; E6/E7 filters to TRAIN/DEV. No active run is class D.

Separately, the historical root-scaling task0 TEST set is already observed: roots 5174–5183, 10 contexts, 450 branches, with 318 successes and 132 failures. `TASK0_TEST_SHARD_HANDOFF.json` records evaluation complete. It is not fresh for V2 Utility and is quarantined from final locked-test claims.

## GPU policy

Do not launch a new heavy GPU job while utilization exceeds 85%. Priority after a slot frees: (1) corrected Utility E5 smoke on DEV-only roots, (2) corrected E6/E7 CPU rescoring where possible, (3) only then genuinely missing simulator validation. Do not duplicate the 720 friction branches, mass collection, belief ensemble, or continuous training.
