# ActiveForcing E6/E7 Existing Work Audit

## Scope

Audit performed over `/home/exouser/FORTE`, `/home/exouser/Tabero`, and the archived `analysis/results` namespaces. Original TEST rows were not loaded by this lane.

| Component | Status | Evidence / boundary |
|---|---|---|
| 3-member physical belief | READY (DEV diagnostic) | New matched architecture, three deterministic seeds, root-bootstrap TRAIN resampling; DEV calibration artifact emitted by this run. |
| Heteroscedastic uncertainty | READY (DEV diagnostic) | Per-member `mu` and `sigma`; uncertainty semantics and TRAIN-only interval scaling are recorded. |
| Decision disagreement | READY (DEV diagnostic) | Member-induced force decisions and model-independent harmful frontier comparison emitted. |
| Validated second query | FAILED_DEV | Historical P7-B pilot failed query qualification; no authoritative second-query continuation was found. |
| Decision-aware re-query | FAILED_DEV | Protocol is frozen, but four-policy scientific comparison is not evaluable without valid second-query evidence. |
| Continuous Direct | FAILED_DEV | Prior GNP-style continuous line reports `CONTINUOUS_FEASIBILITY_STILL_NOT_VALIDATED`; this run preserves that boundary and rechecks the DEV diagnostic. |
| Candidate generators | READY (offline DEV diagnostic) | Fixed/uniform/stratified/proposal interfaces implemented with matched K=5/10/20. |
| Posterior-aware planning | READY (offline DEV diagnostic) | Same Direct backend and candidates; point-estimate ablation retained. |
| rho selection | READY | Candidate set and DEV-only reliability/mean-force selection frozen in `FINAL_RHO.json`. |
| Fallback | READY | Max safe force within task bounds, frozen before any TEST orchestration. |
| Real simulator continuous validation | FAILED_DEV | Not launched after failed continuous gate; no arbitrary-setpoint validation is claimed. |

## Reused historical conclusions

The pre-probe feasibility line had three seeds and TRAIN isotonic calibration, but it was not a physical posterior ensemble. The continuous GNP-style line had 720 continuous TRAIN branches and a 9-context DEV diagnostic, but its safety-first gate failed. The old P7-B pilot had one-query infrastructure only and failed qualification. None of these artifacts authorized locked TEST method tuning.

## Overall audit classification

E6: PARTIAL -> COMPLETE_NEGATIVE because physical ensemble/disagreement are now available but validated second-query evidence is absent.

E7: PARTIAL -> COMPLETE_NEGATIVE because offline candidate/posterior planning and freeze artifacts are available, while continuous Direct reliability and real arbitrary-force simulator validation are not established.
