# MASS FRESH E2E ANALYSIS

Scope: the six corrected fresh contexts only. No Mass branch recollection, no new query, no π0/controller/evaluator/Utility change, and no new force rollout was used in this report. The corrected existing Active and GT arms are already paired at the same persisted query-state hash and therefore provide the allowed one-repetition force-swap evidence.

## 1. Six-context table

See `MASS_FRESH_E2E_SIX_CONTEXT_TABLE.csv`. Corrected outcomes are Active=4/6, GT=2/6, Fixed-Max=1/6, Prior=0/6, Native=0/6.

| Context | GT kg | Active pred kg | Error kg | Active F | GT F | Prior F | Fixed F | Active | GT | Prior | Fixed | Native | Active stage | GT stage |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|
| mass_fresh_t2_root8400_high | 0.20 | 0.5213 | +0.3213 | 4.0 | 2.5 | 1.5 | 4.0 | placement | placement | placement | placement | placement | placement | placement |
| mass_fresh_t2_root8400_low | 0.05 | 0.4128 | +0.3628 | 4.0 | 1.5 | 1.5 | 4.0 | SUCCESS | SUCCESS | placement | SUCCESS | vla_timeout_or_error | NONE | NONE |
| mass_fresh_t2_root8400_mid | 0.10 | 0.4277 | +0.3277 | 4.0 | 1.5 | 1.5 | 4.0 | SUCCESS | transport | placement | placement | vla_timeout_or_error | NONE | transport |
| mass_fresh_t2_root8401_high | 0.20 | 0.5494 | +0.3494 | 4.0 | 2.5 | 1.5 | 4.0 | transport | transport | placement | placement | placement | transport | transport |
| mass_fresh_t2_root8401_low | 0.05 | 0.4378 | +0.3878 | 4.0 | 1.5 | 1.5 | 4.0 | SUCCESS | SUCCESS | placement | transport | placement | NONE | NONE |
| mass_fresh_t2_root8401_mid | 0.10 | 0.4295 | +0.3295 | 4.0 | 1.5 | 1.5 | 4.0 | SUCCESS | transport | placement | placement | placement | NONE | transport |

## 2. Why Active beats GT

The advantage is force selection, not better mass accuracy. Both winning cases are MID:

- `root8400_mid`: Active predicted mass 0.4277 kg -> 4.0 N; GT mass 0.10 kg -> 1.5 N. Active succeeded; GT failed at transport.
- `root8401_mid`: Active predicted mass 0.4295 kg -> 4.0 N; GT mass 0.10 kg -> 1.5 N. Active succeeded; GT failed at transport.

The Active predicted mass is strongly high-biased in all six contexts (+0.3277 to +0.3878 kg). That bias drives the Direct/Utility model to the conservative 4.0 N arm. It is not evidence that the estimator is more accurate; it is a task-specific conservative extrapolation beyond the Direct training mass support (.05/.10/.20 kg). GT-Mass is in the Direct training support and is semantically matched to the training labels.

## 3. Direct and Utility scores for the two Active wins

The score tables below use the frozen TRAIN-only Mass Direct ensemble and the unchanged Expected Utility `p*(8-F)+(1-p)*(-1)`. Values are `(p_success, utility)`.

| Context | Input | 0.5N | 1.0N | 1.5N | 2.5N | 4.0N | selected |
|---|---|---:|---:|---:|---:|---:|---:|
| mass_fresh_t2_root8400_mid | Active predicted mass | 0.041,-0.650 | 0.055,-0.561 | 0.074,-0.446 | 0.238,0.548 | 0.999,3.996 | 4.0N |
| mass_fresh_t2_root8400_mid | GT mass | 0.057,-0.518 | 0.671,4.370 | 0.973,6.297 | 0.974,5.330 | 0.800,3.000 | 1.5N |
| mass_fresh_t2_root8401_mid | Active predicted mass | 0.041,-0.650 | 0.055,-0.560 | 0.074,-0.445 | 0.233,0.517 | 0.999,3.996 | 4.0N |
| mass_fresh_t2_root8401_mid | GT mass | 0.057,-0.518 | 0.671,4.370 | 0.973,6.297 | 0.974,5.330 | 0.800,3.000 | 1.5N |

## 4. Force-swap results

These are existing corrected paired fresh arms, not new rollouts. Active and GT share the same query-state hash in both cases.

| Context | Active force | Active result | GT force | GT result | Same query state |
|---|---:|---|---:|---|---|
| mass_fresh_t2_root8400_high | 4.0N | 0/1 (placement) | 2.5N | 0/1 (placement) | 1 |
| mass_fresh_t2_root8400_low | 4.0N | 1/1 (SUCCESS) | 1.5N | 1/1 (SUCCESS) | 1 |
| mass_fresh_t2_root8400_mid | 4.0N | 1/1 (SUCCESS) | 1.5N | 0/1 (transport) | 1 |
| mass_fresh_t2_root8401_high | 4.0N | 0/1 (transport) | 2.5N | 0/1 (transport) | 1 |
| mass_fresh_t2_root8401_low | 4.0N | 1/1 (SUCCESS) | 1.5N | 1/1 (SUCCESS) | 1 |
| mass_fresh_t2_root8401_mid | 4.0N | 1/1 (SUCCESS) | 1.5N | 0/1 (transport) | 1 |

Interpretation: both Active-win cases show 1/1 at 4.0 N versus 0/1 at 1.5 N. This is direct one-pair evidence that the selected higher force was beneficial in those contexts. It is not a 3-repeat variance estimate; no re-query/new rollout was run under the current constraint.

## 5. Existing corrected force evidence

The authoritative fresh candidate support is 0.5/1.0/1.5/2.5/4.0 N, not 5--8 N. The table `MASS_FRESH_E2E_FORCE_SWEEP.csv` reports every force actually observed in the corrected six-context arms; no new 3--8 N sweep was run because that would require new query/state replay and would violate this round's no-re-query constraint.

For both HIGH contexts, the observed 1.5 N, 2.5 N, and 4.0 N arms all failed (transport/placement). Thus they are classified `force_not_rescuing_observed_support`: the current fresh state/π0 downstream execution is the dominant issue, although forces outside the frozen support are not claimed.

For the two MID Active wins, 4.0 N succeeds while the observed 1.5 N arms fail. They are classified `selector_force_error_for_GT` at one paired trial each. For LOW, both Active and GT succeed despite 4.0 versus 1.5 N, so the force difference is not universally necessary.

## 6. Identifier

Fresh Active estimates are 0.4128--0.5494 kg for GT masses .05/.10/.20 kg, with errors +0.3277--+0.3878 kg. This is far worse than offline MAE 0.0229 kg and is systematically high, so fresh query shift does affect identifier output. However, it is not the primary explanation for Active vs GT: GT avoids the estimate and still shares the same query state; the two Active wins are explained by the force decision caused by the conservative high estimate.

## 7. Final explanation

1. **Why Active is 4/6:** its high-biased fresh estimate selects 4.0 N in all six contexts; that rescues both MID contexts and both LOW contexts. The MID rescues are consistent with the 4.0-versus-1.5 N paired outcomes.
2. **Why GT is 2/6:** GT selects 1.5 N in LOW/MID and 2.5 N in HIGH. It succeeds on both LOW contexts, but under-selects force on both MID contexts and fails transport; HIGH fails downstream even at 2.5 N.
3. **Why Fixed-Max is 1/6:** fixed 4.0 N has no query-state handoff and succeeds only on 8400/LOW. Its failures, especially placement/transport, show that force alone is insufficient and that upstream/query-state execution matters.
4. **Why Active's remaining 2/6 fail:** both are HIGH. Active 4.0 N, GT 2.5 N, and Fixed-Max 4.0 N all fail; observed lower-force Prior also fails. These are not currently attributable to selector choice alone; the evidence points to fresh state / frozen-π0 transport-placement limitations.
5. **Current bottleneck:** not Mass identification alone. The evidence is mixed: (a) a high-confidence corrected selection effect in the two MID Active wins, plus (b) a downstream fresh-state/frozen-π0 failure in both HIGH contexts. The safe overall classification is `MASS_E2E_MIXED_FAILURE`, with `force selection` explaining the Active-vs-GT delta in 2/6 and `fresh query/state + π0 transport/placement` explaining the irreducible observed HIGH failures in 2/6.

## Evidence boundary

Active and GT were each run once per context, so the report establishes paired outcome differences but not repeat-level confidence intervals. The two corrected MID pairs are the strongest force-swap evidence available without re-query. No old 0.5 N runner rows are used in any conclusion.
