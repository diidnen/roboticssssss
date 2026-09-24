# D2 — Hierarchical Force Decision Qualification

Generated: 2026-08-20 07:48 UTC

**Status:** `D2_HIERARCHICAL_FORCE_DECISION_QUALIFIED`

## Rule
# Hierarchical force decision (D2)

Frozen probe: P3 A (+2 mm base-Y, 4 N grasp). No joint NB.

## Stage 1 — 6 N safety veto
Signal: `imb_peak` (peak left/right force imbalance during probe).
Rule: if `imb_peak <= 0.724223` → **6 N** and STOP.
Rationale: low μ shows smaller probe imbalance; conservative θ = max(calib low μ).

## Stage 2 — Mid vs high (only if NOT stage-1 LOW)
Signal: `ftan_hyst` (mean |F_tan_forward − F_tan_return| during probe).
Rule: if `ftan_hyst > 0.0113779` → **4 N**; else **5 N**.
Ambiguous → 5 N (conservative for under-force).

Thresholds fit on **even** calibration seeds only.
Held-out **odd** seeds evaluated once.


## Offline held-out (odd seeds)
- exact: 0.867
- under: 0.000
- over: 0.133
- mean F: 5.17 N
- gate passed: True

## Confirmation (seeds 30–49)
### fixed6
- full SR: 0.983
- mean chosen F: 6.00 N
- under/over: 0.000 / 0.667

### current_p3
- full SR: 1.000
- mean chosen F: 5.67 N
- under/over: 0.000 / 0.667

### hierarchical
- full SR: 1.000
- mean chosen F: 5.07 N
- under/over: 0.000 / 0.067

### oracle
- full SR: 1.000
- mean chosen F: 5.00 N
- under/over: 0.000 / 0.000

