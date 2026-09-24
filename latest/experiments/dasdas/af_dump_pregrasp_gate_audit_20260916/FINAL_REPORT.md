# PRE grasp-establishment gate audit — dump-bin ActiveForcing

Date: 2026-09-16
Protocol freeze: diagnostic query 4 N / 12 mm world-Y, preload, hold, force controller, μ grid, maxF=5, P4, EU, official terminal, π0 chunks, 32×20 dataset, official 18/19, Fig.B, checkpoints — **all untouched**. No 32×20 resample. No retrain.

Core question: after requiring a real bilateral nx+nx wall pinch **before** force selection, do the old dump quantitative force-response anomalies remain?

Answer: **yes on the official 32×20; no on the isolation wall F-grid’s all-F failures.** Verdict **MIXED**.

---

## 1. PRE gate definition (frozen from PRE clusters only)

Operating point: established bilateral wall pinch → diagnostic query → force decision → remainder.

Measurement window: after scripted grasp prefix, settle **20 physics steps**, then snapshot **before query**. Gate never uses query CR, force-sweep outcome, terminal success, or ActiveForcing prediction.

Two PRE clusters on seed 200014 / μ=0.425 / official contact id=1 (episodes 0–9):

- Pinch: aperture ~1 mm, nx+nx, squeeze ≫ 0, bin at grasp height ~0.74 m.
- Miss: aperture ~45 mm, squeeze = 0, no finger–bin contact.

Cutoffs (from that PRE bimodality, not from F outcomes):

| cutoff | value |
|---|---|
| `aperture_miss_m` | 0.02 m |
| `squeeze_alive_n` | 0.05 N |
| `window_bilateral_valid_min` | 0.50 |
| `window_bilateral_borderline_min` | 0.80 |

Core criterion is official **nx+nx** (`grasp_nx` / contact_point_id=1 on both fingers). Window CR is used so a single-frame flicker does not auto-INVALID a living pinch.

## 2. VALID / BORDERLINE / INVALID

**VALID** (default quantitative set):

- PRE window bilateral CR ≥ 0.50
- AND endpoint nx+nx
- AND endpoint squeeze > 0.05 N
- AND gripper aperture < 0.02 m

**BORDERLINE** (report separately; never merge):

- PRE window bilateral CR ≥ 0.80
- AND aperture < 0.02 m
- AND (endpoint not nx+nx OR endpoint squeeze ≤ 0.05 N)

Typical: almost-all-window bilateral grasp with 1–2 frame flicker. None of the 10 seed-200014 roots landed here. Episode 3 is **VALID** at PRE (CR=1.00, nx+nx); its **hold** endpoint is the flicker case (hold CR=0.966, hold endpoint not nx+nx). That is a No-Query hold issue, not a PRE miss.

**INVALID**: everything else. Typical miss: window CR=0, squeeze≈0, aperture≈45 mm, bin not pinched.

**QCR0_PROXY** (historical logs with no PRE window): `qCR==0` → `PRE_INVALID_PROXY`; `qCR>0` → `PRE_VALID_PROXY`. Calibration on the 10-root PRE study: P(qCR=0 | INVALID)=1, P(qCR=0 | VALID)=0. Labeled PROXY, never VALID. Do not join proxy labels with a different physics realization of the same (seed, μ, episode).

## 3. Seed 200014 10-root PRE audit

μ=0.425, official left grasp contact_point_id=1, collision filter on. Source: `query_probe/` via `PROBE_10ROOT_AUDIT.json`.

| ep | gate | L | R | nx+nx | PRE CR | squeeze N | aperture mm | pair mm | bin z | hold alive | query alive | qCR | query squeeze N | query pair mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | VALID | 1 | 1 | 1 | 0.70 | 58.84 | 1.0 | 5.6 | 0.741 | 1 | 1 | 0.86 | 9.22 | 4.7 |
| 1 | INVALID | 0 | 0 | 0 | 0.00 | 0.00 | 45.0 | — | 0.875 | 0 | 0 | 0.00 | 0.00 | — |
| 2 | VALID | 1 | 1 | 1 | 0.65 | 40.45 | 1.0 | 9.8 | 0.742 | 1 | 1 | 0.79 | 13.38 | 10.5 |
| 3 | VALID | 1 | 1 | 1 | 1.00 | 21.69 | 1.1 | 4.7 | 0.742 | 0* | 1 | 0.85 | 8.38 | 4.8 |
| 4 | INVALID | 0 | 0 | 0 | 0.00 | 0.00 | 45.0 | — | 0.742 | 0 | 0 | 0.00 | 0.00 | — |
| 5 | VALID | 1 | 1 | 1 | 0.75 | 22.61 | 0.9 | 4.7 | 0.743 | 1 | 1 | 0.85 | 5.67 | 11.9 |
| 6 | VALID | 1 | 1 | 1 | 0.95 | 21.37 | 0.9 | 4.7 | 0.744 | 1 | 1 | 0.76 | 8.77 | 5.1 |
| 7 | VALID | 1 | 1 | 1 | 0.65 | 14.12 | 1.0 | 4.8 | 0.744 | 1 | 1 | 0.90 | 8.78 | 4.7 |
| 8 | VALID | 1 | 1 | 1 | 0.75 | 22.25 | 0.9 | 8.3 | 0.743 | 1 | 1 | 1.00 | 8.21 | 4.8 |
| 9 | VALID | 1 | 1 | 1 | 0.80 | 11.26 | 1.1 | 12.1 | 0.744 | 1 | 1 | 0.85 | 8.36 | 4.8 |

\*ep3 hold: window CR 0.966, endpoint unilateral flicker. PRE itself is a clean VALID pinch.

## 4. PRE 8/10 reproduced?

**Yes.** VALID 8 / BORDERLINE 0 / INVALID 2. `reproduces_8_of_10 = true`.

## 5. ep1 / ep4 invalid reasons

Both are grasp-establishment failures, not query/force failures.

- **ep1**: gripper open 45.0 mm; squeeze≈0; PRE bilateral CR=0; not nx+nx; no left/right target contact; bin z=0.875 (not in the pinched cluster at ~0.74 m).
- **ep4**: gripper open 45.0 mm; squeeze≈0; PRE bilateral CR=0; not nx+nx; no left/right target contact. Residual table-bottom normal ~0.10 N (bin still on table). Bin z happens to sit near grasp height in this snapshot but fingers are fully open.

## 6. PRE_VALID Query vs No-Query survival

Fork from the same PRE snapshot; stop at handoff; no force sweep.

| | hold (No-Query) | 4 N / 12 mm query |
|---|---|---|
| P(grasp survives \| PRE_VALID) | 7/8 = 0.875 | 8/8 = 1.000 |
| query kills of living wall pinches | — | **0 / 8** |

Formal record: **query does not systematically destroy an established wall pinch.**

## 7. Query effect on squeeze / pair separation

Among VALID prefixes that remain alive at the stage endpoint:

- Hold squeeze mean: **21.41 N** (n=7)
- Query squeeze mean: **8.85 N** (range 5.67–13.38 N)
- Hold pair mean: **4.74 mm**
- Query pair mean: **6.41 mm**, max **11.92 mm** (ep5)

Diagnostic-query robustness stays **MIXED**: contact is kept, but load is unloaded onto the 4 N cap and some pairs open to 10–12 mm. Do not relabel this PASS.

## 8. Grasp-establishment pass rate (PRE_INVALID is not discarded)

| corpus | pass | note |
|---|---|---|
| query_probe seed 200014 | **8/10** | true PRE gate |
| isolation wall F-grid (qCR proxy) | **3/5** | 2 qCR=0 open-gripper misses |
| v4 32×20 (qCR proxy) | **32/32** | every context has qCR=1.0 |

ActiveForcing evaluation on v4 is already conditional on query contact. The 2/10 (and 2/5) misses are counted as **grasp-establishment failure**, not force-selection failure.

## 9. Gate before vs after — qCR=0, monotonicity, curves, retention

### Isolation wall F-grid (5 prefixes × F=0.50/0.75/1.00/1.25)

| | all 5 | PRE_VALID_PROXY (3) | PRE_INVALID_PROXY (2) |
|---|---|---|---|
| qCR=0 | 2 | 0 | 2 |
| all-F fail | 2 | 0 | 2 |
| nondecreasing | 4 | 2 | 2 (vacuous all-zero) |
| retention at every F | 40% | **67%** | 0% |

qCR=0 **equals** all-F fail on this grid. Filtering removes the entire all-force-failure mass.

Remaining proxy-valid pattern: ep2 all-F retain; ep3 F≥0.75 retain (threshold-like); ep1 only F=0.50 retains then drops — later-F drop / possible squeeze ejection.

Do **not** join this F-grid with query_probe PRE labels. PhysX is not repeatable: wall ep0 is qCR=0 here but PRE VALID in the probe; wall ep1 is qCR>0 here but PRE INVALID in the probe.

### Official v4 32×20 (8 μ × 4 seeds × 20 forces)

| | all 32 | QCR0_PROXY valid | QCR0_PROXY invalid |
|---|---|---|---|
| qCR=0 | **0** | 0 | 0 |
| all-F fail | 0 | 0 | 0 |
| nondecreasing | 10 (31.25%) | 10 (31.25%) | — |
| success-vs-F | see below | **identical** | empty |

v4 `query_contact_ratio` is **1.0 on every context**. The PRE-miss contamination that produced qCR=0 on the isolation F-grid is **absent** from the official 32×20.

Success vs F (all = proxy-valid; % of 32 contexts):

```
F     0.25  0.50  0.75  1.00  1.25  1.50  1.75  2.00  2.25  2.50
%     25.0  31.3  31.3  40.6  65.6  75.0  68.8  56.3  53.1  65.6
F     2.75  3.00  3.25  3.50  3.75  4.00  4.25  4.50  4.75  5.00
%     81.3  71.9  87.5  81.3  84.4  90.6  87.5  81.3  87.5  90.6
```

The curve still rises, dips (1.50→2.25: 75%→53%), rises again. Filtering PRE_INVALID does not change one point.

## 10. PRE_INVALID contamination of old feasibility labels

- **Wall F-grid:** 2/2 all-F failures are PRE_INVALID_PROXY (open gripper at handoff, aperture ~45 mm). 100% of that grid’s all-force failure mass is grasp-establishment, not force selection. Those rows must not be read as “F does not matter.”
- **v4 32×20:** **0/32** qCR=0, **0/32** all-F fail. Old feasibility / GRU labels on this dataset are **not** contaminated by never-grasped prefixes. The 22/32 nonmonotonic contexts all had full query contact.

## 11. Filtering and friction–F_min

On v4, filtering does nothing, so the relation is unchanged.

- Spearman(μ, selected F): −0.78, **n=32**, but 31/32 selections are 5.0 N and one is 4.75 N. Not a Coulomb law; it is a saturated maxF policy.
- Spearman(μ, F_min_stable): **+0.75**, n=10 monotonic contexts. Wrong sign vs “low μ → larger retention F_min.” Eight of those ten have F_min=0.25 N (success from the lowest grid point), so F_min is barely identified.
- Mean success rate vs μ is only weakly increasing (0.59 at μ=0.35 to 0.81 at μ=0.85) and not monotone (μ=0.50 is a local dip at 0.56).

**Filtering PRE_INVALID does not make friction–F_min more reasonable on the official 32×20.**

On the 3 proxy-valid wall prefixes the F grid is too short (0.50–1.25 N) and n=3 to claim a μ law; this round did not re-sweep μ.

## 12. Remaining nonmonotonic cases (proxy-valid / true VALID)

v4 proxy-valid residuals (32 contexts, all qCR=1):

| class | n |
|---|---|
| monotonic | 10 |
| nonmonotonic: later-F drop (possible squeeze ejection / dump geometry) | 22 |

No v4 row is qCR=0, flicker-at-query, or all-F fail. The 22 later-F drops are the residual after a PRE-style contact filter. They are **not** classified from PRE geometry (v4 has no PRE nx+nx log). Plausible mechanisms, not proven per-row:

- geometric wedging / dump pose
- squeeze-induced ejection at higher F
- downstream dump failure after a living handoff
- contact flicker during remainder (not PRE)
- controller tracking at high F

Wall proxy-valid residual: 1/3 later-F drop (episode 1: retain only at 0.50 N).

## 13. Verdict: MIXED

| check | result |
|---|---|
| PRE_INVALID explains a chunk of qCR=0 / all-F fail | **Yes on wall F-grid (2/2). No on v4 (0 qCR=0).** |
| Query extra-kills of VALID wall pinches | **0 / 8** |
| Force-response clearly more stable after filter | **Yes on wall (40%→67% retention; all-F fail gone). No change on v4.** |
| Friction / F_min more reasonable | **No on v4.** |

PASS would require the official 32×20 to clean up. FAIL would require VALID pinches to die under query, or residual curves to be unreadable. Neither holds.

Labels kept: wall pinch **PASS**, force controller **PASS**, PRE gate **PASS** (as a filter), diagnostic-query robustness **MIXED**.

## 14. Grasp-surface-only friction isolation next?

**Yes, as the next experiment, not as a silent protocol patch.** Remaining nonmonotonicity lives on prefixes that already have query contact. The next isolation should ask whether dump-bin / table / inner-pad μ is confounding the finger–wall Coulomb story. Do not retune query to chase that.

## 15. Resample 32×20?

**Not this round, and not because of the PRE gate.** The official 32×20 already has qCR=1 on every context; resampling would not remove PRE_INVALID contamination that is not there. A new 32×20 is only justified after a protocol change that actually alters the operating point (true PRE gate logged into the sweep, or grasp-surface μ isolation). This audit did not execute that resample.

---

Files: `GATE_DEFINITION.json`, `PROBE_10ROOT_AUDIT.json`, `WALL_FGRID_REGROUP.json`, `V4_REGROUP.json`, `V4_CONTEXTS.json`, `CONTAMINATION.json`, `MONOTONICITY.json`, `FRICTION_FMIN.json`, `RESIDUALS_PROXY_VALID.json`, `SUMMARY.json`, `run_pregrasp_gate_audit.py`.
