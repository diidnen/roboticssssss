# Formal TEST Entry Gate — ActiveForcing-Direct Amendment

Updated: 2026-08-31

## Pre-TEST freeze

- Proposed method: `ActiveForcing-Direct`
- Online Direct selector: minimum candidate with `p_success >= 0.5`
- Empirical frontier: minimum frozen-π0 full-task force with `>=4/5` successes; `rho_frontier = 0.8`
- Historical AFI ablation: `eta = 0.9`, not the proposed method
- Previous `ACTIVEFORCING_DECISION_SEMANTICS_AMBIGUOUS`: `RESOLVED_BY_PRETEST_USER_FREEZE`

## Runtime gates

| Gate | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | `DIRECT_SELECTOR_SEMANTICS_PASS` | **PASS** | Frozen Direct contract and backend audit; threshold/grid/fallback recorded |
| 2 | `ORIGINAL_ROOT_RESTORE_EQUIVALENCE_PASS` | **PASS** | DEV root 5106, P4-B probe, restore scene max abs diff `4.95091e-06` |
| 3 | `TASK0_NEWTON_FORCE_TO_PI0_PASS` | **PASS** | DEV task0 3/4/5 N; each 220 steps, 22 chunks, 220 measured-force samples; left/right slots equal F/2 |
| 4 | `ACTIVEFORCING_DIRECT_DEV_CLOSED_LOOP_PASS` | **PASS** | DEV P4-B → nine Direct scores → 4.25 N → R0 restore → 220-step frozen π0 loop |

## Formal opening safeguard

Formal TEST remains **NOT OPENED** because the exact frozen Direct implementation
is not executable on the requested TEST roots without an unapproved method/input
change. `gnp_style_continuous.py` calls the frozen `preprobe_full_task_feasibility`
loader with `split="DEV"`; that loader explicitly discards TEST rows and has no
runtime context builder for roots 5174–5179. Existing 5174–5179 artifacts are an
incomplete/aborted context collection and do not provide the exact Direct input
contract for all 18 root×μ cells.

Resolving this requires either a new TEST context representation/builder or a
backend/model swap. Both are forbidden after the pre-TEST freeze. No TEST
simulator rollout, preview, selector inspection, or formal result was run.

## Population status

- TEST roots: `5174–5179` — simulator access: **none**
- μ values: `0.2, 0.5, 1.0`
- Frontier rollouts: `0 / 810`
- Primary method rollouts: `0 / 180` planned matched downstream rows
- Privileged method rollouts: not run
- Scientific verdict: **not estimable; TEST NOT OPENED**
