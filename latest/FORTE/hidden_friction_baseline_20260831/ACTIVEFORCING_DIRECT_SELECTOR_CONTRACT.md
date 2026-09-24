# ActiveForcing-Direct Selector Contract

Status: **FROZEN FOR FORMAL TEST**  
Gate 1: `DIRECT_SELECTOR_SEMANTICS_PASS`

## Exact backend

The proposed method is the `FEASIBILITY_ONLY` decoder implemented by:

`/home/exouser/FORTE/gnp_style_continuous.py`

with `FeasibilityOnly` from:

`/home/exouser/Tabero/analysis/full_task_feasibility_decoder.py`

Frozen checkpoints are the three-seed ensemble in
`/home/exouser/FORTE/gnp_style_continuous_20260830_125107/`:

| seed | checkpoint SHA-256 |
|---:|---|
| 0 | `920bc6308e997355e39c196afdea9fa65202732c7fc68ea25b2696a48a223b71` |
| 1 | `5eeaccbf4e9b83019986c55b5c4d48e61d56c4c3744b60e2682d9b20bafb093d` |
| 2 | `41cf4bdcef697694d046c3ebf1671755c16e4f1d3cc2a44bc633421dc094943f` |

Model input is normalized `H=8 × 71`: the first 17 columns are the frozen
command segment and the final 54 columns are the frozen condition vector.
Normalization is `GNP_STYLE_TRAIN_NORMALIZATION.json`, fit on TRAIN only.

## Frozen TEST context builder

For each TEST case, the builder receives only the current root and the raw
P4-B probe telemetry generated in that case:

1. Build the exact P4-B probe sequence using the already frozen implementation.
2. Encode the raw probe with the existing 46-feature ordering from
   `P5S0C_NORMALIZATION.json`, using its TRAIN-only dynamic mean/std.
3. Apply the unchanged `FRICTION_GRU.pt` estimator to obtain point estimate
   `mu_hat` and diagnostic `sigma_mu`.
4. Read the post-probe EEF command before restoring R0. Construct eight
   `branch_hold` rows, matching the frozen first-H query segment. This is a
   command skeleton only; no post-probe physical state is passed to Direct.
5. For each candidate force, encode force as the existing static field
   `force/8.0`, retain `mu_hat` as the probe-derived context condition, and
   run the unchanged Direct decoder.

The formal context is generated online inside the TEST worker. No cached
`5174–5179` context or old outcome table is read.

## Candidate forces and decision

```text
3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00 N
```

`score(F)` is the mean of the three frozen seed sigmoid probabilities. Select
the minimum candidate satisfying:

```text
score(F) >= 0.5
```

If none passes, use the existing executable fallback unchanged: `5.00 N` and
`fallback_used=true`. The secondary TRAIN isotonic curves are diagnostic only
and do not change the online decision.

## Handoff and separation from AFI

After selection, retain only `F*`, restore exact R0, fresh-initialize frozen
π0, and use the existing Newton override: Cartesian/rotation/gripper action
components are retained; left and right z-force slots are each `F*/2`.

This is not `ActiveForcing-AFI`: there is no `mu_hat → physics simulator/world
model rollout → eta=0.9` selector. AFI `eta=0.9` remains an ablation only.

The online Direct threshold is `0.5`; empirical frontier `rho_frontier=0.8`
is evaluation-only and is never used for selection.
