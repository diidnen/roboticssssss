# ActiveForcing-Direct Selector Contract

Status: **FROZEN FOR PRE-TEST EXECUTION**  
Gate 1: `DIRECT_SELECTOR_SEMANTICS_PASS`

## Exact executable backend

The frozen method-development implementation is the `FEASIBILITY_ONLY`
branch of:

`/home/exouser/FORTE/gnp_style_continuous.py`

Frozen checkpoints are the three-seed ensemble:

| Seed | Checkpoint | SHA-256 |
|---:|---|---|
| 0 | `gnp_style_continuous_20260830_125107/GNP_STYLE_CONTINUOUS_FEAS_seed0.pt` | `920bc6308e997355e39c196afdea9fa65202732c7fc68ea25b2696a48a223b71` |
| 1 | `gnp_style_continuous_20260830_125107/GNP_STYLE_CONTINUOUS_FEAS_seed1.pt` | `5eeaccbf4e9b83019986c55b5c4d48e61d56c4c3744b60e2682d9b20bafb093d` |
| 2 | `gnp_style_continuous_20260830_125107/GNP_STYLE_CONTINUOUS_FEAS_seed2.pt` | `41cf4bdcef697694d046c3ebf1671755c16e4f1d3cc2a44bc633421dc094943f` |

Implementation hashes:

| Artifact | SHA-256 |
|---|---|
| `gnp_style_continuous.py` | `0c919156a9a0dfad5f9b92c0499c17a59e93e2c3dd93acbf19963842d7031259` |
| `full_task_feasibility_decoder.py` | `bf002ca6a5e190f0eff4e0429bd3340f9acfdb4a694dde2528a98581ce8ff83d` |
| `trajectory_physical_imagination.py` | `429761e395e5d6c4f658326e4f5642629eeaaf6498c2217f8f4829695340c2ff` |
| `counterfactual_force_world_model.py` | `c088369626b2cd783f45f5130d932fd7aba6dcc9e54b3504c1a81b4e4317d560` |
| `gnp_style_continuous_20260830_125107/GNP_STYLE_TRAIN_NORMALIZATION.json` | `24d8079a56c02f0b7681ed2c0dad8b21b9d55906885733da032343fe6dfafcd6` |

## Input and representation

- Context source: frozen P4-B probe/context, using the existing strict
  pre-probe last-hold representation; no post-probe state is passed.
- Probe: exact 215-step P4-B sequence and existing 215×46 feature builder.
- TRAIN-only normalization: `GNP_STYLE_TRAIN_NORMALIZATION.json`, applied
  featurewise with stored `x_mean` and `x_std`.
- Model input: normalized 215-step sequence, first 17 columns as the command
  sequence, and the normalized condition vector from columns 17 onward.
- Context representation is unchanged from the frozen method-development
  implementation. No new visual, AFI, mu-hat, or physics-rollout input is
  added by this amendment.

## Model and force encoding

The `FEASIBILITY_ONLY` model is:

```text
command_gru: GRU(input_size=17, hidden_size=64)
condition: Linear(54, 64) + ReLU
head: Linear(128, 64) + ReLU + Linear(64, 1)
```

The candidate force is encoded by the existing `trajectory_physical_imagination`
nominal trace builder in the frozen command/condition fields. The backend
predicts a binary full-task feasibility logit for the candidate force; the
online quantity is the sigmoid of that logit, aggregated as the mean of the
three seed probabilities.

## Online decision rule

For each candidate in ascending order:

```text
score(F) = mean_seed(sigmoid(full_task_feasibility_logit(context, F)))
choose min F where score(F) >= 0.5
```

If no candidate passes, use the existing executable fallback: choose the
maximum candidate force, `5.00 N`, and record `fallback_used = true`. The
fallback is not a new safety rule; it is the frozen max-force fallback used by
the executable decision interface.

The online threshold is exactly `0.5`. It is not `rho_frontier=0.8` and is not
historical AFI `eta=0.9`.

## Downstream handoff

After scoring, retain only the selected force and restore the exact original
root. Fresh frozen π0 is then initialized. The existing P7A/P7B physical
override is used without remapping:

```text
retain π0 Cartesian / rotation / gripper components
zero force slots 7:13
left z-force slot = F/2
right z-force slot = F/2
```

Requested and measured force telemetry, termination, and success are recorded.
The proposed method never uses `mu_hat → physics simulator rollout → eta=0.9`.

## Evaluation-only frontier

The empirical frontier is computed after formal rollout as the minimum force
cell with `>=4/5` frozen-π0 full-task successes. This is `rho_frontier=0.8`
and is not an online Direct selection threshold.
