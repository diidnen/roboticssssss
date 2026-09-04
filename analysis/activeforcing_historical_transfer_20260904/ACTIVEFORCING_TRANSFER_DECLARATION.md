# Historical ActiveForcing artifact transfer

## Purpose

This bundle transfers the smallest auditable set needed by a clean Tabero environment to inspect and reproduce the historical physical-belief prefix:

```text
P4-B diagnostic shear probe
  -> 215-step telemetry
  -> exact 46-D evidence builder
  -> P5-S0-C TRAIN-only normalization
  -> FRICTION_GRU.pt
  -> mu_hat, sigma_mu
```

This is an artifact migration only. No new rollout, training, retraining, posterior update, controller change, or method change was performed.

## Directly reusable assets

- `p4b_probe/p4_collect_probe.py`: executed P4-B probe implementation;
- the P4-B `_apply_friction` implementation;
- `posterior/p5s0d_fresh_e2e_q2f.py`: historical 46-D sequence construction;
- `posterior/evidence_46d.py`: self-contained exact 46-D offline adapter;
- `posterior/FEATURE_SCHEMA_46D.json`: exact model input order;
- `posterior/NORMALIZATION_46D.json`: preserved TRAIN-only statistics;
- `posterior/FRICTION_GRU.pt`: historical physical-friction estimator;
- `posterior/active_friction_imagination_e2e.py`: historical GRU inference;
- `reference_parity/`: three original 215-row parity traces and expected outputs;
- `provenance/`: historical search and execution-path reports.

The probe source visibly preserves approach 45, descend 35, close 70, hold 40, probe_out 10, probe_back 10, and probe_hold 5 for the historical 215-step sequence. It also preserves the approximately 3 N P4-B preload and the existing common contact-frame shear, contact-loss, shear-ratio, displacement, and tactile-marker checks.

## Friction semantics

The simulator setter writes both target-object PhysX material channels:

```text
mats[..., 0] = mu  # static friction
mats[..., 1] = mu  # dynamic friction
```

The true value is recorded in context/episode metadata for audit labels only:

```text
TRUE_FRICTION_IS_PRIVILEGED_ANALYSIS_ONLY = YES
```

True friction is excluded from the 46-D input. The model input is probe evidence, followed by learned inference of the hidden friction state.

## Historical posterior representation

The old estimator is a projection 46 -> 16 with ReLU, GRU hidden size 16, and scalar `mu_hat` / `log_sigma` heads. Runtime returns `mu_hat` and `sigma_mu = exp(log_sigma)`. The historical runtime then formed three clipped point hypotheses `mu_hat - 1.645*sigma`, `mu_hat`, and `mu_hat + 1.645*sigma`.

This is the historical representation, not a claim that an explicit Bayesian density or posterior sampler was implemented.

## What the old chain did not prove

The following are explicitly historical reference only, not the final paper method:

```text
LEGACY_DECISION_BACKEND = HISTORICAL_REFERENCE_ONLY
```

- three-point friction hypotheses;
- deterministic snapshot-branch imagination;
- force candidates `{3.0, 4.0, 4.5, 5.0}`;
- first force with mean imagined success >= 0.9;
- P5-S0-C Q2F direct force-success prediction;
- later E6 physical-belief ensemble;
- later E7/GNP-style feasibility and continuous force planning;
- formal-v2 frontier-posterior code.

The old AFI E2E reached probe, evidence, `mu_hat/sigma_mu`, friction hypotheses, deterministic branches, force selection, and scripted real branches. It did **not** execute the final intended chain:

```text
probe -> physical posterior -> learned full-task feasibility
       -> Expected Utility -> continuous F*
```

No formal full-task feasibility checkpoint is implied by this transfer.

## Historical state limitation

The AFI wrapper captures a post-grasp, pre-shear `s0`, runs the shear probe, and uses post-probe `sq` for downstream branch restores. It is not a strict original-root-start formal E2E with complete simulator/controller/RNG state restoration. See the transferred provenance reports before claiming stronger parity.

## Additional references

`legacy_reference/p5s0c/` contains compact Q2F/force-success source and manifests. It is intentionally marked non-posterior. `legacy_reference/old720/` contains the found 720-row later continuous-force archive plus its audit and generation sources. `legacy_reference/e6/` contains only a later E6 manifest.

`SVR_ckpt.pkl` is not included: it is a 24-D tactile-summary-to-scalar-force regressor, not a friction estimator, posterior, feasibility model, or decision model, and is not imported by the P4-B/FRICTION_GRU chain.
