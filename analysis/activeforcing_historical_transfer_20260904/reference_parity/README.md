# Offline parity reproduction

This directory contains three historical, already-executed 215-row P4-B
probe traces. No Isaac process, rollout, training, or posterior update is
needed to run the check.

## Contract

```text
probe_telemetry.csv
  -> OnlineInference.sequence_array (46-D, exact order)
  -> NORMALIZATION_46D.json dynamic_mean/dynamic_std
  -> FRICTION_GRU.pt
  -> mu_hat, sigma_mu
```

For a clean environment, use the self-contained transferred
`posterior/evidence_46d.py` adapter. It is the same row construction as
`OnlineInference.sequence_array` but does not instantiate the unrelated Q2F
models that the full legacy runner also loads. Then use
`posterior/active_friction_imagination_e2e.py::FrictionGRU` and its
`::estimator_inference` semantics. The checkpoint is the historical file, not
a model to retrain.

For every case, require exactly 215 telemetry data rows and a 46-column
feature array. Compare the new outputs with `expected_output.json` using:

```text
abs(mu_hat_new - mu_hat_old) <= 1e-6
abs(sigma_new - sigma_old) <= 1e-6
```

The historical run achieved exact float equality when replayed on this
server. `QUERY_RESULTS.csv` is the original aggregate output evidence.

The `true_friction` values in `context.json` are audit labels only and must
never be passed into the feature builder or model.
