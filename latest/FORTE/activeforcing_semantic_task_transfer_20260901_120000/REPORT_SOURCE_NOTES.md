# Report Source Notes

- Primary population: archived 720-branch / 24-root / 72-context pooled TRAIN population loaded through the same authoritative parser as `/home/exouser/FORTE/activeforcing_shared_physical_transfer_20260901_094722`.
- Exact target acquisition trajectory SHA256: `b405c18204fbbfe7434589cab11e892e82ac7a1d3b5acdb1aa94d822456f3805`.
- Exact physics predictions SHA256: `d1bbfc6344b69c9ce5f897e4b966d4b0fbcf292e2783e15cebe1defe54321f12`.
- Frozen semantic protocol SHA256: `80a6d0ec19d41646e5bcb0cdf0610eaf28b7df207ac7de0f9896c52ab8aaebd3`.
- Seed-level controller metrics: `SEMANTIC_TASK_TRANSFER_AGG.csv`.
- Exact paired GT decisions: `SEMANTIC_VS_ONEHOT_PAIRED.csv`.
- Data and split QA: `SEMANTIC_TASK_TRANSFER_QA.json` (PASS).
- Scope: archived development only; GT is privileged; task1 has 140/180 reconstructed labels; held-out task and object family are confounded.
- Exclusions: zero new simulator rollouts; root-scaling untouched TEST, World Model, Residual, Joint, Agent, and Probe retraining were not used.
