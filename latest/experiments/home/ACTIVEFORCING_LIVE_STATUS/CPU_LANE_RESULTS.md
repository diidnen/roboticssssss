# CPU recovery lane results

Updated 2026-09-02T14:38Z. All entries below are CPU-only; none authorizes a GPU launch.

| Lane | Result | Deliverable / next gate |
|---|---|---|
| E5 | PASS QA, INCOMPLETE | 21/60 tuples, 105/300 rollouts; next task0 offset7; 39 tuples remain |
| Mass | PASS atomic snapshot | 10/18 TRAIN contexts, 100/180 branches accepted; 8 contexts/80 branches remain; post-gate root8103-mid/high quarantined |
| E3 | 5/5 onboarding data accepted; downstream blocked | assembly/conversion/norm artifacts exist, but ID1 QA hash and norm-path mismatch require coordinator repair; no training/candidate lock/DEV |
| E6 | CPU recompute complete | 36/144 (25%) Expected-Utility member disagreement; no qualified second-query evidence |
| E7 | PASS offline recovery prep | real 8 and 144 rollout evidence absent; runner prerequisite and driver gate unresolved |
| Boundary | CPU manifest complete | 720 old + 0 new; missing minimal force-interface calibration gate |
| Joint | CPU protocol complete | no joint data; prototype not resume-safe and not a genuine joint estimator; waits for Mass closure |
| FORTE/Tabero | CPU provenance complete | 144-tuple paired manifest QA pass; faithful results remain NA |
| E2 | partial / blocked | 72-context TRAIN-only diagnostic input identified; two CPU serialization/startup attempts quarantined, no fusion result |

The project is not scientifically closed. Negative/diagnostic findings are preserved as such and are not promoted into final claims.
