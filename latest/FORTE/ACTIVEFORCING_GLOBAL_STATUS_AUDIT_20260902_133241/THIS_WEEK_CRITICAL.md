# This week critical

Only unfinished submission blockers are listed.

| Priority | Blocker | Remaining experimental steps | Load | Dependency |
|---:|---|---|---|---|
| 1 | E5 balanced final-Utility E2E | QA current tuple; collect 40 missing tuples/200 current-arm rollouts; add/execute missing required arms; aggregate/freeze report | GPU-heavy | shared pi0; scheduler duplicate gate; Mass contention |
| 2 | Mass E8a closure | finish 180-branch formal set; data QA; formal estimator; 3-seed Direct; Utility; fresh E2E; report | mixed CPU/GPU-heavy | current protected Mass worker; valid formal data |
| 3 | E3 onboarding unblock | demo19 replay; strict QA; assemble/convert; CPU norm; 1000-step LoRA; nominal DEV gate | mixed, one short replay + training GPU-heavy | current E5/Mass resource gate; demo19 |
| 4 | E7 real continuous validation | 8 exact-float tracking/parity rollouts; if pass, 144 matched rollouts; verdict | GPU-heavy | current workers; E0/live force-interface parity |
| 5 | Boundary Direct causal test | minimal preflight; four-task adaptive collection; augmented data; 3-seed retrain; same-Utility comparison; fresh validation | GPU-heavy after CPU prep | force preflight and GPU release |
| 6 | E6 Utility re-query | Utility member-decision recompute; genuine second-query data; five-arm DEV/locked comparison | mixed, simulator needed | qualified second-query continuation |
| 7 | E3 formal FullTask-vs-LocalLift | formal long-horizon data; matched two-label models; Utility E2E | GPU-heavy | onboarding checkpoint must first pass nominal DEV |
| 8 | Joint physics E8b | 3x3 pilot; joint estimator; one-axis ablations; Direct/Utility; E2E | GPU-heavy | Mass closure |
| 9 | external baselines | FORTE adapter; Tabero exact paired tuples; common telemetry/table | implementation + GPU | core tuple manifest; FORTE adapter |

## What is not a critical blocker this week

- Do not rerun global Fmax, old Joint, WM residual, predictive verifier, sample-efficiency 60, or old hard-rho E6/E7.
- Do not expand Phys2Real fusion until the E2 matched modality contract is frozen; it is not currently a runnable submission gate.
- Do not open new E4 variants without an explicit paper decision to retain the transfer claim.
