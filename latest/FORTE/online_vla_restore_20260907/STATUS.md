# Online frozen VLA restoration — active development

Master goal: RESTORE_TRUE_FROZEN_VLA_ACTIVEFORCING_AND_COMPLETE_FINAL_ICRA_EVALUATION

## Evidence separation

- SCRIPTED_V5_DEVELOPMENT_EVIDENCE = YES
- SCRIPTED_V5_IS_FINAL_VLA_EVIDENCE = NO
- Existing 21/24, 8/24, 23/24 are excluded from any online-VLA main table.
- No existing evidence or checkpoint is overwritten.

## Current scope

Online inference restoration is verified in completed development branches; Phase B qualification is running. A dedicated server uses the original frozen pi0_lora_tacfield_tabero checkpoint, not the separate task5 fine-tuned candidate. Full checkpoint tree SHA256 was recomputed and equals `0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17`.

Native temporal schedule: 50 predicted actions, execute first 10, then re-infer online. Each executed action must join to a response produced during the same rollout, with request ID, worker ID, observation/action hashes and server model-execution receipt.

The initial VLA chunk is obtained at the post-probe decision state before selecting force; its first eight arm targets supply the feasibility sequence. No scripted motion prefix is allowed. Following the full647-row offline audit, the seven phase columns are now removed from the input and frozen GRU weights. The new candidate takes8x64. Exact removal preserves all647 original probabilities and72 planner decisions. No phase proxy or model replacement is needed for this deletion. Online motion/outcome transfer remains unqualified.

All main fixed baselines have the same corrected probe and post-probe state contract as AF. NoProbe is a separate ablation and cannot be implemented by changing the posterior after a probe.

## Candidate arbitration

The six VLA arm pose channels are passed through exactly after float32 conversion. Raw force channels are masked. During grasp, AF supplies the carried aperture reference and squeeze force. A VLA open intent maps to canonical release aperture .04 and zero active force, identically for every method; raw VLA aperture never substitutes for the AF servo reference. This semantic release mapping is explicitly logged and must pass development qualification. No geometric or scripted trigger replaces a VLA arm target or decides release.

## Candidate label adaptation

The old exact scripted phase schedule is not a valid online-VLA phase annotation. The candidate online label retains a 350-control-step budget, lift >=3cm, no drop/timeout/reset, whole-mesh authored-region containment in all last 50 frames, canonical open plus physically unheld last20 frames, and final support contact. This is a new label version, not old V5 label parity. No VLA release is forced at a predetermined step.

## Evaluation gates

1. Software and provenance smoke; one simulator worker until measured resource qualification.
2. One burned root5100, four tasks, three friction contexts, AF/Fixed3/Fixed5: at most36 primary dev branches. Additional Fixed4 boundary checks remain development-only.
3. Assess action arbitration, OOD behavior, real force boundaries and existing feasibility transfer before further data or training.
4. Final four fresh roots are not yet selected or consumed. No final runtime freeze exists.

Final online VLA results, transfer validity and paper claim support remain unverified until actual qualification and final evaluation succeed.

### Appended goal extension and completed offline audit

RESTORE_ONLINE_VLA_AND_REMOVE_SCRIPT_ONLY_FEASIBILITY_DEPENDENCIES is appended; the active master goal and previous online restoration are preserved. See GOAL_EXTENSION.json, FEASIBILITY_FEATURE_SCHEMA_AUDIT.json and offline_feature_audit_v3/PHASE_SENSITIVITY_SUMMARY.json. No new physics was used for the phase study. Original checkpoint weights remain frozen; phase-removed retraining is an offline ablation only. Full object velocity currently comes from online simulator state and is disclosed as such.

Dev v3 contains6 completed online branches on task0 LOW/MID: AF andFixed5 succeed in both;Fixed3 fails in both. These remain development evidence with their original71D zero-phase candidate, not final evidence or new phase-free results. Measured pre-release drop is independently proven for task0 LOW Fixed3. The global transfer and other3tasks remain unresolved. Final fresh roots consumed:0.

### First physical integration finding

Dev v2 task0/LOW proved live model calls, but exposed an incorrect midpoint aperture interpretation in arbitration V1: the VLA requested .02590m, close to the real established-grasp aperture .02594m, while the carried AF command was .02620m. A .02m threshold incorrectly converted this into full release. This branch is integration-failure evidence, excluded from boundary and feasibility-transfer data. Arbitration V2 requires a near-canonical full-open request (>=.039m for the .04m physical limit). This correction is prospectively checked on new dev rollouts; no outcome is relabeled.

## Latest independently verified progress

23/36 primary dev branches,805 model calls have passed online provenance and geometric checks. Task0/task1/task5 have paired low-force failure versus higher-force success. Task5 MID Fixed3 succeeds where LOW fails. Task6 and four Fixed4 boundary branches remain pending. One late regrasp candidate follows task5 LOW Fixed3 physical failure; early regrasp candidate rate is zero in the joined22-branch review, not a claim of no recovery behavior. A separate12-reference probe sensor capture queue waits for the current coordinator to finish before using the simulator. No final roots, model fitting, or final paper claim. See GOAL_EXTENSION_PROGRESS_STATUS.json and INDEPENDENT_QUALIFICATION_PROGRESS.json.
