# ActiveForcing New Task-Form Final Report

Generated: 2026-09-13T22:48:45.793718+00:00  
Task: `dump_bin_bigbin`  
Formal population: root 200002, eight held-out pi0 paths, three friction settings  
Status: **COMPLETE — 24 paired contexts / 72 formal rollouts; all raw archives and traces independently reverified.**

## Executive result

| Method | Full-task success | Rate (Wilson 95% CI) | Mean commanded force | Mean measured squeeze |
|---|---:|---:|---:|---:|
| Nominal Frozen VLA | 10/24 | 41.7% (24.5%-61.2%) | N/A N | 13.757 N |
| Fixed-Strong 8N | 7/24 | 29.2% (14.9%-49.2%) | 8.000 N | 3.258 N |
| ActiveForcing | 3/24 | 12.5% (4.3%-31.0%) | 4.946 N | 1.379 N |

The formal data do not show AF outperforming Nominal. AF did not outperform Fixed-8 on full-task success.

AF versus Fixed-8 paired cells (both success / AF only / Fixed only / both fail) were **3 / 0 / 4 / 17**; exact two-sided McNemar p=0.125. AF versus Nominal cells were **3 / 0 / 7 / 14**; p=0.01562.

This result supports only same-root, held-out-path performance. It is not evidence of unseen-root generalization or zero-shot transfer to an unseen task form.

## 1. What the original four-task protocol actually was

The authoritative four-task feasibility corpus was a prospective 648-slot design (647 admitted rows after one infrastructure-unknown quarantine). Its split unit was the complete physical root family: four TRAIN roots (5100, 5101, 5102, 5106), one VAL root (6100), and one TEST root (6103). Every split contained all four tasks and all three friction bands; each task-friction context had nine force branches from 3.00 to 5.00 N.

The feasibility corpus used scripted downstream motion, not a VLA. Therefore policy seed and VLA-motion holdout are N/A for its train/validation/test split. The later 96-context main evaluation used eight entirely new roots (170040-170047), online frozen pi0 replanning, and matched method siblings with a shared initial action chunk. Actions after the common prefix were generated online from each branch's own observations.

| Original question | Audited answer |
|---|---|
| Train/validation/test split by physical root? | Yes, complete-root 4/1/1. |
| Split by friction? | No; all frictions occurred in every split. |
| Held-out VLA policy seeds or motions? | No / N/A for feasibility; the corpus was scripted. |
| Main roots separate from feasibility roots? | Yes. |
| Shared branch context? | Matched exposed decision context and common initial online chunk; no claim of byte-identical hidden PhysX solver state. |

## 2. How the new-task protocol matches—and departs from—the original

| Dimension | Original paper | New task formal evaluation |
|---|---|---|
| Feasibility data | Root-disjoint 4/1/1, scripted motion | 192 labels on root 200002; frozen without retraining |
| Formal physical roots | Eight fresh roots | One reused root, 200002 |
| Friction coverage | Three bands | .425, .575, .850 |
| Online VLA paths | One matched online context per physical context | Eight prospectively held-out path seeds, 80200002-80200009 |
| Pairing | Same context and first online chunk across methods | Same post-query handoff, reset pi0 seed, identical first chunk across all three siblings |
| Methods | Fixed-3/4/5 and AF | Nominal Frozen VLA, Fixed-Strong 8 N, AF |
| Claim boundary | Cross-root main evaluation | Same-root held-out-path evaluation only |

The single-root deviation was explicitly requested and frozen before execution. It improves path coverage relative to the prior diagnostics but cannot substitute for a root-disjoint test.

## 3. Existing versus newly collected data and exact counts

| Population | Contexts / labels | Rollouts | Included in formal efficacy denominator? |
|---|---:|---:|---|
| Frozen root-local feasibility corpus | 192 labeled branches; 35 deduplicated belief queries | 192 existing | No; training evidence |
| Nominal development smoke | 1 context | 1 new | No; admission only |
| Formal main | 24 contexts | 72 new | **Yes** |
| Stage-I motion-context study after budget cap | 16 contexts | 96 | No; separate Claim A evidence |
| Pre-physics failed smoke attempt | 0 scientific contexts | 0 scientific rollouts | No |

The formal denominator is exactly 24 contexts and 72 rollouts. The number 96 elsewhere denotes either the original paper's 96 main contexts or Stage-I's 16 contexts x six force policies; neither is mixed into this formal denominator.

## 4. Formal results by friction

| Friction | Nominal | Fixed-8 | AF | AF mean setpoint |
|---:|---:|---:|---:|---:|
| 0.425 | 1/8 | 1/8 | 0/8 | 5.944 N |
| 0.575 | 2/8 | 2/8 | 1/8 | 5.356 N |
| 0.850 | 7/8 | 4/8 | 2/8 | 3.538 N |

Per-seed results and the full 24-context outcome records are preserved in `FINAL_INDEPENDENT_AUDIT.json`; no context was added, removed, or replaced after outcomes were observed.

## 5. Paired comparison

| Comparison | Both succeed | Left only | Right only | Both fail | Difference | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| AF vs Fixed-8 | 3 | 0 | 4 | 17 | -16.7 pp | 0.125 |
| AF vs Nominal | 3 | 0 | 7 | 14 | -29.2 pp | 0.01562 |

The exact McNemar tests use only discordant paired contexts and are descriptive inferential checks for this fixed n=24 population. Equal marginal counts, if present, are not interpreted as equivalence.

## 6. Success-force trade-off

AF selected a mean commanded force of **4.946 N** (distribution: {"2.05": 1, "2.70": 1, "3.40": 1, "3.65": 1, "3.70": 2, "3.85": 1, "4.05": 1, "4.20": 1, "4.25": 2, "4.45": 1, "5.05": 1, "5.25": 1, "5.30": 2, "5.60": 1, "6.30": 1, "6.80": 4, "7.00": 1, "7.45": 1}), compared with the locked 8.000 N Fixed-Strong setpoint. Nominal has no Newton-valued setpoint and is correctly reported as N/A.

Across all paired contexts, AF's mean measured squeeze differed from Fixed-8 by -1.879 N and from Nominal by -12.378 N. Among the 3 contexts where AF and Fixed-8 both succeeded, AF saved a mean 3.217 N of commanded setpoint.

Measured squeeze is sensor-derived and not identical to the controller's commanded setpoint; both are reported to avoid conflating command reduction with realized load reduction.

## 7. Raw-trace intermediate and contact-loss metrics

These metrics were recomputed after completion from every archived physics trace. They are descriptive, not additional success criteria. Sustained contact loss means at least 50 consecutive logged physics steps without qualifying target contact; a descriptive drop failure is an official task failure with that flag.

| Method | Ever lifted bin 4 cm | Ever bin z >= 1 m | Ever all 5 garbage in target z-band | Sustained contact loss | Descriptive drop failures | Mean target-contact fraction |
|---|---:|---:|---:|---:|---:|---:|
| Nominal Frozen VLA | 22/24 | 10/24 | 16/24 | 14/24 | 14/24 | 0.485 |
| Fixed-Strong 8N | 23/24 | 8/24 | 16/24 | 17/24 | 17/24 | 0.415 |
| ActiveForcing | 22/24 | 11/24 | 12/24 | 21/24 | 21/24 | 0.286 |

## 8. Separate Stage-I motion-context evidence

Stage-I is not part of the 24-context formal efficacy test. At the user's budget stop, it retained the first 16 contexts (four policy paths x four frictions) and 96 rollouts across AF plus Fixed-1/3/5/6/8. The original precommit was 32 contexts; the mid-stream budget revision is disclosed, so this is supporting rather than pristine preregistered evidence.

Among 24 within-friction motion pairs, exact fixed-force feasible sets differed at rate **58.3%** (bootstrap 95% CI 20.8%-58.3%); mean Jaccard distance was 0.403. Nonmonotone force landscapes occurred in 5/16 contexts (31.2%, bootstrap 95% CI 12.5%-56.2%).

Stage-I AF succeeded in 6/16 contexts versus 4/16 for Fixed-8. Their paired cells were both success 2, AF only 4, Fixed-8 only 2, both fail 8. These numbers describe the intervention study and are not pooled with formal-main success.

## 9. Claim audit

| Claim | Verdict | Evidence boundary |
|---|---|---|
| AF is useful on a distinct task form | **PARTIALLY SUPPORTED; NOT A RELIABILITY ADVANTAGE** | AF succeeded in 3/24 contexts versus 7/24 for Fixed-8 and 10/24 for Nominal. It had 0 AF-only wins against Fixed-8 and 0 against Nominal; in the 3 AF/Fixed-8 joint successes it saved 3.217 N of commanded setpoint on average. Inference is restricted to root 200002. |
| Downstream motion context affects force feasibility | **SUPPORTED as a budget-capped Stage-I population result** | Within-friction feasible-force sets differed in 58.3% of 24 motion pairs (bootstrap 95% CI 20.8%-58.3%); this evidence is separate from formal efficacy. |
| AF beats Nominal Frozen VLA | **NOT SUPPORTED** | Paired matrix (both/AF-only/Nominal-only/both-fail) = 3 / 0 / 7 / 14; exact McNemar p=0.01562. |
| AF reduces load versus Fixed-Strong | **SUPPORTED for commanded setpoint and measured mean squeeze** | AF mean setpoint 4.946 N versus 8.000 N; mean measured squeeze 1.379 N versus 3.258 N. |
| Zero-shot transfer to a new task or unseen root | **NOT CLAIMED** | Training and formal evaluation both use root 200002. |

## 10. Nominal baseline validity

The one-context admission smoke passed. Nominal used the frozen pi0 14D native joint-position output, including native gripper-position commands at indices 6 and 13; it installed no AF selector, no fixed-N override, and no force servo. It began at the same established-grasp/post-query handoff, executed finite native actions, produced nonempty squeeze telemetry, and kept commanded force as N/A. Smoke task success was not required and the observed task result was retained.

## 11. Engineering incident and repair

The first smoke attempt stopped before any query step or policy action because `AF_INFERENCE_CONTEXT was absent from the child environment required by rim20_formal_binding.` It generated no scientific outcome. The repair bound `AF_INFERENCE_CONTEXT` to the same frozen `SPEC.json` already supplied as `AF_FORMAL_CONTEXT`, then reran the same smoke context without changing its seed. The extension records queue source 43a4b4e4fca8cb95aabe8943077e2808965f09d19519adee31c085f2de0adede -> 3c9a89daeaa99a5ca5bf3908fb83eaed096acafa15d506487e0a59877ee8f957 and explicitly marks model, metrics, context/seed, and scientific protocol as unchanged. The failed attempt remains archived under `engineering_failures/`.

## 12. Integrity and independent audit

The execution queue performed a raw per-context audit before archiving and deleting local working copies. A separate final verifier then reconstructed the frozen 24-context denominator, checked the model/runtime/plan locks, validated all 24 compact records and archive receipts, reproduced every aggregate and paired matrix, and confirmed 72 unique trace-content hashes.

A second verifier ran where the raw tarballs reside on Anvil. In a single pass over every archive, it recomputed each tarball SHA256, hashed every archived member against `ARCHIVE_FILE_MANIFEST.json`, revalidated the archived per-context audits, replayed all 72 compressed physics traces, recomputed squeeze statistics and final success from raw actor states, and confirmed all results.

| Artifact | SHA256 |
|---|---|
| `FINAL_INDEPENDENT_AUDIT.json` | `b85b7c22f2dfa26d05d845c470200a1615719be15acd01da33c6a314d946ca31` |
| `ANVIL_FORMAL_TRACE_AUDIT.json` | `8c244fa44c0c5166e72dfae176fba5a217e1c9a68ef17aaa91272a4ff6192c54` |
| `STAGE_I_BUDGET16_POPULATION_RESULTS.json` | `cc55395e195e805d414023062213a6f80fd321a03c8cf840730eaea4e898d63c` |
| `generate_final_report.py` | `abef777520701ebaa60ce6f086a02d03274bf01c9cb34bf3f303f22aebfe94cb` |
| `contexts_sha256` | `ef4c4bb6fa704e5ff1c67fdc7e7558288de9fdf1166c000fb7b6c445c9186a89` |
| `freeze_lock_sha256` | `3895dffc0cd10fe8e3350ec5c3e7dcd9b1f0eaf8f9fcd23ad80ad93a2b94e4ff` |
| `non_exposure_audit_sha256` | `ea17bf1d23a4ff81801dff7cc08994ef2225f8b6adb06228edf52fa9309293da` |
| `protocol_sha256` | `97237a062742f0279f095e710d3c5c94826aa046e5347edd58637867c9b04738` |
| `runtime_manifest_sha256` | `01d0289b14b812a4a12455e826a0b213ef4bf4fcdb27717892289b7e137b5034` |
| `training_complete_sha256` | `f99f76022919440bc8a05d561996cebd63e332240e161cbf0c9d645a54d6a44a` |

Archive receipts and the deep audit establish 24/24 verified tarballs and 72/72 verified raw traces. The Anvil deep-audit output is itself retained with the experiment and copied to the report directory.

## 13. Frozen-component confirmation

No feasibility model was retrained or reselected. The frozen pi0 checkpoint, `models_v4` ensemble, 58D belief representation, phase-free 8x64 feasibility architecture, posterior integration, maxF8 utility `p*(8-F)/8 - (1-p)`, 0.05 N deployment search, success predicate, squeeze controller, arm-action path, and force support were unchanged. Formal outcomes did not control queue membership, force selection rules, retries, seed replacement, or reporting denominator.

## 14. Limitations

- All feasibility labels and all formal tests use physical root 200002. The formal paths are held out, but the physical root is not.
- Eight path seeds x three frictions provide 24 paired contexts, not 24 independent physical roots.
- The exact post-query handoff and first action chunk are matched; later online replanning intentionally diverges with each branch's observations.
- Stage-I was budget-capped at 16 of the originally precommitted 32 contexts; its motion-landscape result is kept separate and explicitly qualified.
- Intermediate and contact-loss summaries are post-hoc descriptive metrics. The frozen full-task predicate remains the only success outcome.
- Wilson intervals and exact McNemar p-values quantify sampling uncertainty over the fixed path-friction population; they do not repair the absence of root-level replication.

## Bottom line

The formal run is technically complete and auditable. ActiveForcing achieved 3/24 full-task successes at a mean selected setpoint of 4.946 N; Fixed-8 achieved 7/24 and Nominal achieved 10/24. AF therefore demonstrates a lower-force operating point in a small set of successes, but the formal data do not support an overall reliability advantage. Interpretation must remain at the same-root held-out-path level. The separate 16-context Stage-I study supports motion-dependent force feasibility, with its budget revision clearly disclosed.
