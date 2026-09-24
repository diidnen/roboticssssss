# STATUS

COMPLETE — the preregistered conditional workflow stopped before context-model training and before Probe because neither global calibration nor the deployment-legal context oracle passed its trigger/gate.

# SINGLE SCIENTIFIC GOAL

Determine whether the remaining continuous Feasibility-only error is mainly a global probability-scale error or missing GNP-style observable context x, without collecting force/repeat data or reopening architecture search.

# CONNECTION TO GNP

GNP conditions feasibility on observable context x, hidden dynamics z, and candidate action a. Here z≈friction μ and a=continuous grip force F were already present; this run tested whether x was too impoverished. We borrowed this conditional decomposition, not the Neural Process architecture.

# CURRENT INPUT AUDIT

Actual tensors contain H=8 nominal π0 Cartesian motion, seven task phases, four-way task identity, candidate force, μ, strict pre-probe state/mask, and gripper opening. They do **not** contain RGB, visual/language embeddings, object identity/geometry, grasp/object pose, or a π0 latent/context token. No hidden VLA state is exposed by the websocket interface.

# GLOBAL CALIBRATION

| Feas variant | Probability MAE | Brier | NLL | Frontier MAE | Under-force | Finite decisions | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| RAW | 0.245 | 0.086 | 0.463 | 0.325N | 2/8 | 6/8 | FAIL |
| PLATT | 0.133 | 0.038 | 0.318 | 0.087N | 4/8 | 8/8 | FAIL |
| TEMPERATURE | 0.210 | 0.084 | 0.474 | 0.171N | 1/8 | 7/8 | FAIL |
| ISOTONIC | 0.134 | 0.044 | 0.334 | 0.094N | 3/8 | 8/8 | FAIL |

All mappings were fit on the 1,008 TRAIN Bernoulli outcomes only and applied to both the 27 real DEV points and every 0.05N query before thresholding. Raw Feas mean signed bias is -0.186; it is globally conservative. Platt and isotonic correct probability/frontier error but cross the safety boundary too early. Temperature preserves more of the raw ordering/safety direction but misses both the 0.20 probability threshold and 0.10 under-force threshold.

# IS THE MODEL SIMPLY UNDERCONFIDENT?

**NO.** It is underconfident, but a single global scale cannot satisfy probability accuracy and safety simultaneously. The best probability calibration (Platt, MAE 0.133) creates 4/8 under-force decisions.

# CONTEXT-DEPENDENT RESIDUALS

Context identity explains 34.0% of DEV residual variance descriptively, but task already explains 16.3% and the deployment-observable pre-probe pose association is weak (maximum |Spearman| across eef xyz = 0.262). These are nine contexts/27 cells, so this is structure evidence, not a causal attribution.

# SAME FRICTION, DIFFERENT CONTEXT

Six preregistered same-task pairs met |Δμ|≤0.03 and |ΔF|≤0.06. Four have a 0.20 real-probability difference. In every pair the old backend's normalized inputs were already distinguishable through nominal motion/opening; therefore these pairs do not show that the backend collapses truly identical existing inputs. Absolute eef_xyz is absent, but its observed variation is small.

# CONTEXT-ID ORACLE

The deployment-legal family oracle was built from TRAIN-only task-specific pre-probe `[eef_x,eef_y,eef_z,opening]` centroids; DEV mapped to the nearest TRAIN centroid within task, never to a DEV/root/split ID. It improves probability MAE by only 8.8% (0.245→0.223), below the 30% trigger. Frontier MAE improves (0.325N→0.175N), but under-force worsens from 2/8 to 5/8; Brier and NLL also worsen (0.086→0.121, 0.463→0.657). The oracle therefore does not support missing observable grasp context as the main bottleneck.

# WHAT OBSERVABLE CONTEXT IS AVAILABLE FROM THE FROZEN VLA?

Agent-view/wrist RGB are available to π0 online but were not archived as aligned inputs for this population. Hidden/visual/recurrent embeddings are not exposed by the frozen websocket interface. Task identity, phase, nominal π0 motion, and opening are already used. The only new frozen, deployment-observable channel available without new collection was pre-probe eef_xyz; privileged object pose was explicitly excluded.

# CONTEXT-AUGMENTED FEASIBILITY

NOT EXECUTED. The preregistered oracle trigger failed; training a new model would be result-driven feature fishing.

# CONTEXT-AUGMENTED JOINT

NOT EXECUTED for the same reason. Joint architecture search was not reopened.

# DOES CONTEXT FIX PROBABILITY ESTIMATION?

**NO.** The legal context-family diagnostic delivered only 8.8% relative MAE gain and materially worsened safety.

# DOES JOINT HELP AFTER BOTH MODELS GET THE SAME CONTEXT?

**EVIDENCE LIMITED.** No repaired context representation passed the trigger, so the matched Context Feas/Joint comparison was scientifically not reached. This run neither revives nor kills Joint.

# GT CONTINUOUS GATE

**FAIL.** No Feas calibration family simultaneously achieved probability MAE≤0.20, frontier MAE≤0.20N, under-force≤0.10, finite decisions≥80%, and monotonic response. The oracle also failed. Gate coverage of real frontiers remained 8/9 (88.9%), but model reliability failed.

# PROBE VS STRICT NO-PROBE

NOT REACHED. The frozen rule forbids Probe evaluation after GT gate failure; no active-probe continuous claim is made.

# PRIMARY_CLASSIFICATION

**MODEL_ERROR_REMAINS_AFTER_CONTEXT_AND_CALIBRATION**

The dominant remaining blocker is representation/model error beyond the tested global calibration and legal pre-probe grasp-context factor. Global underconfidence is real but secondary and insufficient; missing context was not supported by the preregistered oracle.

# SECONDARY_PROBE_CLASSIFICATION

**NOT_REACHED**

# WORLD-MODEL / JOINT STATUS

**EVIDENCE_LIMITED** — the fair same-context comparison was not triggered.

# WHAT IS NOW PROVEN

- The raw Feas model is globally conservative on this frozen repeated DEV benchmark.
- TRAIN-only global mappings can reduce probability/frontier error, but currently trade it for unacceptable under-force.
- A deployment-legal pre-probe grasp-context family does not explain enough error to justify context-model retraining.
- No variant passes the GT continuous reliability gate.

# WHAT IS STILL NOT PROVEN

- No cross-object claim.
- No unseen-task claim.
- No original TEST was loaded.
- No fresh E2E experiment.
- No when-to-probe agent.
- No conclusion about Joint after a genuinely effective common context repair.
- Five-repeat empirical probabilities remain finite-sample estimates, not exact physical truth.

# METHOD IMPLICATION

Do not freeze the active-probe continuous method yet. The candidate form remains conceptually `Frozen VLA context x + friction belief + F → full-task feasibility → minimum reliable force`, but the present x/representation does not support reliable rho=0.80 thresholding.

# NEXT_METHOD

Audit which **deployment-time physical/context variable** is still missing before collecting more force data. Priority should be an explicitly archived, decision-time object/grasp observation (aligned RGB/object-region representation or deployable geometry/pose estimate) and a model diagnostic that can preserve safety under calibration. Do not add repeats or forces until such a variable is identified and frozen.
