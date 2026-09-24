# STATUS

**COMPLETE WITH PREREGISTERED CONDITIONAL STOP.** The actual π0 visual pathway exists online, but no aligned strict-preprobe RGB, image token, visual feature, object crop, or restorable scene snapshot was archived for the 72 TRAIN and 9 frozen DEV contexts. The required visual-context experiment therefore cannot be run without creating a new observation dataset.

# SINGLE SCIENTIFIC GOAL

Test whether the remaining continuous-feasibility error is caused by an impoverished observable object/grasp context `x`, then test Joint under exactly the same repaired `x` and continue to Probe only if the GT gate passes.

# GNP CONNECTION

GNP conditions feasibility on observable context `x`, hidden dynamics `z`, and action `a`. Here `z≈friction μ` and `a=continuous force F` already exist. The intended new `x` was a frozen, deployment-time π0/object/grasp visual representation. This run audited whether that `x` could be aligned to the already collected outcomes; it did not assume availability from architecture documentation.

# CURRENT INPUT GAP

The current feasibility tensor contains task and phase one-hot features, H8 nominal π0 Cartesian motion, strict-preprobe gripper opening/state, μ, and F. It contains no RGB, visual token, object geometry representation, object crop, grasp-relative visual geometry, or π0 latent.

# FROZEN VLA / VISUAL CONTEXT AVAILABLE

At live deployment, yes: `agentview_cam` and `eye_in_hand_cam` supply 224×224 RGB to `TaberoTacFieldInputs`; π0 calls PaliGemma/SigLIP image feature extraction before action generation. In the current websocket deployment interface, however, only actions are returned. Hidden/image-token tensors are neither exposed nor persisted.

For this frozen historical experiment, **no**: all three authoritative result roots contain zero eligible image, video, image-array, or visual-feature files. The only aligned evidence is a strict-preprobe state hash.

# PRE-PROBE VISUAL ALIGNMENT

Coverage is **0/72 TRAIN** and **0/9 DEV**. The continuous collector captured `env.scene.get_state(...)` at the last hold, used it in memory for restores, and wrote hashes/telemetry. It did not serialize the snapshot or RGB. A state hash cannot reconstruct pixels. Rendering now would create a new observation and cannot be asserted to equal the historical decision-time frame.

# SELECTED VISUAL REPRESENTATION

None. All four preregistered precedence levels fail the historical-alignment requirement:

1. π0 visual representation: produced online but not exposed/persisted.
2. Exact backbone output: callable, but aligned RGB input is missing.
3. Recomputed pooled same-encoder feature: blocked by the same missing RGB.
4. Existing object-region representation: none is archived or deployed.

No external vision model was introduced and no layer fishing was performed.

# TRAINING FAIRNESS

The intended matched comparison was fair—same 1008 feasibility outcomes, same `x`, same μ/F/context, same seeds; Joint would receive only corrected-valid physical supervision. Because `x` is absent, this comparison cannot be instantiated. Zero new checkpoints were trained.

# OLD FEAS / OLD JOINT

The frozen old baselines remain unchanged:

| model | probability MAE | Brier | NLL | frontier MAE | under-force | finite decisions | monotonic contexts |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLD FEAS | 0.245 | 0.086 | 0.463 | 0.325 N | 2/8 | 6/8 | 9/9 |
| OLD JOINT | 0.207 | 0.135 | 0.939 | 0.125 N | 3/8 | 8/8 | 3/9 |

# VISUAL FEAS

**NOT EXECUTED.** There is no eligible aligned `x`; training would silently associate outcomes with newly generated or mis-timed images.

# VISUAL JOINT

**NOT EXECUTED** for the same common-input reason. Joint is not prejudged.

# PROBABILITY ESTIMATION

No new probability estimate exists. The frozen OLD FEAS probability MAE remains 0.245. This run cannot measure the causal effect of visual context.

# CONTINUOUS FRONTIER

No new frontier estimate exists. The frozen OLD FEAS frontier MAE remains 0.325 N with 6/8 finite decisions.

# UNDER-FORCE / EXCESS FORCE

No new safety result exists. The frozen OLD FEAS under-force rate remains 2/8; OLD JOINT remains 3/8.

# DOES VISUAL CONTEXT FIX THE ERROR?

**NOT TESTED.** The scientific hypothesis remains plausible but unvalidated. The experiment was blocked by missing aligned historical observations, not by absence of a vision encoder.

# DOES JOINT HELP AFTER BOTH MODELS SEE THE SAME CONTEXT?

**EVIDENCE LIMITED.** Neither matched visual model could be trained, so no independent Joint value can be assigned.

# GT CONTINUOUS GATE

**NOT REACHED.** There is no visual-context backend to evaluate. The existing backend already failed the frozen gate.

# PROBE VS STRICT NO-PROBE

Not run because the GT visual-context gate was not reached.

# DOES PROBE NOW IMPROVE CONTINUOUS FORCE?

**NOT REACHED.**

# QUANTIZATION UNMASKING

Not applicable because no GT-valid visual-context backend exists.

# PRIMARY_CLASSIFICATION

**DEPLOYABLE_VISUAL_CONTEXT_UNAVAILABLE**

Here “unavailable” means unavailable as an aligned feature for this frozen historical TRAIN/DEV experiment. Live deployment does have RGB and a frozen π0 visual encoder.

# JOINT STATUS

**EVIDENCE_LIMITED**

# PROBE STATUS

**NOT_REACHED**

# WHAT IS NOW PROVEN

- The deployed π0 code path genuinely uses two RGB streams and PaliGemma/SigLIP image tokens.
- The current feasibility model does not receive those pixels or tokens.
- The authoritative continuous TRAIN/DEV artifacts do not preserve aligned preprobe pixels/features or a serialized restorable snapshot.
- Consequently, the proposed frozen historical visual-context comparison is not identifiable from current artifacts.

# WHAT IS STILL NOT PROVEN

- It is not proven that too little `x` caused the feasibility error.
- It is not proven that visual context fixes probability or frontier error.
- It is not proven whether Joint helps after matched visual context.
- There is no cross-object generalization claim.
- There is no unseen-task claim.
- No original TEST was loaded.
- No fresh E2E was run.
- There is no when-to-probe agent yet.

# FINAL METHOD IMPLICATION

The desired formulation remains:

`Frozen VLA visual/object/task context x + Active Probe friction belief + continuous force F → Full-Task Feasibility → minimum reliable grip force`.

It cannot yet be frozen as the method because `x` was not archived for the training/evaluation outcomes.

# NEXT_METHOD

Do not add visual layers blindly. First make the existing deployment observation auditable: at the strict-preprobe last-hold event, persist both 224×224 RGB inputs and one preregistered pooled output from the exact frozen π0 vision backbone, together with the strict-state hash, camera configuration/hash, timestamp/step, and no-probe/no-outcome provenance. Only then repeat the same matched Feas/Joint study on a prospectively aligned population. If pixels cannot explain same-μ/same-F differences, audit deployment-inferable mass/COM/object geometry rather than using simulator-private values.
