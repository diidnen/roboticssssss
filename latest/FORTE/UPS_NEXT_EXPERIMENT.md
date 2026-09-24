# One recommended next scientific experiment

## Paired Continue-versus-documented-UPS residual pilot on NutAssemblySquare

This is the **only** recommended next experiment. Do not train a new model and do not test duration variants first.

### Why this experiment

The current PEG_INSERT evidence mixes at least two possible effects:

1. a general failure to estimate whether intervention improves the terminal outcome; and
2. artifacts of replacing R20 with V1 for a forced 24-step segment and handing control back.

The clean discriminator is to remove that fixed-horizon mechanism and measure paired outcomes using the official learned residual gate on UPS's original downstream task.

### Prerequisite / current release blocker

The experiment must not be represented as “official UPS” until the following pinned artifacts are obtained from the authors or made public:

- `diffusion4robotics` commit `c347ddfdd969a0393f4e4d4f9b918b44bb281ad7`;
- the missing `scripts/run_ups` and evaluation/correction-processing scripts at root commit `f7bad4d7cd6d5e4350751a958ba9bb146a99ecec`;
- the intervention-classifier and residual-policy checkpoints/configs;
- the classifier decision threshold and exact action composition/clipping logic; and
- the instruction-consistent NutAssembly success evaluator and actual horizon/seed configuration.

The currently public Hugging Face diffusion repository contains a base-policy run but no identifiable classifier/residual checkpoints. Running an inferred reimplementation now would test a reconstruction, not the official mechanism.

### Hypothesis

Even after replacing the 24-step policy replacement with the official mixed base/combined candidate pipeline and its documented per-timestep residual gating, the learned mechanism will sometimes select a path whose residual intervention reduces instruction-correct terminal success. A paired Continue-versus-Intervene oracle will therefore exceed the deployed mechanism, establishing a counterfactual intervention-value gap independent of fixed handoff timing.

### Environment

- Pinned authors' robosuite `NutAssemblySquare` at commit `a3f2467375e434aa3c74eeea52e5d592a0cdc4d6`.
- Pinned RoboMimic wrapper at `e10526b9a40c78b41f1e37e60041dc0ec0a5f60f`.
- Same Panda/controller, two 96×96 cameras, initial-state distribution, left/right instructions, horizon, and evaluator as the released UPS evaluation.
- Headless deterministic rendering; remote VLM transcripts recorded once per branch and frozen for replay.

### Frozen components

- base diffusion checkpoint and normalization;
- intervention classifier, threshold, and all preprocessing;
- residual policy and action scaling/clipping;
- world model and high-level verifier outputs;
- user instruction;
- environment/controller configuration;
- evaluation horizon; and
- all Python, NumPy, Torch CPU/CUDA, diffusion-sampler, and environment seeds.

No model is updated within or between branches.

### Independent variable

Post-`t0` control from the same state:

- **Continue (C):** official base policy only for the remainder of the episode.
- **Intervene (I):** the complete official UPS re-deployment path: generate the released `K/2=5` classifier-gated combined-policy candidates and `K/2=5` base-only candidates, apply the official high-level verifier, execute its selected plan until the next released verification phase, and repeat at the released cadence.

There is no forced 24-step duration and no custom handoff.

### State selection

Use a pilot of **40 valid `t0` states**, sampled from held-out base-policy rollouts and stratified across:

- grasp versus place phase;
- pre-contact, contact, and post-grasp regimes; and
- classifier scores below, near, and above its released threshold.

The score strata are diagnostic only; do not choose states from terminal outcomes.

### Paired branch construction

For each `t0`:

1. Instantiate two fresh copies of the pinned environment and policies.
2. Restore the same initial XML and flattened MuJoCo state.
3. Replay the identical recorded prefix into both copies up to `t0`; this reconstructs controller goals/buffers that qpos/qvel alone do not capture.
4. Clone base-policy action chunks, previous images/observations, classifier/residual state if any, and all RNG states.
5. Use common random numbers for base-policy candidates shared by C and I. Record I's world-model, narration, prediction-set, clarification, and VLM transcript; replay that transcript rather than re-querying a remote model during diagnostics.
6. Verify equal observations and equal next base action before branching. Reject the pair on mismatch.
7. Run C and I to the same official terminal horizon. I must use the released mixed-candidate/verification semantics, including decoded latent inputs to the residual gate if that is what the recovered driver implements.

### Primary metric

The paired four-way terminal table:

- `BOTH_SUCCESS`
- `INTERVENE_ONLY`
- `CONTINUE_ONLY`
- `BOTH_FAILURE`

Report the paired net intervention benefit

```text
(INTERVENE_ONLY - CONTINUE_ONLY) / valid_pairs
```

with an exact paired/binomial confidence interval and McNemar test on discordant pairs.

### Diagnostic metrics

- deployed official-gate success rate;
- base-only success rate and paired oracle success rate;
- classifier score and gate state at every timestep;
- selected candidate identity and whether it is base-only or combined;
- number of gate transitions, active-step fraction, and longest active run;
- residual norm and clipped-action fraction;
- first divergence time between branches;
- phase/contact regime at first gate;
- task completion versus correct left/right handle orientation;
- episode length; and
- manual/blinded taxonomy for each `CONTINUE_ONLY`: unnecessary override, contact-induced damage, grasp destabilization/drop, switching/chatter, base-distribution shift, or other. `BAD_HANDOFF` is used only if the official code actually contains a handoff.

### Exact terminal success/failure criterion

- **Success:** the official NutAssembly evaluator reports the square nut successfully on the peg **and** the final handle orientation satisfies that trial's left/right instruction before the official fixed evaluation horizon.
- **Failure:** either placement or instruction-consistent orientation fails by that horizon.
- **Invalid pair:** any mismatch before branching in XML, MuJoCo state, controller/policy history, rendered observations, next base action, instruction, or seed schedule. Invalid pairs are excluded and counted explicitly; the pilot is not interpretable if more than 5% are invalid.

### Experiment-level decision criterion

The counterfactual-value hypothesis is supported in this pilot if all are true:

1. at least one `CONTINUE_ONLY` case occurs under the official mechanism;
2. paired oracle success exceeds deployed official-gate success; and
3. at least half of `CONTINUE_ONLY` cases are not attributable to a reconstruction mismatch or a non-official modification.

It is provisionally not supported if `CONTINUE_ONLY = 0` and the upper 95% bound on its rate is below 7.5% (with 40 valid pairs, zero events gives an approximate rule-of-three upper bound of 7.5%). Otherwise the pilot is inconclusive and should inform sample-size planning, not a contribution claim.

### Compute estimate

- 40 prefixes replayed in two instances and 80 branch continuations, with the normal `K=10`, `M=2` UPS imagination workload in Branch I.
- No training.
- Expected budget: approximately **8–16 GPU-hours** on one modern GPU for diffusion inference, world-model imagination, and headless simulation, excluding one-time environment setup. The exact runtime depends on the released horizon and logging.
- Approximately 80 normal verifier-phase calls for Branch I (`40 × M=2`), plus any clarification calls. Transcripts are recorded once and reused; there are no repeated calls for reruns of the same branch.

### Interpretation

| Outcome | Scientific interpretation |
|---|---|
| Material `CONTINUE_ONLY` and oracle > official gate | Counterfactual benefit/collateral risk is a genuine UPS-relevant gap, not just a 24-step artifact. Prioritize value/risk supervision. |
| `CONTINUE_ONLY` disappears, while PEG handoff failures were common | The strongest prior weakness was likely caused by fixed replacement/handoff. Do not claim it against UPS. |
| Both `INTERVENE_ONLY` and `CONTINUE_ONLY` are rare | Intervention has little leverage on the selected states; revise state sampling before method development. |
| Many rapid gate transitions correlate with failures | Duration/smoothing becomes a defensible secondary hypothesis, but only after verifying exact official threshold logic. |
| Harm appears mainly from residual clipping or action composition | Audit/repair reproduction fidelity first; do not interpret it as a classifier-value gap. |
| Pair-replay invalidity exceeds 5% | The harness is not state-complete. Fix controller/policy/RNG restoration before any scientific conclusion. |

This experiment maximally separates a true causal intervention-value weakness from the current 24-step handoff artifact while preserving the audit-first constraint.
