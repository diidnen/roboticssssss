# Official UPS versus the current PEG_INSERT intervention

## Representation verdict

**MAJOR_DEVIATION**

The current design—`R20 → replace with V1 for exactly 24 steps → explicit handoff to R20`—is not representative of the **documented** UPS learned-intervention mechanism. UPS is described as keeping the base action in the loop, predicting an additive residual, and re-evaluating an intervention classifier at every timestep while constructing combined-policy candidates. Its high-level UPS verifier then reasons over multiple imagined candidate futures and separates ACT, ASK, and LEARN.

This verdict is based on the paper/README description. The public release omits the runtime implementation, so per-step switching, exact addition/clipping, and the absence of hidden latch logic cannot be certified from executable code. That evidence limitation does not make a fixed policy-replacement horizon a close match.

## D. Side-by-side comparison

| Feature | Official UPS | Current MetaWorld PEG_INSERT | Scientifically important difference? |
|---|---|---|---|
| Base policy | Image-conditioned stochastic diffusion U-Net; two cameras; action chunk 16; frozen checkpoint for residual learning. Executable base wrapper/config available. | Frozen R20. | **Yes.** Different policy family, observation modality, temporal chunking, and task distribution. |
| High-level intervention mechanism | Documented ACT / ASK / LEARN choice from calibrated VLM prediction sets over narrated, world-model-imagined futures. Integration code absent. | Passive learned router chooses Continue vs Rescue from current observation/history. | **Yes.** UPS first asks whether candidate futures satisfy the language-conditioned task; PEG router predicts a binary route. |
| Learned low-level intervention policy | Documented additive residual conditioned on observation and base action. Executable implementation unavailable. | V1 replaces R20. | **Yes—major.** Residual correction preserves a base-action reference; replacement does not. |
| Intervention-classifier input | Paper: camera images + proprioception + base action. | Current-observation MLP or GRU history; no stated base-action conditioning. | **Yes.** Action conditioning can disambiguate risk at the same state. |
| Intervention-classifier label | Paper: human-control timestep label (1 during teleoperation, 0 otherwise). Code unavailable. | Paired route labels / passive success routing as described by the current experiment. | **Yes.** Behavioral intervention imitation is not causal benefit supervision. |
| Action candidates/futures considered | High-level UPS: yes, `K=10` imagined base/combined-policy outcomes. Low-level classifier itself is documented as local observation + base action. | No explicit candidate-future comparison in the passive MLP/GRUs. | **Yes.** Do not conflate UPS high-level futures with its low-level intervention label. |
| Trigger frequency | Paper: low-level classifier every timestep while producing combined candidates; high-level verifier twice in NutAssembly (grasp/place). Whether the gate is recomputed on live observations during execution is code-unknown. | Single trigger followed by 24 forced rescue steps. | **Yes—major.** |
| Duration | No documented fixed residual-active horizon; within combined candidates, activation emerges from successive gate outputs. A selected action sequence runs until the next high-level verification phase. Exact code unavailable. | Exactly 24 rescue steps regardless of gate state. | **Yes—major and likely an artifact risk.** |
| Handoff | No documented explicit handoff; next negative gate implicitly selects base-only action. | Explicit transfer to R20 after step 24. | **Yes.** BAD_HANDOFF cannot automatically be attributed to UPS. |
| Action composition | Paper: combined base and residual; residual target `human - base`. Exact scaling/clipping code unavailable. | Whole-policy replacement with V1. | **Yes—major.** |
| Temporal smoothing / hysteresis | Not stated; code unavailable. | Fixed horizon itself supplies a latch/debounce. | **Yes.** UPS switching/chatter cannot be evaluated from public code. |
| Terminal-outcome supervision | Not code-determinable. Paper describes weighted BCE and delta-action MSE, not terminal-success losses. | Paired terminal outcomes are measured; passive models failed despite them. | **Yes.** Evidence source and objective differ. |
| Counterfactual supervision | Not code-determinable. No paired Continue/Intervene training is described. | Exact paired branches produce 541 / 285 / 101 / 370 categories. | **Yes—central.** |
| Task-success supervision | High-level verifier calibration uses user-labeled correct candidate outcomes; low-level residual/classifier are documented as imitation losses. Exact code unavailable. | Terminal success is the route evaluation criterion. | **Yes.** UPS has semantic outcome labeling at a different level, not necessarily low-level causal intervention value. |
| Intervention cost | Paper metric: intervention steps / trajectory length; clarification-only counts as one step. | Current reported primary results emphasize success and collateral categories; no stated optimized intervention-cost objective. | **Yes.** |
| Downstream task | RoboMimic / robosuite `NutAssemblySquare`, Panda, square nut, left/right handle modes. | MetaWorld `PEG_INSERT`. | **Yes.** Contact dynamics and task semantics differ. |
| Primary metrics | Paper: coverage, clarification rate, set size, task success/TP, intervention rate, confusion categories. No public UPS evaluator. | paired outcome categories; Always Rescue 63.69%; oracle 71.47%; passive routing success. | **Yes.** Paired causal estimand is new relative to documented UPS evaluation. |

## What the current results do and do not establish about UPS

The 1,297 valid PEG_INSERT pairs establish a genuine intervention-selection problem in the current system: 285 states benefit only from rescue while 101 succeed only under continuation. The 7.78 percentage-point gap between Always Rescue (63.69%) and the Continue-vs-Rescue oracle (71.47%), plus 101 collateral failures, motivates counterfactual selection. The current-observation MLP reached 60.76%, GRU-8 reached 63.15%, and GRU-32 matched Always Rescue at 63.69%, so the tested passive routers did not recover that oracle gap.

They do **not** establish that UPS has a 24-step handoff problem. Forty-three unnecessary overrides, 37 bad handoffs, 13 contact-induced failures, and eight grasp destabilizations were observed under a mechanism UPS is not documented to use.

They also do not prove that a UPS-like per-step residual gate solves the problem: the public implementation is incomplete, and a per-step behavioral classifier can still misroute or chatter.

## E. Hypothesis verdicts

The required verdicts are code-audit verdicts. Because the relevant executable implementation is unavailable, paper statements are included only as triangulation.

| Hypothesis | Verdict | Code evidence | Paper/README triangulation |
|---|---|---|---|
| H1: UPS uses a fixed intervention duration and therefore has a handoff-timing problem. | **NOT_DETERMINABLE_FROM_CODE** | `scripts/run_ups` and residual/classifier sources are unavailable. | The paper instead says the classifier is queried each timestep; no fixed horizon/handoff is described. Thus the existing 24-step failure is likely a current-system artifact, not evidence against UPS. |
| H2: UPS learns when to intervene by imitating human intervention labels rather than estimating causal benefit. | **NOT_DETERMINABLE_FROM_CODE** | Label construction and trainer are private/absent. | Paper explicitly defines 1 during human operation and 0 otherwise, with weighted BCE. This supports the statement at paper level. |
| H3: UPS does not explicitly compare downstream outcomes under Continue versus Intervene. | **NOT_DETERMINABLE_FROM_CODE** | The integrated high-level sampler and low-level deployment code are absent. | Paper compares multiple imagined base and combined-policy candidates semantically, but does not describe paired identical-state terminal Continue-vs-Intervene labels. This makes the strong wording subtle: UPS considers futures, but not the requested causal paired estimand. |
| H4: UPS intervention learning does not optimize terminal task success directly. | **NOT_DETERMINABLE_FROM_CODE** | Private training code and data schema. | Paper describes classifier BCE and residual delta MSE, not terminal-success optimization. High-level verifier calibration uses correct-option labels, a separate objective. |
| H5: UPS residual intervention can cause collateral damage, but its training objective does not explicitly model counterfactual risk. | **NOT_DETERMINABLE_FROM_CODE** | Residual trainer and integration unavailable. | Paper acknowledges possible overcorrection/mode bias and mixes five combined-policy with five base-only candidates; documented delta MSE does not model paired harm explicitly. |
| H6: UPS only determines whether help is needed, but does not explicitly optimize how long intervention should persist. | **NOT_DETERMINABLE_FROM_CODE** | Duration/termination logic is in absent runtime code. | Paper's per-step classifier can induce adaptive duration implicitly, so “only determines whether help is needed” is too coarse. No explicit duration target or termination head is described. |

Machine-readable versions are in `UPS_HYPOTHESIS_VERDICTS.json`.

## G. Candidate research gaps

“Supported by official code” means established by the public executable release. A “no” can mean the release is insufficient, not that the private implementation disproves the gap.

| Candidate | Supported by official code? | Supported by PEG paired results? | Novelty relative to documented UPS | 24-step artifact risk | Priority |
|---|---:|---:|---|---|---|
| **A. Counterfactual intervention value** | **No**—not establishable from incomplete code. | **Yes.** 285 INTERVENE/RESCUE_ONLY versus 101 CONTINUE_ONLY and a 7.78 pp oracle gap. | High. Documented low-level target is human-control imitation; high-level future verification is not paired causal benefit learning. | Low. Paired benefit remains meaningful under per-step residual gating. | **1 — highest** |
| **D. Collateral-risk-aware intervention** | **No**—not establishable from incomplete code. | **Yes.** 101 continuation-only successes and explicit collateral categories. | High. Documented UPS mitigates overcorrection by retaining base-only candidates, but does not describe a counterfactual harm target. | Medium. Some observed harm is specific to replacement/handoff, but CONTINUE_ONLY is mechanism-agnostic. | **2** |
| **C. Action-conditioned consequence reasoning** | **No**—public world-model code accepts action sequences, but the integrated comparison is absent. | **No.** Passive failures motivate it but do not isolate consequence reasoning. | Medium. UPS already considers imagined candidate futures at the high level and includes base action in the low-level classifier; the novelty must be causal candidate comparison, not merely adding action input. | Low–medium. | **3** |
| **B. Intervention duration / termination** | **No**—runtime code unavailable. | **Yes for the current mechanism.** Handoff/damage categories and forced 24 steps make duration relevant. | Medium if an explicit option/termination objective is absent; paper already describes implicit per-step adaptive gating. | **High.** The strongest evidence may disappear under documented UPS-like gating. | **4** |
| **E. Gate calibration** | **No**—classifier threshold/calibration unavailable. | **No.** Passive collapse does not by itself diagnose calibration rather than representation/label mismatch. | Low–medium. UPS's conformal calibration applies to the high-level VLM, not shown for the low-level gate. | Medium. | **5 — lowest** |

## Real gap

The best-supported research gap is **counterfactual intervention value**, with collateral risk as its negative side:

```text
Delta(s, a_base) = P(instruction-correct terminal success | official intervention)
                 - P(instruction-correct terminal success | continue base)
```

This is more defensible than a duration claim. It survives the change from 24-step replacement to per-step residual gating, distinguishes helpful from harmful intervention at identical states, and is not equivalent to the paper-described label “human was controlling.”

It is not yet a paper contribution claim. The next step is to test whether this gap exists under the actual official NutAssembly residual mechanism.
