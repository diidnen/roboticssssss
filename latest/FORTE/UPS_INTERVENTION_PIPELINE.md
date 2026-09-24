# Reconstructed UPS intervention pipeline

Scope: intervention audit only. Evidence is separated into **EXECUTABLE** (present in the frozen repository/submodules) and **DOCUMENTED** (README/paper source). “Exact runtime code” is unavailable because `scripts/run_ups` is absent and `diffusion4robotics` is private.

## Two different intervention decisions must not be conflated

UPS has a high-level ACT / ASK / LEARN decision and, after learning, a low-level learned residual gate. They are not the same classifier.

### High-level UPS steering (partially executable, integration absent)

```text
current observation + user instruction
              |
              v
base diffusion policy samples K action futures
              |
              v
world model imagines their observation futures
              |
              v
VLM narrates outcomes; “none of the above” is added
              |
              v
calibrated prediction set (implementation absent)
       /              |                 \
singleton action   set size > 1       singleton “none”
     ACT               ASK                 LEARN
execute candidate   clarify intent     request teleoperation
```

The available code supports the base-action, world-model, narration grouping, and “not listed” portions: `BasePolicy.get_action_seq` ([`policy/sim_policy.py:161-170`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:161)), `WMPredictor.predict` ([`policy/wm_predictor.py:212-218`](/home/exouser/FORTE/_official_ups/policy/wm_predictor.py:212)), and `VLMVideoTranslator.process` ([`policy/vlm_video_translator.py:476-600`](/home/exouser/FORTE/_official_ups/policy/vlm_video_translator.py:476)). The conformal-set and closed-loop portions are absent.

### Learned low-level intervention policy inside re-deployment (paper-only reconstruction)

```text
observation o_t ------------------------------+
                                               |
base policy --> base action a_base,t ----------+----> classifier p_t
                                               |          |
                                               |     gate at time t
                                               |          |
                                               +----> residual delta_t
                                                          |
if gate negative: a_exec,t = a_base,t                     |
if gate positive: a_exec,t = combine(a_base,t, delta_t) <-+
                         |
                         v
              candidate-policy next step
                         |
                         v
                 repeat decision at t+1
```

For UPS, the paper says `o_t` is a decoded latent observation while imagining combined-policy candidates; for the DAgger baselines it is a real observation. UPS forms five combined-policy and five base-only candidates, the high-level verifier selects a candidate, and a selected action sequence is executed until the next verification phase ([`method.tex:300-304`](/home/exouser/FORTE/_ups_sources/paper/sections/method.tex:300), [`appendix.tex:427`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:427)). The missing driver prevents determining whether the low-level gate is also recomputed from live real observations during physical execution.

The paper states that `combine` is base plus residual in meaning, but the exact executable arithmetic, scaling order, threshold, and clipping are not public. No fixed 24-step residual activation is documented.

## B1. Intervention classifier

| Question | Code-audit answer | Labeled triangulation |
|---|---|---|
| Exact implementation | **NOT_DETERMINABLE_FROM_CODE.** No class or function is public; expected private `diffusion4robotics`. | Paper Appendix “Residual Policy.” |
| Inputs | **NOT_DETERMINABLE_FROM_CODE.** | Paper: camera images, proprioceptive state, and current base-policy action. Modified ResNet-18 per camera; image features plus proprio through a 2-layer MLP with 512 hidden units; base action through one linear layer, then concatenate ([`appendix.tex:419-423`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:419)). |
| Output | **NOT_DETERMINABLE_FROM_CODE.** | Paper: 1-D output followed by sigmoid, interpreted as intervention probability ([`appendix.tex:423`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:423)). |
| Positive label | **NOT_DETERMINABLE_FROM_CODE.** | Paper: 1 while the human is operating, 0 otherwise—not “intervention causally improves terminal success” ([`appendix.tex:417`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:417)). |
| Positive/negative construction | **NOT_DETERMINABLE_FROM_CODE.** | Paper implies timestep labels from human-control status. README routes `corrections.hdf5` through `process_corrections` then classifier buffer conversion, but both implementations are missing ([`README.md:129-149`](/home/exouser/FORTE/_official_ups/README.md:129)). |
| Uses timestamps / segments / terminal outcomes / counterfactuals? | **NOT_DETERMINABLE_FROM_CODE.** | Paper says human-operation timestamps. No paper statement that reward, terminal success, outcome improvement, or paired counterfactual branches label the classifier. |
| Class imbalance | **NOT_DETERMINABLE_FROM_CODE.** | README says compute `pos_weight`; paper says positive samples are weighted ([`README.md:138-147`](/home/exouser/FORTE/_official_ups/README.md:138), [`appendix.tex:423`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:423)). Exact value is not published. |
| Loss | **NOT_DETERMINABLE_FROM_CODE.** | Paper: binary cross entropy with positive weighting. |
| Threshold | **NOT_DETERMINABLE_FROM_CODE.** | No classifier threshold is published in accessible code/config. README `qhat=0.7` is the **high-level conformal verifier threshold**, not proven to be the low-level classifier threshold ([`README.md:84-90`](/home/exouser/FORTE/_official_ups/README.md:84)). |
| Calibration | **NOT_DETERMINABLE_FROM_CODE.** | No documented classifier calibration. High-level VLM calibration must not be attributed to this gate. |

Bottom line: the documented positive target is “human had control here.” It is not a causal benefit label. That distinction is paper-level, not code-certified in the incomplete release.

## B2. Runtime gating

### Executable finding

**NOT_DETERMINABLE_FROM_CODE.** The decisive runtime file `scripts/run_ups` is absent. There is no public residual wrapper or classifier call site. Therefore the audit cannot code-confirm:

- per-timestep re-evaluation;
- threshold comparison;
- fixed or minimum active duration;
- hysteresis/debounce;
- explicit handoff;
- additive arithmetic;
- scaling/clipping order; or
- whether switching can chatter.

### Documented behavior

The paper states that the classifier is queried “at each timestep”; a positive prediction queries the residual and produces “combined base and residual actions,” otherwise it produces the base action alone ([`appendix.tex:427`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:427)). For UPS candidate imagination those inputs are decoded latent observations. The high-level system executes a selected action sequence until the next verifier call. It does not describe a fixed residual-active duration, minimum duration, termination head, hysteresis, smoothing, or handoff operation.

The documented best reconstruction is therefore:

| Property | Documented UPS | Confidence |
|---|---|---|
| Gate frequency | every timestep while producing the combined-policy candidate; live execution cadence unknown | paper statement; not executable verification |
| Trigger latch | none described | absence in paper is not code proof |
| Fixed intervention horizon | none described | not code-determinable |
| Minimum active duration | none described | not code-determinable |
| Return to base | implicit on the next negative gate within the combined candidate; exact physical execution semantics unknown | paper inference |
| Handoff operation | none described | not code-determinable |
| Blend | base plus learned residual in conceptual definition | paper statement |
| Scaling/clipping | residual labels normalized to `(-1,1)` and residual head uses tanh; execution ordering unknown | paper statement / code unknown |
| Rapid switching risk | possible under a raw per-step gate, but threshold/hysteresis code is unavailable | inference, not observed fact |

This does **not** match “trigger once, replace the base with V1 for 24 steps, then hand back.”

## B3. Residual policy

| Item | Code-audit answer | Labeled triangulation |
|---|---|---|
| Implementation path | **NOT_DETERMINABLE_FROM_CODE**; expected private `diffusion4robotics`. | README names config `finetune_residual_policy` ([`README.md:148-149`](/home/exouser/FORTE/_official_ups/README.md:148)). |
| Inputs | **NOT_DETERMINABLE_FROM_CODE.** | Paper: current camera images, proprioception, and base action. |
| Target | **NOT_DETERMINABLE_FROM_CODE.** | Paper: `delta_a = a_human - a_base` at each correction timestep ([`appendix.tex:417`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:417)). |
| Direct vs residual action | **NOT_DETERMINABLE_FROM_CODE.** | Paper explicitly says residual delta, not direct human action. |
| Temporal operation | **NOT_DETERMINABLE_FROM_CODE.** | Paper describes a feed-forward per-timestep MLP head, not a duration-conditioned/recurrent policy. |
| Dataset construction | **NOT_DETERMINABLE_FROM_CODE.** | README: correction HDF5 → `process_corrections` → `combine_traj_to_buffer.py --residual`; implementations absent. |
| Loss | **NOT_DETERMINABLE_FROM_CODE.** | Paper: MSE; tanh output; batch 64, LR `1e-4`, 10k iterations, dropout 0.2, base-action embedding 128, action MLP 2×256 ([`appendix.tex:423-445`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:423)). |
| Terminal success supervision | **NOT_DETERMINABLE_FROM_CODE.** | Not described in paper; documented objective is timestep delta MSE. |
| Failed correction inclusion | **NOT_DETERMINABLE_FROM_CODE.** | No public filtering code or correction dataset. |
| Correction-quality filtering | **NOT_DETERMINABLE_FROM_CODE.** | No public evidence. |
| Duration awareness | **NOT_DETERMINABLE_FROM_CODE.** | No duration input/target described. |
| Termination signal | **NOT_DETERMINABLE_FROM_CODE.** | None described; documented classifier supplies the per-step gate. |

## B4. Human correction / LEARN pathway

### High-level decision

The paper says a singleton conformal set containing only “none of the above” triggers low-level interactive imitation learning; UPS randomly selects one candidate trajectory and asks a human to correct potential failures ([`method.tex:328-339`](/home/exouser/FORTE/_ups_sources/paper/sections/method.tex:328)). The public calibrated-verifier and request loop are absent.

### Low-level correction record

Implementation status: **NOT_DETERMINABLE_FROM_CODE**. The absent `run_ups` and `process_corrections` own the record schema.

Paper/README triangulation:

- human uses a SpaceMouse (keyboard is documented as an alternative);
- base control is relinquished while human control is active;
- base actions continue to be computed without being deployed;
- per-step residual target is `human - base`;
- per-step intervention labels are 1 during human control and 0 otherwise;
- the human manually hands authority back;
- 40 corrections are collected in the described simulation workflow.

What cannot be established from code: HDF5 keys, whether pre/post states are included, terminal outcome/reward fields, segment start/end format, duration metadata, episode-success filtering, correction-quality filtering, and exact alignment between action/observation timestamps.

## C. Original simulation downstream

### Task and environment

The paper identifies `Robomimic NutAssemblySquare`; the pinned environment class is `robosuite.environments.manipulation.nut_assembly.NutAssemblySquare`, which forces `single_object_mode=2`, `nut_type="square"` ([`nut_assembly.py:685-692`](/home/exouser/FORTE/_official_ups/robosuite/robosuite/environments/manipulation/nut_assembly.py:685)).

| Property | Established evidence |
|---|---|
| Robot/task | Paper: Franka Emika Panda, square nut onto square peg, handle-left vs handle-right modes. Environment code provides the square-only task. |
| Cameras | Base wrapper uses `agentview_image` and `robot0_eye_in_hand_image`; paper says both are 96×96 for diffusion policy ([`sim_policy.py:112-125`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:112), [`appendix.tex:343-350`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:343)). |
| Proprioception | Wrapper constructs eef position (3), quaternion (4), gripper qpos (2), total 9 ([`sim_policy.py:107-113`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:107)). Base checkpoint config says `use_obs:false`; paper says classifier/residual use proprio. |
| Action | 7-D is executable fact (`BasePolicy` assertion and HF config). Paper describes Cartesian pose plus gripper. Exact controller mapping/scaling in `run_ups` is unavailable. |
| Base architecture | Image-conditioned diffusion U-Net with separate modified ResNet-18 camera encoders; action chunk 16; two-frame visual history. |
| Demonstrations | Paper: 120 total, 60 handle-left and 60 handle-right ([`experiments.tex:7-14`](/home/exouser/FORTE/_ups_sources/paper/sections/experiments.tex:7)). README says “around 100” only for creating a new dataset; that is not the reported experiment. |
| World-model data | Paper: 120 demos + 480 base rollouts, then 100 additional interleaved-policy rollouts. |
| Initial randomization | Pinned authors' robosuite fork samples square-nut x in `[-0.115,-0.11]`, y in `[0.11,0.225]`, z offset `-0.04`, rotation `(2.8797905, 3.4033895)` radians ([`nut_assembly.py:407-423`](/home/exouser/FORTE/_official_ups/robosuite/robosuite/environments/manipulation/nut_assembly.py:407)). Robot initialization defaults include noise unless overridden; exact UPS kwargs are unavailable. |
| Success predicate | Environment `on_peg`: x/y within 0.03 m and sufficiently low; `_check_success` also requires the gripper be far enough (`r_reach < 0.6`) ([`nut_assembly.py:368-381`](/home/exouser/FORTE/_official_ups/robosuite/robosuite/environments/manipulation/nut_assembly.py:368), [`nut_assembly.py:614-638`](/home/exouser/FORTE/_official_ups/robosuite/robosuite/environments/manipulation/nut_assembly.py:614)). Paper evaluation additionally requires the instruction-correct handle mode. Exact mode predicate is absent. |
| Episode horizon | **NOT_DETERMINABLE_FROM UPS code.** Robosuite class default is 1000 at 20 Hz; generic RoboMimic square PH registry uses 400. Missing `run_ups` / dataset metadata could override either. |
| Verifier phases | Paper: `M=2`, grasp then place; `K=10` candidates per phase. |
| Trials/seeds | Paper: calibration 80 scenarios, test/evaluation 40 scenarios, split 20 straightforward / 20 ambiguous; closed-loop success averaged over 20 trials per category. Exact random seeds are not published. |

### Official primary metrics

No UPS evaluation script is public, so these are paper-defined, not code-verified:

- **Coverage**: fraction of test scenarios whose complete ground-truth option set is contained in the prediction set.
- **Clarification rate**: fraction with prediction-set size greater than one.
- **Set size**: number of returned options.
- **Task success rate**: true positives / all samples, where TP means the selected action leads to the instruction-correct outcome.
- **Human intervention rate**: for each trajectory, human intervention steps divided by trajectory length, averaged over trajectories; a clarification-only question counts as one intervention step ([`experiments.tex:182-210`](/home/exouser/FORTE/_ups_sources/paper/sections/experiments.tex:182)).
- **Diagnostic confusion rates**: TP, TN (correctly asks for help), FP (selected action fails), FN (unnecessary help), shown for simulation in [`appendix.tex:318-335`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:318).

No code-defined calibration error metric, episode-length metric, or additional intervention-efficiency metric was found. `model_based_irl_torch/common/utils.py::evaluate` is a generic helper and contains an apparent inverted success test (`reward == 0` increments success); it is not cited by an available UPS driver and must not be treated as the official evaluator ([`common/utils.py:137-159`](/home/exouser/FORTE/_official_ups/model_based_irl_torch/common/utils.py:137)).

## F. Exact-state paired counterfactual feasibility

### Verdict: feasible with a state-complete harness, not with MuJoCo state alone

The pinned stack exposes:

- `EnvRobosuite.get_state()`: XML + flattened MuJoCo state ([`env_robosuite.py:363-373`](/home/exouser/FORTE/_official_ups/robomimic/robomimic/envs/env_robosuite.py:363));
- `EnvRobosuite.reset_to()`: reload XML, set flattened state, forward simulation ([`env_robosuite.py:162-212`](/home/exouser/FORTE/_official_ups/robomimic/robomimic/envs/env_robosuite.py:162)); and
- `MjSim.get_state()`: time, qpos, qvel; `set_state_from_flattened()` restores those fields ([`binding_utils.py:1152-1181`](/home/exouser/FORTE/_official_ups/robosuite/robosuite/utils/binding_utils.py:1152)).

This covers object and robot qpos/qvel because they are in MuJoCo state. Camera images are derived from fixed model geometry plus simulator state; no separate camera-state snapshot is needed for fixed task cameras.

It does **not** cover all branch state:

- environment counters (`timestep`, `cur_time`, `done`) and cached success/observable state;
- controller goals and buffers such as OSC `goal_pos`, `goal_ori`, orientation reference, interpolators, and gripper goal;
- base-policy `act_history`, `prev_img`, and `prev_obs` ([`sim_policy.py:86-89`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:86), [`sim_policy.py:138-175`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:138));
- classifier/residual state (unknown until implementation is available);
- Python, NumPy, Torch CPU/CUDA, diffusion-sampler, world-model, and option-grouping RNG state; and
- remote Gemini nondeterminism, if the high-level verifier is re-called inside branches.

### Minimal robust implementation

1. Freeze environment XML/kwargs, all model commits/checkpoints, language instruction, horizon, and random seeds.
2. Use two fresh environment and policy instances per pair.
3. Restore the same initial XML/MuJoCo state and replay the same recorded prefix actions to `t0` in both instances. Prefix replay reconstructs controller goals/buffers more safely than copying only qpos/qvel.
4. Clone base-policy action history, image history, and all local RNG states at `t0`; verify pixel/proprio and next base action equality before branching.
5. Record and freeze each branch's high-level verifier transcript. Do not allow repeated remote calls to add uncontrolled variation.
6. Branch C runs the official base policy only. Branch I runs the complete official re-deployment path: five combined-policy and five base-only imagined candidates, official verification/selection, execution until the next official verifier phase, and repetition at the released cadence.
7. Continue to the identical fixed evaluation horizon and score both the environment predicate and instruction-consistent handle orientation.
8. Reject a pair if pre-branch observations, policy histories, controller outputs, or a no-op one-step determinism check differ beyond a fixed tolerance.

This enables `BOTH_SUCCESS`, `INTERVENE_ONLY`, `CONTINUE_ONLY`, and `BOTH_FAILURE`. Implementing it requires no simulator modification, but a true **official UPS** experiment still requires the missing classifier/residual code, weights, exact threshold, integration, and evaluator.
