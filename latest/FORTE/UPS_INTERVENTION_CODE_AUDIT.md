# UPS intervention code audit

Audit date: 2026-08-29 UTC  
Scope: **INTERVENTION only**. This audit does not import VLS or the separate active-probe project.

## Executive finding

The public release is not sufficient to establish the exact learned-intervention implementation from executable code. The top-level repository says that the classifier, residual policy, correction processing, and deployment pipeline live in `diffusion4robotics/` and `scripts/`, but:

- `diffusion4robotics/` is a gitlink to an inaccessible repository;
- the documented `scripts/` directory is absent from the frozen commit;
- the public `policy/` directory contains a base-policy wrapper, a world-model wrapper, and a VLM translator, but no residual-policy wrapper, intervention classifier, calibrated verifier, `run_ups`, correction processor, or evaluation driver;
- the public Hugging Face model repository contains one base diffusion-policy training run, not identifiable classifier and residual checkpoints; and
- the README-linked base-demonstration repository is unavailable, while the residual world-model dataset repository is empty except for `.gitattributes`.

Consequently, implementation questions whose answers live in those unavailable files are marked **NOT_DETERMINABLE_FROM_CODE**. Paper and README statements are retained below as explicitly non-executable triangulation, never as code proof.

## A. Frozen provenance

| Item | Frozen value |
|---|---|
| Repository | `https://github.com/CMU-IntentLab/uncertainty_aware_policy_steering.git` |
| Local read-only audit checkout | `_official_ups/` |
| Commit | `f7bad4d7cd6d5e4350751a958ba9bb146a99ecec` |
| Branch | `main` |
| Commit subject/date | `add submodules`, 2026-06-03 |
| Remote | `origin https://github.com/CMU-IntentLab/uncertainty_aware_policy_steering.git` |
| Status after inspecting accessible submodules | top level clean except intentionally uninitialized `diffusion4robotics` gitlink |
| Tags/other remote branches | none observed; only `origin/main` |

Pinned submodules are declared in [`_official_ups/.gitmodules`](/home/exouser/FORTE/_official_ups/.gitmodules:1) and the root tree:

| Submodule | Pinned commit | Availability at audit |
|---|---|---|
| `diffusion4robotics` | `c347ddfdd969a0393f4e4d4f9b918b44bb281ad7` | **Unavailable**: unauthenticated clone asks for GitHub credentials |
| `robomimic` | `e10526b9a40c78b41f1e37e60041dc0ec0a5f60f` | Available and checked out detached |
| `robosuite` | `a3f2467375e434aa3c74eeea52e5d592a0cdc4d6` | Available and checked out detached |

No scientific source was modified. Downloaded paper/config evidence is separate from the checkout under `_ups_sources/` and `_ups_hf/`.

## Release-completeness and reproducibility findings

1. The README assigns base policy, intervention classifier, and residual policy training to `diffusion4robotics` and correction processing / pipeline execution to `scripts` ([`README.md:23-29`](/home/exouser/FORTE/_official_ups/README.md:23)). The former is inaccessible and the latter does not exist in the commit.
2. The documented commands refer to absent files: `combine_traj_to_buffer.py`, `compute_class_weights.py`, `finetune.py`, `scripts/generate_calibration_data.py`, `scripts/run_calibration`, `scripts/run_ups`, and `scripts/process_corrections` ([`README.md:96-160`](/home/exouser/FORTE/_official_ups/README.md:96), [`README.md:178-182`](/home/exouser/FORTE/_official_ups/README.md:178)).
3. The README says `policy/` contains base/residual wrappers and a conformal verifier ([`README.md:25-28`](/home/exouser/FORTE/_official_ups/README.md:25)); the frozen directory contains only `sim_policy.py`, `wm_predictor.py`, `vlm_video_translator.py`, and an empty `__init__.py`.
4. `VLMVideoTranslator` imports three prompt files through author-local absolute paths; one of them, `mcq_verifier_prompt.txt`, is not in the repository ([`policy/vlm_video_translator.py:39-45`](/home/exouser/FORTE/_official_ups/policy/vlm_video_translator.py:39)). The module therefore is not portable as committed.
5. `WMPredictor` imports `dreamer.prediction.ClassifierLatentTrainer`, but no `dreamer/prediction.py` exists in the public tree ([`policy/wm_predictor.py:21-29`](/home/exouser/FORTE/_official_ups/policy/wm_predictor.py:21)). It also uses author-local normalization paths ([`policy/wm_predictor.py:54-61`](/home/exouser/FORTE/_official_ups/policy/wm_predictor.py:54)).
6. The README installation names `environment.yaml`, while the committed file is `environment.yml`, and it names a root `requirements.txt` that is absent ([`README.md:42-50`](/home/exouser/FORTE/_official_ups/README.md:42)).

These are audit findings, not evidence that the authors' internal implementation lacks the described behavior.

## Relevant environment and dependency information

The committed Conda environment is Python 3.9.23 and pins, among others, NumPy 1.26.4, SciPy 1.13.1, MuJoCo 3.3.6, robosuite 1.5.1, PyTorch 2.5.1, torchvision 0.20.1, diffusers 0.35.1, transformers 4.46.2, Hydra 1.2.0, Google GenAI 1.47.0, h5py 3.14.0, OpenCV 4.11.0.86, and WandB 0.22.0 ([`environment.yml:39-55`](/home/exouser/FORTE/_official_ups/environment.yml:39), [`environment.yml:98-186`](/home/exouser/FORTE/_official_ups/environment.yml:98), [`environment.yml:306-368`](/home/exouser/FORTE/_official_ups/environment.yml:306)).

The pinned source commits, rather than the `robosuite==1.5.1` environment entry, are authoritative for this audit.

## Checkpoints and data artifacts

The README links:

- diffusion/residual: `https://huggingface.co/jzyuan04/ups_nutassembly_diffusion_checkpoints`
- world model: `https://huggingface.co/jzyuan04/ups_nutassembly_wm_checkpoint`
- base demos: `https://huggingface.co/jzyuan04/ups_nutassembly_base_policy_demos`
- world-model data: `https://huggingface.co/datasets/jzyuan04/ups_nutassembly_wm_data`
- residual world-model data: `https://huggingface.co/datasets/jzyuan04/ups_nutassembly_wm_data_residual`

Observed Hub state on the audit date:

| Artifact | Revision / observed content | Finding |
|---|---|---|
| `ups_nutassembly_diffusion_checkpoints` | `17ad9ee2efb432e71e3d0e1ac730f08a77e9cf57` | 19 checkpoints, steps 5k–95k, plus one Hydra/config bundle. The config is a base `DiffusionUnetAgent`; no separately named classifier/residual config or checkpoint is present. |
| `ups_nutassembly_wm_checkpoint` | `6e7ecaa1656dad88e05b0fe2dd8e3ff83dfe18db` | `pretrain_joint_250000.pt` and `pretrain_joint_160000_residual.pt`. |
| `ups_nutassembly_base_policy_demos` | unavailable (401/not found as model or dataset) | Cannot inspect demonstrations or embedded environment metadata. |
| `ups_nutassembly_wm_data` | `904d157f1421e7c8c3579466ef67186f259c4935` | One HDF5 file and a two-line license card. |
| `ups_nutassembly_wm_data_residual` | `84326dc0ee5836736e875447005f708b9f10f04f` | Only `.gitattributes`; no rollout data. |

The downloaded base checkpoint config identifies `data4robotics.models.diffusion_unet.DiffusionUnetAgent`, separate ResNet-18 visual features, two cameras, action dimension 7, action chunk 16, two images per camera, 100 training diffusion steps, 16 evaluation diffusion steps, dropout 0.1, and `use_obs: false` ([`_ups_hf/diffusion_configs/agent_config.yaml:1-29`](/home/exouser/FORTE/_ups_hf/diffusion_configs/agent_config.yaml:1)). Training used behavior cloning, AdamW at `1e-4`, batch 64, seed 292285, and 100,000 iterations ([`_ups_hf/diffusion_configs/.hydra/config.yaml:58-92`](/home/exouser/FORTE/_ups_hf/diffusion_configs/.hydra/config.yaml:58)). This is base-policy evidence only.

`BasePolicy` declares wrapper defaults `PRED_HORIZON=128` and `ACT_HORIZON=128` ([`policy/sim_policy.py:18-20`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:18)), but the published network config outputs a 16-action chunk. Slicing a 16-action output at 128 still yields 16; the missing driver may also pass overrides. The scientifically relevant released model value is therefore 16, while the exact deployment override is not code-determinable.

The README states `qhat=0.7` ([`README.md:84-90`](/home/exouser/FORTE/_official_ups/README.md:84)), 40 collected corrections ([`README.md:129-136`](/home/exouser/FORTE/_official_ups/README.md:129)), and class weighting via `compute_class_weights.py` / `agent.pos_weight` ([`README.md:138-149`](/home/exouser/FORTE/_official_ups/README.md:138)). The implementation and exact resulting classifier threshold are unavailable.

## Exact code-path inventory

| Concern | Available path / symbol | Audit conclusion |
|---|---|---|
| Base policy | `policy/sim_policy.py::BasePolicy` | Available. Loads the Hydra agent/checkpoint, two cameras and proprio, maintains image/action history, denormalizes 7-D actions ([`sim_policy.py:35-92`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:35), [`sim_policy.py:107-170`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:107)). |
| Intervention classifier | expected in `diffusion4robotics` | **Unavailable; no public class/function path.** |
| Residual policy | expected in `diffusion4robotics` and a wrapper in `policy/` | **Unavailable; no public class/function path.** |
| Correction collection | expected `scripts/run_ups` | **Absent.** |
| Correction preprocessing | expected `scripts/process_corrections` and `diffusion4robotics/combine_traj_to_buffer.py` | **Absent/private.** |
| Training | expected `diffusion4robotics/finetune.py`, configs `finetune_classifier`, `finetune_residual_policy` | **Private.** Base-only config is available on HF. |
| Inference/deployment | expected `scripts/run_ups` | **Absent.** |
| RoboMimic evaluation | expected pipeline script | **Absent.** Generic upstream wrappers are available but are not the UPS evaluator. |
| World model | `policy/wm_predictor.py::WMPredictor`; `model_based_irl_torch/dreamer/dreamer.py::Dreamer`; `.../models.py::WorldModel` | Available but not runnable as committed due missing import/local paths. `WMPredictor.predict` passes proposed actions to the world model ([`wm_predictor.py:118-223`](/home/exouser/FORTE/_official_ups/policy/wm_predictor.py:118)). |
| Narrator / uncalibrated selector | `policy/vlm_video_translator.py::VLMVideoTranslator` | Available. Groups narrations by keywords, samples options, adds “not listed,” and queries Gemini ([`vlm_video_translator.py:243-381`](/home/exouser/FORTE/_official_ups/policy/vlm_video_translator.py:243), [`vlm_video_translator.py:476-600`](/home/exouser/FORTE/_official_ups/policy/vlm_video_translator.py:476)). It does **not** implement the documented conformal set construction. |
| Conformal verifier | expected in `policy/` / `scripts/run_calibration` | **Absent.** |
| Environment | `robosuite/.../nut_assembly.py::NutAssemblySquare`; `robomimic/.../env_robosuite.py::EnvRobosuite` | Available at pinned commits. |

## What executable code establishes

- The base wrapper consumes `agentview_image`, `robot0_eye_in_hand_image`, and a 9-D concatenation of end-effector position, quaternion, and gripper qpos ([`policy/sim_policy.py:107-136`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:107)). The supplied base checkpoint config sets `use_obs: false`, so the frozen base network is image-conditioned despite the wrapper constructing proprioception.
- The wrapper holds open-loop action chunks in `act_history` and re-samples only when empty ([`policy/sim_policy.py:138-151`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:138)); the HF config sets chunk length 16.
- Actions are denormalized to seven dimensions using `ac_norm.json` and asserted to be 7-D ([`policy/sim_policy.py:153-159`](/home/exouser/FORTE/_official_ups/policy/sim_policy.py:153)).
- The public VLM component explicitly reasons over narrated candidate outcomes and a “not listed” option, rather than a passive current-observation classifier. This is the **high-level ACT/ASK/LEARN** mechanism, not the learned low-level intervention gate.
- `VLMVideoTranslator` defaults to `gemini-3-flash-preview`, four workers, and constructor temperature 1.0 ([`policy/vlm_video_translator.py:46-68`](/home/exouser/FORTE/_official_ups/policy/vlm_video_translator.py:46)); its multiple-choice query hard-codes temperature 0.0 ([`policy/vlm_video_translator.py:383-395`](/home/exouser/FORTE/_official_ups/policy/vlm_video_translator.py:383)). It returns an empty log-probability dictionary in this path, so it is not the missing calibrated verifier.
- No public executable source establishes classifier labels, threshold, switching, residual addition, action clipping, correction filtering, or termination.

## Paper/README triangulation (not executable evidence)

The paper source says:

- the high-level system forms `K=10` candidate outcomes and a “none” option, calls the verifier twice in simulation (`M=2`), and treats singleton-action / multi-option / singleton-none sets as execute / clarify / request correction;
- during human correction the base policy continues to run without execution and records `delta = human - base`, with intervention labels 1 during human control and 0 otherwise;
- classifier and residual inputs are observation (camera plus proprio) and base action;
- the classifier is a sigmoid one-dimensional output trained with weighted BCE;
- the residual regresses normalized delta action with MSE and tanh output;
- at deployment the classifier is queried each timestep and the combined base/residual action is used when positive; and
- UPS samples five imagined sequences from the combined policy and five from the base policy to mitigate overcorrection.

These statements are in [`sections/method.tex:293-339`](/home/exouser/FORTE/_ups_sources/paper/sections/method.tex:293) and [`sections/appendix.tex:414-445`](/home/exouser/FORTE/_ups_sources/paper/sections/appendix.tex:414). They strongly indicate that a fixed 24-step policy replacement is not the documented UPS design, but the missing runtime code prevents code-level confirmation.

## Audit conclusion

The official public implementation, as frozen, cannot support a code-certified claim about learned intervention duration, handoff, classifier target, residual composition, or causal supervision. The correct code-audit result is a bounded one: the **documented** UPS mechanism is per-step gated residual addition, whereas the **public executable release** omits the implementation needed to verify it. Any scientific claim about UPS itself must either obtain the missing pinned submodule/scripts/checkpoints or be labeled as a paper-level claim rather than a code-audit fact.
