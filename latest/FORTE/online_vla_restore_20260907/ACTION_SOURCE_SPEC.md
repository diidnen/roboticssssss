# Candidate online VLA action provenance

This specification describes the running development candidate, not a final runtime freeze.

| Component | Actual source / behavior |
|---|---|
| Established grasp and physical probe | Existing corrected current runtime core and P4-B probe; exact prefix reconstruction and exposed post-probe state equality |
| VLA weights | Original `pi0_lora_tacfield_tabero/.../49999`; whole-tree SHA256 `0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17` |
| OpenPI checkout | `1ed9cf44c05bc63fa3b3dbc0ca83ce9dbc8b7b2e` plus actual source-file hashes in server metadata |
| Tabero checkout | `f2c18c691face021261fa53f1a379a76b90836ed` plus actual frozen runtime source hashes |
| Server entry | Dedicated `server.py`, original `create_trained_policy`, actual `Policy.infer` / JAX `Pi0.sample_actions` |
| Observation | Original two camera views (`agentview_cam`, `eye_in_hand_cam`), original 224px padding/resize, 7D measured state, original tactile marker-history construction, official task instruction |
| Model preprocessing | Checkpoint-local normalization; original `TaberoTacFieldInputs`, tokenization/image preprocessing and model transforms |
| Raw model output | 50×32 sampler output captured before the unchanged native output transforms |
| Native postprocessing | Original output unnormalization, absolute-action transform on first six coordinates, effective 50×13 action extraction |
| Temporal execution | Execute first10 of each fresh50 chunk, then re-infer from the new observation; unused40 are discarded |
| Initial feasibility sequence | First8 XYZ targets of the freshly inferred decision-state chunk; displacement and increments computed from those targets; explicit missing VLA phase channels (zero), no scripted prefix |
| VLA arm mapping | Physical XYZ + axis-angle slots0:6 copied exactly after float32 conversion; no waypoint, interpolation, placement path, or cached trajectory |
| Gripper during grasp | Carry the last probe command, apply unchanged AF force-servo feedback; discard raw VLA aperture as an aperture reference and discard VLA force slots7:13; inject F/2 in slots9,12 |
| Release intent | Native gripper scalar is an absolute width, not a probability. A near-full-open request >=.039m maps to canonical .04m and zero active squeeze; same rule for every force method |
| Policy randomness | Request-local Gaussian initial flow noise, seed derived from context and native replan step, shared seed law across force methods; policy weights and native sampler unchanged |
| Inference provenance | Server records request/worker/context IDs, input hash, normalized-input hash, raw/postprocessed action hashes, checkpoint hash, server PID and actual inference start/end times |
| Executed-action provenance | Each physics step logs request ID, chunk index, VLA arm/gripper command, AF selected and active force, executed13D action, controller internal force reference and measured force |
| Raw observation preservation | Full per-control-step observation arrays and both raw cameras; each native RPC also preserves the exact canonical payload |

The release intent is part of manipulation semantics. It does not permit the VLA to change the AF squeeze reference while holding. Gripper masking/release interpretation remains a development qualification item until the formal runtime freeze.

The VLA metadata calls the checkpoint loaded only after the original loader succeeds. The client rejects missing receipts, wrong checkpoints, wrong shapes, mismatched hashes, duplicate/rerouted requests and non-online action sources. An independent audit joins each action to the server log and verifies exact arm equality and force replacement.

A failed online inference is an execution error, not a negative supervised label. Scripted and precomputed-trajectory fallbacks do not exist in this downstream executor.
