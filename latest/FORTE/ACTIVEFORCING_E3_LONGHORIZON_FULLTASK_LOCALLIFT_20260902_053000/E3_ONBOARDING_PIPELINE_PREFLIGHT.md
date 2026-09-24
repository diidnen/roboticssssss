# E3 five-demo onboarding pipeline preflight

Status: **CPU PREFLIGHT PASS; FINAL 13D REPLAY DATA STILL RESOURCE-GATED.**

- Task: `libero_10/task5`, black book to the back compartment of the caddy.
- Source: the five preregistered successful project demos `[1, 2, 11, 12, 19]` from `NathanWu7/Isaaclab_Libero`; no ActiveForcing or TEST outcome selected them.
- Authoritative initialization: project-local OpenPI `pi0_lora_tacfield_tabero`, checkpoint step `49999`, commit `1ed9cf44c05bc63fa3b3dbc0ca83ce9dbc8b7b2e`.
- Isolated worktree: `/home/exouser/FORTE/agent_lanes/Tabero_VTLA_e3_onboarding_20260902`, branch `activeforcing-e3-task5-onboarding`.
- Frozen onboarding code commit: `b3239ba85281b691242e34534dcd36fffb6e78cf`.
- The v2.1 compatibility conversion was tested on all 5 demos: 5 episodes, 901 frames, one exact language task, 7D state and 7D control actions.
- CPU full-transform smoke passed with batch 8: three 224x224 RGB slots, 32D padded state, 50x32 action chunks, no synthesized tactile field, and exactly zero values in padded action dimensions.
- Final training is **not** authorized from this preflight dataset. Agent B's preregistered final path first replays each source demo in Isaac to obtain real 13D action/force and tactile-marker observations, performs strict schema QA, and then recomputes norm stats in a new timestamped directory.
- The onboarding loss is restricted to the first seven control dimensions (`tactile_loss_weight=0`, `padding_loss_weight=0`), so benchmark semantic adaptation cannot train against a chosen force setpoint or rewrite padding targets.
- At deployment the optional tactile adapter uses exactly the original `TaberoTacFieldInputs` flattening when real `tactile_marker_motion` is present; visual, language, action normalization, chunk semantics, and inference rate remain project-local OpenPI.

Preflight hashes:

- `libero_policy.py`: `340b71dacf75e2e73eab3bb00179e783ce49e42f5ea9d9911334cff7494f765d`
- `config.py` after final staged 7dpf config: `296b01a51c985583bab688402808296315bd449ddd2aa343f4f5458ba1d1295e`
- 7D preflight norm stats: `8063afdc6f25d44446a0d6f1c29a681237a76a04330a2a216ee5924880c249a3`

This is benchmark task onboarding, not ActiveForcing training. The Utility config and selector are untouched.
