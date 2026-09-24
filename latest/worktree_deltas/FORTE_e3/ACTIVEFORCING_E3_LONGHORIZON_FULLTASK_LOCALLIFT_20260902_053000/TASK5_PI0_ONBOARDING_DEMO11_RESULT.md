# E3 task5 onboarding replay — demo 11

Status: `DEMO11_REAL_TACTILE_7DPF_GATE_PASS`.

Frozen TRAIN demonstration 11 was replayed once in the project-native `Isaac-Libero-Franka-Replay-Camera-Tactile-v0` environment with the `7dpf` recorder. It completed the semantic task successfully in 200 steps. The exported action tensor is `[200,13]`; `eef_pose`, `gripper_pos`, real `gripper_net_force`, and real `gripper_marker_motion` are present, finite, and nonzero where physically expected. The source and replay initial-state digests match exactly.

All four required media streams decode to 200 frames: agent view, wrist view, left tactile RGB, and right tactile RGB. The HDF5 SHA-256 is `6f0e122d60ae05ea56a20da8d0dbebfca48b567fc6359ac73d7aa64e574d2baa`; the independent gate is `TASK5_PI0_ONBOARDING_DEMO11_GATE.json`.

This is benchmark task onboarding data only. It uses no TEST root, ActiveForcing outcome, Utility target, Fmax inference, synthetic tactile, or force-selector supervision. It does not change the authoritative π0 checkpoint or Utility hash.
