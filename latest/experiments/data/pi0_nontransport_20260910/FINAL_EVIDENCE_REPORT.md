# Corrected pi0 non-transport continuation: final evidence

The frozen official `pi0_libero` checkpoint was reused read-only from `/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero`, tree SHA-256 `92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103`. No checkpoint, cache, or large continuation artifact was written to that volume.

| task | archived native baseline (init 0) | state/contact audit | same-init one-property intervention | native outcome |
|---|---:|---|---|---|
| 0, open middle drawer | success, 145 steps | robot contact steps 96--144; middle drawer joint delta -0.14114 m | `model.geom_friction[wooden_cabinet_1_g22]` `[0.95,0.3,0.1] -> [0.2,0.3,0.1]` | success, 146 steps |
| 5, push plate | success, 159 steps | robot contact steps 55--158; plate translation 0.28955 m; plate/table contacts reconstructed | `model.geom_friction[table_collision]` `[0.6,0.005,0.0001] -> [2.0,0.005,0.0001]` | failure, 310 steps |
| 7, turn on stove | success (initial receipt) | robot contact steps 60--102; button joint delta 0.50842 rad | `model.dof_damping[flat_stove_1_button]` `1.0 -> 5.0` | success, 96 steps |

The audit was an explicit deterministic reconstruction of archived raw actions with no policy connection or inference. It is not presented as new policy behavior. Contact records identify direct robot-object contacts, but do not measure normal force, contact force, gripper force, or joint torque.

Protocol and property readbacks: `protocol/FROZEN_PROTOCOL.json`; audit: `audit/CONTACT_MOTION_AUDIT.json`; each new rollout directory has `receipt.json`, `actions.npy`, `agentview.mp4`, and online inference timing receipts. The original baselines remain read-only in the original result namespace.

Claim boundary: this establishes official pi0 success on three initial non-transport forms and a small controlled same-init physics robustness/contact-demand smoke. It does not establish ActiveForcing or generalized contact-force adaptation. The drawer has observed handle/body contact but no validated force proxy or interceptable/force-calibrated squeeze interface; plate pushing is surface-contact robustness only; stove uses constrained contact/rotation without torque or force sensing.
