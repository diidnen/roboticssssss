# V3 method contract

V1 and V2 are immutable negative engineering evidence. V2's historical stop decision is preserved; its physical conclusion is unresolved because normalized-action saturation, state/contact drift, online replanning, and pre-step force/action row alignment confounded it.

R1 is an exact-state one-control-step counterfactual. The canonical online pi0 prefix is saved once. Each branch reconstructs that exact prefix and must pass state, controller, contact, observation, nominal-action, and direction parity before it may execute. Force samples are labeled `pre_action` and `post_action`; the latter is the causal response. A branch is invalid, not a negative result, on parity failure, clipping, or contact loss.

R2 may only follow an R1 diagnosis that identifies V2's engineering confound. It uses a local adapter, never a shared robosuite edit. A disabled adapter must match untouched OSC strictly. Newton units are a hypothesis until contact measurements validate them.

A3 uses only previous completed-step force for feedback. Online pi0 replanning after a legitimate task branch is logged as divergence and never treated as calibration pairing. Every failure, raw contact sum, and actuator limit event is retained.
