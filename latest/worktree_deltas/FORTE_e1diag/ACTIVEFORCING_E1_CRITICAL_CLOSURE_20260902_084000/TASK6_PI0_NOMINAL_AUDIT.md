# Task6 pi0 nominal-capability audit

## Gate decision

`PI0_REPAIR = NOT_NEEDED`

This status means the repair authorization gate was not triggered; it does **not** mean E1 proves native pi0 task6 capability is high.

The authoritative E1 archive runs a scripted post-query downstream branch (`p5s0a_true_matched_dataset.py`) after a matched reset. It does not execute the frozen pi0 nominal policy from reset to completion. Consequently, Fixed-Max 36/36 is strong evidence that the task6 post-query state, force support, controller, and scripted downstream path can succeed at robust force, but it is not a native-pi0 nominal SR estimate.

Within the evidence that actually measures the E1 anomaly, the three Query-Ignored-success -> Active-failure cases are explained downstream of the query by force choice: GT friction preserves the failing choice, while a higher archived force succeeds. There is no affirmative evidence that `UPSTREAM_PI0_FAILURE` is the primary task6 bottleneck. The required precondition for few-demo repair is therefore false.

No demos were collected, no checkpoint was trained, no checkpoint was changed, and no onboarding artifacts were emitted. If a later native-pi0, same-root robust-force E2E audit shows nominal task6 failure, onboarding must use TRAIN/DEV demos only and the resulting frozen checkpoint must be shared by all task6 methods.

Evidence gap: a faithful native-pi0 robust-force task6 E2E dataset is required to estimate reach, grasp pose, transport, orientation, placement, semantic-execution, and evaluator-specific failure rates. Existing task6 E5 smoke did not produce rollouts. This gap blocks a positive nominal-capability claim but does not authorize repair.
