# E3 reserve long-horizon task and demo audit

## Why reserve expansion may be required

The 720-row archive has 720/720 LocalLift positives and cannot support a non-degenerate LocalLift classifier. Long-horizon task5 achieved all three regimes on TRAIN after its preregistered support extension, but that support did not transfer: held-out root7500 was `LOCAL_FAILURE` at all fixed DEV anchors. Task5 therefore failed its robust-transfer gate and activated the frozen reserve order. The earlier “in-domain”/boundary language was invalid because E1 task5's 5 N Fmax belongs to a different suite and object.

## Outcome-independent reserve order

1. LIBERO-10 task2: turn on stove AND put moka pot on it.
2. LIBERO-10 task8: place both moka pots on stove AND turn on stove.
3. LIBERO-Goal task3: open top drawer AND put bowl inside.

The order comes from the pre-existing reserve list and task/evaluator structure, not ActiveForcing performance. Task2 is tried before task8 because it has one manipulated object and two independent final goals; task8 retains a stricter three-goal, two-object fallback. Goal-task3 provides a distinct drawer-manipulation fallback.

The project-local configs contain explicit conjunctive goals. The generic `libero_goals_reached` evaluator ANDs them for final success. Task2 and task8 also use the explicit stove `turnon` range `(0.5, 2.1)`; goal-task3 requires the top-drawer `open` condition plus object-in-drawer. The current project-local evaluator uses a broadened wooden-cabinet open interval `(-0.22, -0.12)` rather than its commented historical `(-0.20, -0.14)` range. This is recorded as evaluator lineage, not changed for E3, and every method would share it.

Static wrapper QA used the isolated Tabero commit `80ab3be09ce884f86cfc2037d3af30bc28061426`. The historical B5 client SHA-256 remains `3b38317efad4ee0b2d9e1bfc4379198bbe0d2eb146665e170070b3ad82ec0a18`; the transformed source compiled successfully at 93,541 bytes. Candidate USD assets are present in the immutable project data tree: `moka_pot.usd`, `flat_stove.usd`, `akita_black_bowl.usd`, and `wooden_cabinet.usd`. The launcher reads code/config from the isolated worktrees and raw assets/HDF5 only from `/media/volume/newdata/exouser/tabero/data/Isaaclab_Libero`.

## Data availability and lineage

A filesystem search under `/media/volume/newdata/exouser` initially found no local HDF5/demo for these three reserve candidates. The only locally assembled `libero_goal` file found was goal-task1 (`put the bowl on the stove`), which is a different task and may not be renamed or substituted. No unrelated HDF5 was renamed or substituted.

After all frozen candidates failed robust-force nominal qualification, two exact book-to-caddy sources were acquired. The upstream LIBERO HDF5 is 7D/robosuite and is reference-only because it lacks the real tactile/force fields required by the authoritative π0. The project-native `NathanWu7/Isaaclab_Libero` source was then frozen at revision `fe6263d3ff1f600f123522e298d6d755bc827d33`; it provides the exact task5 Isaac assembled states plus a fixed 18-success replay manifest. This activates, but does not yet complete, the project's validated `8D assembled -> Isaac tactile replay -> 13D 7dpf -> strict Tabero conversion` route. Full hashes, schemas, selection rule, and the five-demo gate are in `E3_ONBOARDING_SOURCE_COMPATIBILITY_AUDIT.md` and `E3_TASK5_5DEMO_ONBOARDING_PLAN.json`.

No few-demo training has started. The first action is a single selected-demo tactile replay smoke under a fresh GPU/MASS gate. The staged schedule remains 5 demos, then 10 only if nominal DEV success remains inadequate, then at most 20. Demos teach nominal execution only; no force-selector, ActiveForcing, or TEST outcome supervision is admissible.

## Scientific boundary

The 8 N qualification setpoint is not a new-task Fmax and is not an ActiveForcing result. No new-task Utility experiment may begin until task-specific force support/Fmax is frozen from authoritative safety/controller/config lineage or a TRAIN/DEV-only protocol. The final selector remains Expected Utility with the frozen existing config hash `c5bc4f39861a84b9ba95d55cdb5f8633a8b004d205b3864a02799340653d9b10`; hard-rho is prohibited.

## Qualification update

Reserve task2 completed one valid TRAIN nominal episode at root7600, friction0.6, diagnostic slot8N. Frozen pi0 failed before grasp: grasp/lift/transport/place/turn/FullTask were all zero, with both authoritative goals `[0,0]` over 800 steps. The exact result and hashes are in `E3_RESERVE_TASK2_QUALIFICATION_RESULT.md`.

Task8 also completed as a valid nominal failure: all three goal flags remained zero for 800 steps and frozen pi0 did not grasp. Its exact result/ownership/hashes are in `E3_RESERVE_TASK8_QUALIFICATION_RESULT.md`.

LIBERO-Goal task3 then completed as the third valid nominal failure: the query state was not reached, the bowl did not move materially (`1.54e-7 m` maximum XY displacement), open/grasp/lift/place were zero, and both final goal flags remained zero for 800 steps. Thus the complete frozen reserve list failed nominal capability under the unchanged π0.

Current status: `ALL_FROZEN_LONGHORIZON_CANDIDATES_AUDITED_ONBOARDING_REPLAY_SMOKE_RESOURCE_GATED`. The next authorized scientific step is the preregistered five-demo task5 benchmark onboarding path, beginning with exactly one selected-demo tactile replay smoke. No Utility/Fmax claim is attached to onboarding.
