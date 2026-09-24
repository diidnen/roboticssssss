# Corrected handoff — mass and joint physics

Mass and joint friction×mass remain required under `ACTIVEFORCING_FULL_CLAIM_V2_UTILITY`.

## Compatible work

- Mass asset audit, P4-B identifiability, development roots 6100–6105, heldout roots 6200–6201, and task0 qualification roots 7000/7001 are selector-independent.
- Preserve these outputs and their provenance. They are development/qualification evidence, not final mass adaptation evidence.
- Do not duplicate the running/completed task0 mass qualification.

## Required correction

Every learned mass-aware or joint controller must select:

`argmax_F E_{z~b_phi(z|Dq)}[p_D(success|x,z,F)*(Fmax-F)/Fmax + (1-p_D)*(-1)]`.

No runtime rho is allowed. Empirical mass/joint force frontiers remain evaluation-only.

## E8a

After a task passes mass-sensitivity qualification, freeze mass belief, Direct, candidates, force bounds, and Utility. Compare nominal/friction-only, point mass, posterior mass, Fixed-Max, Success-Only, and Full Utility. Report identification calibration plus Full SR, mean/max force, under/excess, delayed failure, realized Utility, and decision changes.

## E8b

Run a fresh 3×3 friction×mass factorial with the same initial-state and task semantics. Compare point joint, posterior joint, friction-only, mass-only, and fixed baselines. Report both MAEs, cross-confusion, joint decision accuracy, interaction effect, SR, force, under/excess, and Utility.

The old `POOLED_JOINT_NOVISUAL`/“Joint” neural architecture remains rejected and cannot be used as the joint-physics method.

## Split warning

Roots 6200/6201 have already been used as development heldout. They are not fresh locked TEST. Reserve new mass/joint roots after the final query, belief, task, Utility, and candidate contract is frozen.
