# Explicit SysID Channel Audit

**Explicit response-curve identification is available, but direct Coulomb slip-threshold identification is not.**

The 46D legal trace contains measured normal/tangential force, `Ft/Fn`, bilateral contact flags/state, force imbalance, gripper opening, commanded tangential increment, accumulated displacement, marker motion/tangential displacement/velocity, EEF displacement, phase, and tactile validity. The fixed probe remains in bilateral contact, but no authoritative slip-onset flag or saturated Coulomb threshold is recorded; observed `Ft/Fn` is much smaller than GT μ. Therefore `max(Ft/Fn)` cannot be interpreted directly as μ.

The explicit baseline predefines four one-dimensional physical response summaries: peak `Ft/Fn`, peak tangential force, tangential force per marker displacement, and probe-out force-asymmetry variation. For each LOTO turn it selects one formula by grouped source-root CV, fits only an affine sensor-scale/nuisance calibration on source tasks, clips to the legal [0.2,1.0] range, freezes it, and evaluates the untouched target task. It is non-neural and uses no task ID, target label, validation outcome, or GT μ at inference.

Selected source-only formulas:
- target task0: `probe_asymmetry` (source-CV selection only)
- target task1: `probe_asymmetry` (source-CV selection only)
- target task5: `probe_asymmetry` (source-CV selection only)
- target task6: `probe_asymmetry` (source-CV selection only)
