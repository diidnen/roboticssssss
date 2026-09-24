# Direct ρ=0.8 Historical Provenance

## Finding

There is no single recovered historical executable that simultaneously implements:

`P4-B → FRICTION_GRU.pt → task0 Direct calibrated P(success) → minimum force at probability ≥0.8 → real execution`.

Three distinct historical meanings must not be conflated.

## 1. Original Bayesian ρ=0.8 predecessor

`/home/exouser/Tabero/analysis/results/p3_probe_belief_decision_20260819_104513` implements a pre-lift physical probe followed by the minimum discrete force whose posterior expected success is at least 0.8. This is executable historical evidence for the scientific principle, but it predates P4-B, uses a Bayesian probe model, and does not load `FRICTION_GRU.pt`.

## 2. Task0 visual Direct raw probability threshold

`/home/exouser/FORTE/task0_visual_generalization.py` defines `RHO = 0.80`. In `frontier_metrics`, it selects the first dense-grid force whose raw ensemble probability is at least 0.8. The matching result directory is `/home/exouser/FORTE/task0_visual_generalization_20260831_040609`.

This is executable Direct-like `p≥0.8` code. It is not the explicit friction-GRU AFI pilot, and later audit found calibration/backend weaknesses. It must not be silently substituted for a different frozen controller.

## 3. Continuous empirical F*0.8 and calibrated decision threshold 0.5

`/home/exouser/FORTE/continuous_probe_joint.py` defines:

- empirical frontier target `RHO = 0.80`;
- five repeats per force;
- reliable cell = at least 4/5 successes;
- `F*_rho` = the lowest tested reliable force;
- calibrated model selection threshold `THRESHOLD = 0.5`.

Therefore its `ρ=0.8` applies to the empirical full-task frontier label, not directly to `calibrated_probability >= 0.8`. The run `/home/exouser/FORTE/continuous_probe_joint_20260830_110712` completed 135/135 scripted branches but failed the frozen GT backend gate; Probe/NoProbe was not reached.

## 4. Explicit friction-GRU AFI decision

`/home/exouser/Tabero/analysis/active_friction_imagination_e2e.py::choose_from_curves` selects the minimum force whose mean deterministic imagined success is at least 0.9. It uses candidate forces {3.0,4.0,4.5,5.0}. This is the only recovered actual E2E that loads the exact `FRICTION_GRU.pt`, but its rule is η=0.9, not ρ=0.8.

## Conclusion

Historical executable `ρ=0.8` semantics were found for both the original Bayesian controller and the later visual Direct model/frontier label. An exact executable bridge from the recovered explicit friction GRU to the frozen task0 Direct `ρ=0.8` controller was not found. This is a method-identity/provenance question and must be resolved from the already-frozen specification or an authoritative historical checkpoint; it must not be redesigned from TEST outcomes.
