# Verifier diagnostic and existing-data force-critical benchmark

## Plain conclusion

The old verifier looked useful because it was evaluated on a small, already-viewed fixed-scene population with more force-rescuable cases, repeat/seed averaging, and an explicit max-force fallback. In the later fully grouped-root pooled evaluation, Direct was already at 95.1% macro success and only 5/144 episodes were force-rescuable. The verifier rejected many already-successful proposals, so putting it in series after Direct reduced unconditional success and coverage.

Adding full-task success/failure supervision to the World Model representation was helpful in one earlier taskwise diagnostic, but that advantage did not reproduce in the matched pooled experiment. It is evidence for an outcome-shaped representation, not evidence that the runtime verifier improves force selection.

## What the old verifier was

- World Model: frozen Joint-lineage H8 predictor, outputting an H8×13 predicted physical trajectory.
- Outcome Verifier: `concat(final, mean, std, max)` trajectory summaries (52 dimensions) → `Linear(52,1)`.
- Training target: real final full-task success/failure.
- Training domain: predicted TRAIN trajectories, not real trajectories.
- Loss: unweighted BCEWithLogits, 80 epochs, AdamW, three seeds.
- Forbidden verifier inputs: force, task, root/context ID, friction, Direct probability, Joint latent, real future trajectory.
- Original diagnostic boundary: probability >0.8. It accepted only 1/24 Direct proposals and rejected 19 proposals that were actually perfect.
- Later hard-search boundary: native `logit>0`, upward-only search. This rule was specified after the p>0.8 diagnostic was inspected and is retrospective.

## What “explicit success/failure in the World Model” means

`WM-Current` came from Joint training and received an outcome/feasibility gradient in addition to trajectory and intervention-effect losses. `WM-PhysicsOnly` used the same architecture, inputs, H8 target, IE target, optimizer and epochs, but optimized only trajectory + IE.

Earlier taskwise grouped-CV diagnostic:

- Current outcome-shaped WM verifier AUROC: about 0.843.
- PhysicsOnly WM verifier AUROC: about 0.770.

This supported the limited statement that the representation benefited from outcome shaping.

Later matched shared-pooled grouped-root CV:

- Current AUROC 0.828; PhysicsOnly AUROC 0.830.
- Current FPR 0.331; PhysicsOnly FPR 0.319.
- Current strict control SR 0.850; PhysicsOnly 0.824.
- Current coverage 0.884; PhysicsOnly 0.856.
- Trajectory standardized MAE 0.233 for both.

Thus the discriminative advantage did not reproduce under the final pooled protocol. Outcome shaping slightly improved controller coverage/SR relative to PhysicsOnly, but neither verifier beat Direct.

## Why adding the runtime World Model made performance worse

1. **Direct was near ceiling.** Direct macro SR was 0.951. Only 5/144 controller episodes were force-rescuable, so maximum possible gain was small.
2. **The verifier is a serial veto.** Every already-correct Direct proposal can be lost through a false-negative or `NO_VALID_FORCE`. The verifier had many more opportunities to reject good proposals than to rescue the five under-force proposals.
3. **H8 does not always identify final-task success.** Later transport/rotation/placement failures and stochastic boundaries can be indistinguishable in the first H8 trajectory.
4. **Old and new evaluation grains differed.** Old DEV averaged repeats and three seed scores inside each context-force cell; pooled CV retained each repeat and each seed's Boolean decision. Averaging hid disagreement and stochastic false-negatives.
5. **Old DEV was an easier context-generalization problem.** It was fixed-scene/held-friction retrospective DEV. The new protocol held out whole root families within every task.
6. **Fallback supplied much of the old apparent gain.** Strict verifier coverage was only 19/24. The 93.8% row executed Fmax after `NO_VALID_FORCE`; it did not mean the verifier judged Fmax successful.

## Existing-data, outcome-only force-critical subsets

Membership was defined without Direct/Verifier/World Model outputs: adjacent force cells, both low-force repeats fail, both adjacent high-force repeats succeed.

### Shared pooled OOF population

- 14 contexts, 11 independent task-root families.
- task0: 6; task5: 8.
- All 14 low-force failures occurred after successful pick/lift and were downstream grip-loss cases.
- Direct nevertheless selected a high enough force in all 28 repeat episodes: SR 1.000, mean force 4.421 N.
- Strict verifier: SR/coverage 0.857, with 14.3% collateral rejection.
- Max fallback restored SR to 1.000 but did not improve over Direct.

This is a valid physical pressure subset, but it contains no Direct proposal failures and therefore gives the verifier no rescue opportunity.

### Old fixed-scene DEV

- Six strict outcome-only contexts: task0×2, task1×2, task5×2.
- Direct: SR 0.833, mean force 4.555 N.
- One-Step: SR 1.000, mean force 4.900 N.
- Fixed-Max: SR 1.000, mean force 5.138 N.
- Strict verifier: coverage 0.667, conditional SR 1.000, mean finite force 4.626 N.
- Verifier + MaxFallback: SR 1.000, mean force 4.850 N.

The favorable statement is narrow: on this retrospective six-context stress subset, Verifier+MaxFallback matched the success of One-Step and Fixed-Max, used 0.050 N less mean force than One-Step and 0.289 N less than Fixed-Max. However, strict verifier rejected 2/6 contexts and the fallback was essential. This is not evidence that the verifier alone rescued failures.

## Other-method executability audit

- **π0-Default:** a task0 full-reset runner exists, but there is no validated shared multi-task exact-snapshot runner matching the pooled branch-bank state and repeat semantics.
- **Tabero-Neutral / Oracle-Language:** the formal baselines were never executed. Oracle-Language must be marked privileged.
- **FORTE:** the available implementation is only a privileged GT-slip `FORTE-inspired` surrogate. It is not an official FORTE reproduction and was never formally executed.
- Existing baseline tables correctly contain `NOT_RUN`, not reusable metrics.

These methods cannot be computed from the saved force branches without simulation: π0/Tabero change policy actions, and reactive FORTE changes force during execution. Treating a fixed-force branch as one of these methods would be a mislabeled proxy.

## Paper-safe use

Use the pooled grouped-root result for the main method: ActiveForcing-Direct-GT, macro SR 95.1%. Use the old six-context subset only as a clearly labeled retrospective force-critical stress analysis. The strongest defensible verifier phrasing is:

> On a retrospective, outcome-defined force-critical subset, verifier plus explicit max fallback retained 100% success with slightly less force than uniform one-step escalation; the verifier alone had limited coverage and did not pass the pooled promotion gate.

Do not claim that outcome-shaped World Model supervision universally improves runtime force selection, and do not report π0/Tabero/FORTE proxy numbers as real baselines.
