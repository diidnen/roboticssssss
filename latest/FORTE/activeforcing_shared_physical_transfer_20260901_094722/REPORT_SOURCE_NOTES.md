# Report source and visualization notes

- Audience: technical.
- Primary question: how many target-task physical rollouts are required after learning from three source tasks, and whether explicit source-only system identification can replace the learned Probe estimator.
- Primary controller denominator: 36 root-heldout episodes per target task, method, budget, and seed; 144 episodes across four tasks.
- Probe denominator: 36 target-task-excluded Probe contexts per target task; 144 contexts total.
- Figure `FIG_NEW_TASK_FEWSHOT_SR`: ordered budget learning curve; chosen because the preregistered question is explicitly about B=0/10/20/30/60 and the non-monotonic shape is decision-relevant. Error bars are the standard deviation of the four-task macro metric across three Direct seeds.
- Figure `FIG_NEW_TASK_FEWSHOT_PER_TASK`: 2x2 small multiples expose task heterogeneity that the macro curve hides; axes are shared.
- Figure `FIG_PROBE_LOTO_COMPARISON`: grouped bars compare estimator MAE by held-out task/object; Learned Probe bars are three-seed means and Explicit SysID is deterministic.
- Visual QA: all three PNGs were inspected after rendering. The initial Probe figure exposed a string/integer task-key mismatch and was regenerated after fixing the reindex; final charts contain visible marks, labels, legends, and nonempty data.
- HTML delivery: canonical `artifact.json` validation and packaging passed. Verification is `structural_only` because no compatible local Chromium was installed; semantic chart tables remain embedded.
- Source limitations: task identity and object family change together; task1 has 140/180 reconstructed terminal labels; B=0 is not semantic zero-shot under the frozen task one-hot; task6 B10/B20 are one-class target adaptation sets.
