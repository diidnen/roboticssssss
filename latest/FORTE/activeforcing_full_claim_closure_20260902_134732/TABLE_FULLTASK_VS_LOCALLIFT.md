# FullTask vs LocalLift matched supervision

Status: **FAILED_SCIENTIFICALLY**

- 720 branch labels audited: `720`
- LocalLift label definition: trace reaches post-lift phase (`transit`, `over_basket`, `place`, `release`, or `settle`).
- Label cross-tab: `[{'local_lift_success': 1, 'full_task_success': 0, 'n': 145}, {'local_lift_success': 1, 'full_task_success': 1, 'n': 575}]`
- LocalLift positive rate: `1.0000`

The label is degenerate in the authoritative 720 archive, so a matched LocalLift classifier has no negative training examples. Reporting a learned LocalLift-vs-FullTask comparison would be scientifically invalid. The previously observed downstream divergence in P6G0 is retained as a separate non-matched diagnostic.
