# v4 EU selector eval (frozen checkpoint)

Selector-only. Feasibility weights, belief, query, remainder, Fig.B, and official 18/19 untouched.

- Checkpoints: `.../af_dump_liftstyle_feas_v4_relabel_20260915/models/member_seed{0,1,2}.pt`
- `p_full` source: stored ensemble curves in `TRAIN_RESULTS.json` (no re-forward)
- EU: `U = p*(5-F)/5 - (1-p)`, `F* = first argmax U`
- Matched outcomes: existing snapshot-fork branches in `main_records/`

Verdict: **MIXED**. EU restores context-dependent F* (mean 2.48 N, 2/32 at 5 N). TRAIN/VAL keep success while cutting load. TEST collapses to 1/8 because overconfident low-F `p_hat` is wrong on seed 200014.
