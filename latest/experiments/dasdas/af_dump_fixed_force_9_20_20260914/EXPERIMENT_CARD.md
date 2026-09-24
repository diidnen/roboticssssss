# Fixed 16/18/20 N upward fill on existing range contexts

Reuses the completed 5/8/10/12/15 N 120-rollout sweep on seeds `80200002–09`.
This run only adds commanded setpoints **above 15 N** on the **same 24 contexts**, so
per-context curves can be joined without recollecting the lower grid.

## Frozen design

- Task / root: `dump_bin_bigbin` / `200002`
- Population: same formal 24 contexts (`80200002–09` × `.425/.575/.85`)
- New paired branches: Fixed `16/18/20 N` (3 arms) from common P4 handoff
- Denominator: **24 × 3 = 72** rollouts
- Smoke: `16/20 N` on friction 0.85, seed 80200002 (excluded from analysis)
- No feasibility / utility

## Join rule

After completion, analyze joined success curves using prior labels at
`10/12/15` plus new `16/18/20` on matching `context_id`s.

## Claim boundary

Physical upward-range fill only. Commanded force ≠ realized contact load.
