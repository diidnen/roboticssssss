# Open-loop high-force collection (prior-successful 12N import)

Import a **previously successful 12N** arm trajectory from
`af_dump_fixed_force_5_15_20260914_v3`, then replay identical arm actions
across commanded forces. No online π₀ recording in this protocol.

## Precise claim

**Force feasibility conditioned on a prospectively fixed prior-successful
12N trajectory** (imported; reference force fixed a priori).

Training / force-causal diagnostic labels only. **Not** online continuous AF
efficacy.

## Design

- Root `200002`
- Population: all v3 contexts with **12N success** (10 contexts)
- Replay forces: `10/12/15/16/18/20 N`
- Smoke: μ0.85 seed06 (historically 5/5), forces `12/16/20`, **12N must succeed**
- Serial only (`AF_OPENLOOP_PARALLEL=1`); replay does not call π₀

## Why not fresh π₀@12N record

Fresh open-loop recording at 12N produced failing trajectories (smoke + first
main). Locking a failing motion is invalid as a reference. Use proven-success
sources instead.
