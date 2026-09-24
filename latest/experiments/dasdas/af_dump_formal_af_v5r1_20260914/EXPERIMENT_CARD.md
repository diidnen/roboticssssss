# AF-v5r1 formal continuous selector (Nominal reused)

Decision: on the previous formal same-root population, does continuous AF-v5r1
(`0.5–15 N`, step `0.05 N`, realized-squeeze utility) improve over the already
completed Nominal Frozen VLA?

## Population (aligned to prior formal protocol)

- Physical root: `200002`
- Policy seeds: `80200002`–`80200009` (same as formal Nominal)
- Frictions: `.425 / .575 / .85`
- Contexts: 24
- Methods run now: **ActiveForcing-v5r1 only**
- Methods reused: **Nominal Frozen VLA** from `af_dump_formal_single_root_20260914` (10/24)
- Methods skipped: Fixed-Strong (per user request)

## Runtime contract

- Same P4 handoff / first VLA chunk / online replanning stack as formal
- Feasibility: frozen v5r1 ensemble, continuous EU on 291-point grid
- Belief: frozen v4 ensemble (unchanged)
- Smoke: AF-only on friction `.575`, seed `40200002`

## Claim boundary

Same-root path evaluation only. Seeds `80200002–05` overlap v5r1 high-force TRAIN
contexts, so this is a protocol-matched online check against reused Nominal, not a
fresh never-viewed confirmation set.
