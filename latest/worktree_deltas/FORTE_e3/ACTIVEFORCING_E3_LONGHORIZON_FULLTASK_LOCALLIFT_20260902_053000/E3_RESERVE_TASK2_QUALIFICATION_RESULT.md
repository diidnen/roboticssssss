# E3 reserve task2 nominal qualification

## Verdict

`RESERVE_TASK2_NOMINAL_QUALIFICATION_FAILED_PI0_NO_GRASP`

The clean retry completed naturally with one episode and 800 step rows. It is a valid TRAIN nominal-capability failure, not a resource abort. Frozen pi0 did not grasp or lift `moka_pot_1`, did not move it meaningfully, did not turn on the stove, and satisfied neither authoritative goal.

## Evidence

| Field | Value |
|---|---:|
| Suite/task | `libero_10/task2` |
| Root seed | 7600 |
| Root-state hash | `b9006260dcdf73bc08cff649a65b18be9b1d1860593e0dd4ad36c0a81ab55760` |
| Friction | 0.6 |
| Diagnostic force slot | 8 N |
| Steps / chunks | 800 / 80 |
| Grasp / lift / transport / placement | 0 / 0 / 0 / 0 |
| Stove turn-on | 0 |
| Goal status ever / final | `[0,0]` / `[0,0]` |
| Official FullTask | 0 |
| Maximum object XY displacement | 0.000171 m |
| Mean / peak measured force | 0 / 0 N |

The step table contains a positive contact/proximity flag on 54/800 rows but zero nonzero measured-squeeze rows. It is therefore not treated as an established-contact physical-controller diagnostic. The failure is upstream nominal execution/no grasp.

## Provenance

- Output: `/media/volume/newdata/exouser/activeforcing_e3/RESERVE_LONGHORIZON_QUALIFICATION_RETRY_20260902_101609/t2`.
- Episode CSV SHA-256: `d336f92dd933c16be205f8eec7ad4f49cda7c27923c7c5b83191e201c6f41b61`.
- Step CSV SHA-256: `e468f0783bea81b8c18af6cd6d691e70b2aa75d7b9f3c3ef9ddb923036acffe5`.
- Raw chunks SHA-256: `66789f4defaba18b0e52ab7a290febd192bf39c76189e369592559d569729abb`.
- Summary JSON SHA-256: `0b52f1545976fe054f22fd6ea45570af6679f5f647156416895a50024f20875f`.

No TEST data, pi0 update, Fmax inference, Utility evaluation, or selector change occurred. The 8 N slot was only the preregistered robust nominal diagnostic and is not a safety/normalization limit.

## Next protocol step

Per the frozen reserve order, task8 readiness is prepared offline. No task8 launch is authorized until a fresh resource audit and explicit root GO. Local few-demo onboarding remains unavailable because the prior filesystem audit found no correctly labeled task2 long-horizon demonstration; no unrelated HDF5 may be substituted.
