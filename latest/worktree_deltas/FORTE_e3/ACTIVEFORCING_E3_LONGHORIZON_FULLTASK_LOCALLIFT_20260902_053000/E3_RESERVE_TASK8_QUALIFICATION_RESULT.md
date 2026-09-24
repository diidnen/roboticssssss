# E3 reserve task8 nominal qualification

## Verdict

`RESERVE_TASK8_NOMINAL_QUALIFICATION_FAILED_PI0_NO_GRASP`

The externally/queued, pending-ownership task8 process matched the frozen reserve cell exactly and completed naturally. It produced one episode and 800 step rows. Frozen pi0 did not grasp/lift either monitored first moka pot, transport/place it, turn on the stove, or satisfy any of the three configured FullTask goals.

## Evidence

| Field | Value |
|---|---:|
| Suite/task | `libero_10/task8` |
| Root seed | 7600 |
| Root-state hash | `81fd20a39b0059dbe5276cc6726fd9c87447080f2ec6cc16bf4bc53a31cc5528` |
| Friction / diagnostic slot | 0.6 / 8 N |
| Steps / chunks | 800 / 80 |
| Grasp / lift / transport / placement | 0 / 0 / 0 / 0 |
| Stove turn-on | 0 |
| Goal status ever / final | `[0,0,0]` / `[0,0,0]` |
| Official FullTask | 0 |
| Maximum monitored-object XY displacement | 0.000118 m |
| Mean / peak measured force | 0 / 0 N |

The contact/proximity flag was positive on 18/800 rows but measured squeeze was zero on every row, so this is not an established-contact controller diagnosis. The primary failure is upstream frozen-pi0 nominal execution/no grasp.

## Provenance and ownership

- Output: `/media/volume/newdata/exouser/activeforcing_e3/RESERVE_LONGHORIZON_QUALIFICATION_RETRY_20260902_102500_t8/t8`.
- Launcher/Python: PID621733/621736, PPID466936; `PROTECTED_PENDING_OWNERSHIP_EXTERNAL_OR_QUEUED`, not an Agent B invocation.
- Episode CSV SHA-256: `0f6c64a992140dda66a573bf20f6379edda93a952108216a081a19ee13b2d63b`.
- Step CSV SHA-256: `fcdc8b9352cf466cab75b3d0c9668c7e1f0852f467dd2dc1f54fcd5603755211`.
- Raw chunks SHA-256: `7e77a62a9760d857a1c0d1b933251f0df1e0cdda4ef7726b49ba4270d37a62a5`.
- Summary JSON SHA-256: `0bdf34e2be365e174753ca7ef12b3a897c6626d274bc8383048359b8e9619f4d`.

No TEST data, pi0 update, Fmax inference, Utility evaluation, or selector change occurred. The result is valid for nominal screening because the command exactly matched the preregistered cell and all telemetry/hash checks passed; it was excluded from launch-ownership claims.
