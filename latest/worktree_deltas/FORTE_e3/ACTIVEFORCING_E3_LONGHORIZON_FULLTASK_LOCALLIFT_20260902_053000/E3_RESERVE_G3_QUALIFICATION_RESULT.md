# E3 reserve `libero_goal/task3` nominal qualification result

## Verdict

`VALID_NOMINAL_QUALIFICATION_FAILURE_PI0_NO_CONTACT`

The exact frozen reserve cell completed naturally from the isolated launcher lineage. It is retained as scientific nominal-capability evidence although process ownership was `PROTECTED_PENDING_OWNERSHIP_EXTERNAL_OR_QUEUED` and no signal was sent.

- Output: `/media/volume/newdata/exouser/activeforcing_e3/RESERVE_LONGHORIZON_QUALIFICATION_RETRY_20260902_102800_g3/g3`.
- Suite/task: `libero_goal/task3`.
- TRAIN root: `7600`; friction: `0.6`; diagnostic force setpoint: `8 N`.
- Steps: `800`.
- Root-state hash: `85fd7c61d46a6fdf4701271cee5058b31cbc513fa8d35cc03c6e561459a1f797`.
- Final official goals: `[0, 0]`; query/contact/grasp/lift/open/full-task: all `0`.
- Mean and peak measured force: `0 N`, consistent with no contact.
- Episode SHA-256: `02872c6766b93772c2af15f35b3b1736e332aa90094a6120cd4c7a492eb2bf5e`.
- Steps SHA-256: `cc4adbf909846c5c5f8566abf8bf349b5bdd099cca307c9cb88711985b3db107`.
- Chunks SHA-256: `cee0d8ddbd6eeb40e932ad1db24626fed8db7ef188dc16bd6d8e24b4382167ff`.

This is a π0 nominal failure, not force-range or Utility evidence. No Fmax was inferred, no Utility configuration changed, and no TEST result was used. Together with valid nominal failures for reserve t2/t8 and task5's held-out DEV non-transfer, this activates the preregistered few-demo onboarding fallback.
