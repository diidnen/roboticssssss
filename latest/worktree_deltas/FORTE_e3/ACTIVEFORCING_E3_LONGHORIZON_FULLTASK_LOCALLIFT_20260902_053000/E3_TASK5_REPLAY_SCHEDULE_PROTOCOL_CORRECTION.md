# Task5 replay schedule protocol correction

## Decision

`AUTHORITATIVE_REMAINING_EXECUTION = ONE_BATCH_IDS_2_11_12_19_AFTER_ID1_GATE`

The authoritative pre-outcome source is `TASK5_PI0_ONBOARDING_REAL_REPLAY_READINESS.md`, SHA-256 `92911e9774d385acca7443f1107d4cd0d4663a796737d5a4f42ec1e7d289241a`, filesystem mtime `2026-09-02 10:45:21Z`. It froze a two-job design before the accepted ID1 result: one ID1 smoke, then one four-demo batch containing IDs `[2,11,12,19]` if and only if the smoke passed.

ID1 later passed. `TASK5_PI0_ONBOARDING_REMAINING4_BATCH_PLAN.json`, SHA-256 `dd47464c937703cd7c603fac35cd34ea243bd8cc580743a0d96491b85c1633c8`, implements that preregistered second job and does not use ID1 outcome to change demo membership, task semantics, Utility, Fmax, or success criteria.

## Conflicting post-smoke edit

At `2026-09-02 11:10:06Z`, after evidence commit `64a4168ecae2252a4e33a43a8cb40e0738aa8c56` (`2026-09-02 11:07:48Z`), the working copy of `E3_TASK5_5DEMO_ONBOARDING_PLAN.json` acquired one uncommitted field directing exactly one remaining demo after each E5 shard and explicitly prohibiting a batch.

- Post-smoke conflict SHA-256: `9f89c369d8506074c2f1a2f427bff27a69eac21a3f2776d7c412ab1208953b81`.
- Byte-exact preserved path: `E3_TASK5_5DEMO_ONBOARDING_PLAN.json.POST_SMOKE_SCHEDULE_CONFLICT_20260902_111006`.
- Committed pre-edit file SHA-256: `856b92873cbf3793886a2b75c84bb7c2167ad104920dade7f29cff02e974d772`.
- The uncommitted insertion is classified `POST_OUTCOME_SCHEDULING_CHANGE_NOT_AUTHORITATIVE`.

The canonical working file was restored byte-for-byte to the committed pre-edit version only after preserving the conflicting version above. Its legacy phrase `one timestamped output file per demo` is an output-identity requirement, not an execution-scheduling override; the earlier two-job readiness is authoritative for scheduling.

An untracked `validate_task5_onboarding_dataset.py` also appeared with mtime near the conflicting edit and SHA-256 `71ef6df1d538de1e6d5cf88eef1e37a1e0a48ea940358e1e0ed3369b66f7ec76`. It is preserved unchanged, is not a replay scheduler, and is not evidence that the post-smoke schedule was preregistered.

## Safety consequence

No remaining replay may start until a fresh resource/duplicate audit and explicit coordinator GO. When reopened, exactly one batch may replay TRAIN IDs `[2,11,12,19]`; ID1 remains excluded and immutable. Any per-demo staggered schedule would require an explicit new protocol decision and would be reported as a deviation rather than silently substituted.
