# E3 lineage repair amendment

Recorded: 2026-09-02 UTC.

This amendment preserves all prior gates and replay artifacts. It does not edit an expected hash and does not recompute or overwrite the existing normalization file.

## ID1 provenance

- The original ID1 retry gate remains historical and unchanged.
- Its HDF5 hash and all four media hashes match the live retry output.
- The live producer-QA file is `E3_TASK5_ONBOARDING_REPLAY_QA.json`, whose current SHA-256 is `2f4236637035cb4ac1fbd5a06f55f6cd74f6d21daf4f65c8f779e3c97092e418`.
- The old gate recorded producer-QA SHA-256 `1ff2851139caadf3bc46f29b3be4697613280ef0f9d132a7a64394965e73af79`.
- A new independent read-only refreeze is recorded in `TASK5_PI0_ONBOARDING_ID1_LINEAGE_REFREEZE_GATE_20260902.json` with `ID1_LINEAGE_REFREEZE_PASS`.
- The replay HDF5/media data was not modified; no partial evidence was promoted.

## Normalization path

The existing valid normalization artifact is:

`/media/volume/newdata/exouser/activeforcing_e3/TASK5_ONBOARDING_7DPF_ASSETS_20260902_113000/pi0_lora_tacfield_e3_task5_5demo_7dpf/activeforcing_e3_task5_5demo_7dpf/norm_stats.json`

SHA-256: `492a36faac4809acfe98098f67b56738f89a5600b997f1ea8d72d8bcded8ae32`.

The repaired execution chain now resolves this as `${ASSETS}/${CONFIG}/activeforcing_e3_task5_5demo_7dpf/norm_stats.json`. The old missing path was a `PATH_ALIAS`/missing-config-directory bug, not a wrong checkpoint or changed data version.

## Gate status

- ID1: `PASS_AFTER_READ_ONLY_REFREEZE`
- Existing five-demo dataset QA: remains PASS, with the historical ID1 hash drift explicitly documented.
- Training: not launched by this amendment.
- DEV: remains fail-closed until a frozen training checkpoint, candidate lock, and fresh coordinator gate exist.
