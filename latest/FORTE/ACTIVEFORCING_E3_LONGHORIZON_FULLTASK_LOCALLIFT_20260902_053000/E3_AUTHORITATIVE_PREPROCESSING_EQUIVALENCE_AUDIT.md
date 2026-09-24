# E3 authoritative preprocessing equivalence audit

UTC: 2026-09-02T11:31:22Z

Status: **PASS BEFORE FINAL TRAINING**

The E3 benchmark-onboarding policy remains project-local OpenPI initialized
from `pi0_lora_tacfield_tabero` step 49999.  A CPU-only transform audit found
that the earlier optional adapter flattened real marker motion correctly but
retained the marker-free preflight's third-image mapping.  That mapping was not
identical to the frozen deployment adapter and therefore was not acceptable for
final training.

Commit `31049447d685cb36ddaeddda4f1d62fec0bc6392` changes only the optional E3
adapter: when `tactile_marker_motion` exists, the entire sample is delegated to
the original `TaberoTacFieldInputs`.  Consequently image values, image masks,
state, marker flattening, actions, and prompt are produced by the exact frozen
deployment transform.  The marker-free branch remains available only for the
already-labelled compatibility preflight and is forbidden by final dataset QA.

An independent synthetic CPU check compared every returned field and array
between `E3OptionalTaberoTacFieldInputs(ModelType.PI0)` and
`TaberoTacFieldInputs(ModelType.PI0)` with real marker motion and passed:

`E3_ADAPTER_AUTHORITATIVE_EQUIVALENCE_PASS`

Frozen post-audit hashes:

- `src/openpi/policies/libero_policy.py`:
  `f5eb0161b831c1f4a65b763f24183f5333a3efe54ae0537088a9450fdd54781a`
- `src/openpi/training/config.py`:
  `296b01a51c985583bab688402808296315bd449ddd2aa343f4f5458ba1d1295e`

No π0 server, checkpoint, Utility setting, controller, MASS file, or running
experiment was changed.
