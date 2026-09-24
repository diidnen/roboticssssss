# B3 - DeliGrasp-Style Semantic-Prior Baseline

Status: `B3_BLOCKED_BY_DELIGRASP_API`.

This directory is an isolated B3 artifact directory for the frozen Tabero
positive benchmark set `[0, 1, 2, 5, 6]`.

No D2, B2/B2-R2, E2E, Tabero core, oracle, benchmark, probe, rule, or
learned network files were modified.

## What Was Completed

- Official DeliGrasp code was inspected.
- `DELIGRASP_CODE_MAP.md` maps prompt, force formula, slip loop,
  hardware parameters, API dependency, constants, and clamps to source
  files and line ranges.
- `DELIGRASP_PORT_SPEC.md` defines the Tabero port as
  `DELIGRASP_STYLE_TABERO_PORT`.
- `DELIGRASP_FORCE_MAPPING.md` defines the planned force mapping from
  DeliGrasp total force to Tabero squeeze target.
- Frozen B2-R2 reference tables were copied into B3 reference CSVs.

## Why The Run Is Blocked

Official DeliGrasp depends on an OpenAI chat completion call. The current
environment has no `OPENAI_API_KEY` or `CORRELL_API_KEY`, and the Python
environment does not have the `openai` package installed. Existing local
DeliGrasp cache files do not cover all five B3 objects using the official
prompt. Therefore no credible `estimated mass / friction / stiffness`
response was available.

Following the B3 instruction, no parameters were hand-invented and no
downstream Tabero evaluation was run.

## To Resume

Provide official-prompt raw responses for the five object identities, or
enable the official API path with fixed model settings. Then generate
`DELIGRASP_PREDICTIONS.csv`, run first-smoke and main evaluation, and
replace the blocked main-result placeholders.
