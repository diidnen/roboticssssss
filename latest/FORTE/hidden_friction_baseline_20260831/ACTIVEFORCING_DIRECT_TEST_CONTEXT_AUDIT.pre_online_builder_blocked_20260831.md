# ActiveForcing-Direct TEST Context Audit

Status: **BLOCKED — ACTIVEFORCING_DIRECT_TEST_CONTEXT_CONTRACT_MISSING**

The frozen Direct implementation is executable on the DEV context population,
as demonstrated by Gate 4. Its exact loader calls
`preprobe_full_task_feasibility.all_contexts_for_split("DEV", active)`;
the loader's guarded manifest reader excludes every TEST row. The model path
therefore has no frozen context object or runtime context-builder call for
TEST roots 5174–5179.

The requested formal design needs 18 root×μ contexts (six roots and μ values
0.2, 0.5, 1.0). The available 5174–5179 files are incomplete/aborted context
collection outputs and cannot supply the exact Direct context for those cells.

The following resolutions are forbidden by the pre-TEST freeze:

- creating a new context representation or feature adapter;
- swapping to the prospective visual backend;
- substituting AFI or its `eta=0.9` selector;
- using a TEST outcome/frontier label as a Direct input.

Consequently, formal TEST simulator access is zero, frontier completion is
`0/810`, and no scientific comparison is claimed. The four DEV entry gates
remain runtime PASS and the old semantic ambiguity remains resolved by the
user freeze.
