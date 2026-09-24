# Snapshot validation

- All 35,799 exported files were hashed and compared to their server sources before packaging.
- All 219 top-level FORTE Python files from the prior audit are present at their exact audited versions, including the 217 versions missing from the previously checked GitHub branches.
- Twelve source symlinks were materialized from targets already present and reviewed in the snapshot.
- The credential-pattern scan found no matches in the exported text.
- 24 focused CPU unit tests passed for decision-state validation, execution snapshot copying/restoration, friction probe budgeting, and command handoff guards.
- Python parsing and shell syntax checks covered 27,708 file paths (1,783 distinct source contents), before adding the twelve identical-content symlink copies.

## Existing syntax exceptions preserved

1. `Tabero/E3_HDF5_PREFLIGHT_CPU/P1_SIMPLIFIED_RUNTIME_LAUNCH_TEMPLATE.sh` contains the intentional placeholder `ROOT_ID=<PRE_REGISTERED_ROOT_ID>`. It requires filling in its template values before Bash can parse it.
2. `robotwin_taskforms/experiments/af_dump_original_restore_20260912/eigen_3_4_0/scripts/relicense.py` is a vendored Python 2-era helper and is not Python 3 syntax compatible.

These files were preserved without repairs. No GPU rollout, training job, full
simulator regression, checkpoint validation, or physical hardware test was run
for this upload. The snapshot guarantees source transfer integrity, not that
all historical experiment variants are runnable in one environment.
