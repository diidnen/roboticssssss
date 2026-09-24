# Current Jetstream ActiveForcing source

This snapshot was copied from `grossly-excited-leech` on 2026-09-24. It includes
the current source, selected configuration/protocol files, and documentation
from the main FORTE and Tabero trees, the RoboTwin task-form project, related
experiment directories, and unmerged worktree differences.

The 217 FORTE Python file versions missing from the previously checked GitHub
branches are now included. All 219 top-level Python files from the earlier
FORTE audit were matched exactly to the snapshot.

## Where to start

| Directory | Purpose |
| --- | --- |
| `FORTE/` | Main FORTE working tree, ActiveForcing contracts, belief/inference/training scripts, and online VLA work |
| `Tabero/` | Current server Tabero source, force-position implementation, and available analysis scripts |
| `robotwin_taskforms/` | RoboTwin task environments, local pi0/XPolicyLab overlay, and task-form experiments |
| `experiments/dasdas/` | Later dump-bin, grasp, force, friction, and retention experiment code |
| `experiments/data/` | Table-push, knob damping, nominal comparisons, and additional online VLA experiment code |
| `experiments/newdata/` | Online VLA experiments and related source stored on the second data volume |
| `home_scripts/` | Top-level launch, recovery, and analysis scripts selected by their ActiveForcing references |
| `worktree_deltas/` | Separate unmerged branch variants; consult the source map before applying them |

`SOURCE_MANIFEST.json` records every exported path, original server path, file
mode, SHA-256, and Git blob hash. `SOURCE_ROOTS.md` gives a readable directory
map. Copies embedded in experiment directories are retained so historical runs
keep their original source; these copies are not all separate active models.

Twelve source-file symlinks were materialized with the exact contents of their
targets. Their original targets are recorded in the manifest.

## Running the source

This is a source snapshot, not a self-contained training environment. Existing
absolute server paths and dependency expectations are preserved. Read the
component README and `pyproject.toml`/requirements files before installing or
running a component. Model weights, simulator assets, datasets, rollout arrays,
and environments remain external inputs. Some vendored source originally
embedded in experiment directories is retained along with its license files.

The `Tabero/` snapshot preserves the specified Jetstream host's working tree.
It does not overwrite the separate `diidnen/Tabero` repository or automatically
merge its portability changes. The original `activeforcing/` transfer bundle
at the repository root remains available separately.

## Focused verification

From this directory, in an environment with NumPy and PyTorch:

```bash
cd FORTE
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
python3 -m unittest -v \
  test_activeforcing_decision_state \
  test_activeforcing_execution_snapshot \
  test_activeforcing_probe_friction_contract \
  test_activeforcing_command_handoff.HandoffTests.test_preserves_command_not_measured_joint \
  test_activeforcing_command_handoff.HandoffTests.test_mismatched_controller_rejected \
  test_activeforcing_command_handoff.HandoffTests.test_out_of_range_rejected
```

These 24 tests passed on the isolated export. See `VALIDATION.md` for the scope
and the original template/legacy syntax exceptions.
