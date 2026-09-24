# Task5 onboarding ffmpeg engineering fix audit

## Scope

The isolated file `/media/volume/newdata/exouser/Tabero_e3lh/scripts/tools/common/replay_utils.py` was changed only after all running replay processes exited. Physics, task/evaluator, action execution, recorder schema, success criteria, and HDF5 export logic are untouched.

## Change

- Previous file SHA-256: `a2f33afc06147c83b5e62a4e9337f001eb3d6e4bc28a781db619f6cddb996a8f`.
- Patched file SHA-256: `bd9a831c34219fb5ae5f318863784b0daba6809d228fd1448f23fc941c9a6097`.
- Added deterministic `TABERO_FFMPEG_BIN`; default remains `/usr/bin/ffmpeg` for compatibility and PATH is never searched implicitly.
- Missing/non-executable binary now raises immediately.
- ffmpeg nonzero exit now raises with the exit code and stderr.
- Because callers clean raw-frame directories only after successful return, either failure preserves all raw RGB/tactile frames.
- Frozen rerun binary: `/media/volume/newdata/exouser/softvtbench/miniforge3/bin/ffmpeg`, FFmpeg 8.1.2, SHA-256 `8ca0469917dae545e734473175974de1aa1e500a1f5fb7f0208513cb023bf495`.

## Validation

- `git diff --check`: PASS.
- AST parse: PASS.
- `py_compile` with external pycache: PASS.
- Missing-binary test: PASS; raised `FileNotFoundError` and retained the raw PNG.
- Explicit-binary test: PASS; created a nonempty H.264 MP4 and only then removed the two input frames.
- Worktree status: only `scripts/tools/common/replay_utils.py` modified, plus the previously created data symlinks are ignored/untracked by git status.

## Rerun gate

A fresh-output ID1 rerun is prepared with `TABERO_FFMPEG_BIN` set to the frozen binary above. It is not launched. A new host GPU/process audit and explicit root GO are required. The prior HDF5 is evidence only and will not be reused as final training data.
