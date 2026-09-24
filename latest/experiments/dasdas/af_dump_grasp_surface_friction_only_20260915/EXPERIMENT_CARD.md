# Grasp-surface-only friction isolation (dump_bin)

New directory. Does not overwrite official 18/19, checkpoints, Fig.B, or v4 labels.

Method unchanged: scripted grasp 12 N → dump shear query → 8-step hold chunk → snapshot-fork remainder. Force grid = v4 `[0.25,5]@0.25`. Official terminal criterion unchanged.

Change: only outer grasp-surface triangles of `063_tabletrashbin` receive object-side μ. Inner walls/floor and bottom stay engine-default 0.3. Fingers/table/balls unchanged.

Seeds: 200014 sanity; 200003 geometric-wedging negative control. Frictions: official dump 0.425 / 0.575 / 0.85.

No belief/feasibility retrain. No 32×20 recapture in this directory.
