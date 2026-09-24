# Phase A existing ablation audit

Exact subset: roots 5100 and 6200; tasks 0/1/5/6; LOW/MID/HIGH; 24 contexts and 24 rows per variant. Full AF reference is present.

Local-Lift parity: YES. Same 647 valid rows, roots/splits, phase-free schema, architecture, normalization, optimizer/training budget, seeds and checkpoint selection; only target changes to lift_success_y. Physical rollouts are online VLA; supervision remains historical scripted auxiliary.
