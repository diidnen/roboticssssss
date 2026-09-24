# Direct utility → predicted trajectory → hard success search

The world model outputs only an H8 physical trajectory. The separately trained trajectory evaluator emits a hard SUCCESS/FAIL decision at its native logit-zero boundary. The controller does not consume probability. Starting at the Direct-utility proposal, only higher forces are searched.

Strict finite decisions: 19/24; finite SR=0.974, under-force=0.000, mean force=3.809 N, realized utility=1.006.

With explicitly labeled maximum-force fallback: SR=0.938, under-force=0.000, mean force=4.069 N, realized utility=0.571; world-model-validated rate remains 0.792.

This rule was specified after inspecting the earlier p>0.8 diagnostic and is therefore retrospective only.
