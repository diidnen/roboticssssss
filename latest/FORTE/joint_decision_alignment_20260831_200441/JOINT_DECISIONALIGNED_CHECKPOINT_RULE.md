# Joint-DecisionAligned Checkpoint Rule

The checkpoint rule is frozen before training: use the final epoch-80 checkpoint for every seed and fraction. No DEV metric, force-selection proxy, full-task SR, or post-hoc epoch choice is used for checkpoint selection. This preserves the Original Joint training budget and isolates the objective change. DEV ranking, calibration, selection, and SR are evaluation metrics only.
