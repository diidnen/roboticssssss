# Joint Decision Cost Contract

This contract is frozen before DEV comparison. It is an audit definition and is not added as a loss in the first ranking-only variant.

For a context boundary defined as the minimum observed successful force with a lower observed failure: exact boundary cost is `0`; selecting below boundary has cost `2`; selecting one approximately `0.25 N` step above has cost `0.25`; larger over-force has proportional mild cost `0.25 * ceil(over_force_N / 0.25)`. Under-force is therefore eight times the cost of a one-step over-force. The current force samples remain the existing nonuniform support; no force grid is changed.
