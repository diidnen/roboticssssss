# Dump lift-style remainder relabel v4

Same method as v3 / lift V5: scripted grasp (12 N cap) → dump shear query
(4 N / 0.012 m) → one 8-step EEF hold chunk → snapshot-fork scripted dump
remainder. Not π0. Not official AF 18/19.

Why new data:
- v3 VAL seed 200003 was all-success, so feasibility checkpoint NLL could not see failures.
- v3 TRAIN+TEST labels did not encode Coulomb (low μ often succeeded at 2 N; some high-μ fails were lost contact).
- Online belief collapse (posterior ~1.24) is a separate P4 z-score issue; this queue only relabels remainder.

Seeds actually used (probed stable): TRAIN 200002+200003, VAL 200010, TEST 200014.
200004/200006/200012 UnStableError. Force grid [0.25, 5] @ 0.25. VAL is mixed (not all-success).
Remainder still does not encode Coulomb (high μ often succeeds at lower F).
