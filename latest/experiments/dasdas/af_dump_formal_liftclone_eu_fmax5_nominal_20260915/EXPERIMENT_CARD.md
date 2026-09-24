# Matched Nominal on the EU Fmax5 prefix

Official comparator for `af_dump_formal_liftclone_eu_fmax5_20260915`.
Not the old pre-grasp Nominal 10/24. Not the unused v3 Nominal card
(`12 N first`, no original P4 keep-hold).

Same physical prefix as that AF queue:
1. original dump P4 query
2. 12 N force cap; skip grasp_actor if P4/live still bilateral
3. dump 4 N / 0.012 m shear
4. handoff = last gripper command after that probe
5. π0 first chunk without moving
6. native π0 remainder (no AF force choice, no squeeze servo)

Remainder is Nominal Frozen VLA. Planned 24; prefix skips are not outcomes.
Official table pairs contexts that have both an AF record and a Nominal record.
