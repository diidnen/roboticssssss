# E3 task5 DEV F4 fixed-anchor diagnostic

Status: **VALID_FROZEN_DEV_ANCHOR_CELL**

An independently scheduled single DEV episode completed for task5 at
friction 0.6 and commanded force 4 N on root seed 7500. The episode reached
pick but not lift, transport, placement, or official final success:

- pick success: 1
- lift success: 0
- official full-task success: 0
- exact requested force: 4 N
- root-state SHA-256: `a7e8dec7e85187d880d50d67e1040c74e7e825e2b28ed8d60ac55135db7f51a9`

The original 1--5 N TRAIN grid did not contain a FullTask positive, but a
subsequent lineage audit established that the `task5=5 N` Utility entry is for
`libero_object/task5:tomato_sauce_1_to_basket`, not this suite-local
`libero_10/task5:black_book_1_to_caddy` candidate. A predeclared TRAIN-only
support extension then stopped at its first FullTask success (7 N) and passed
the three-regime support gate without defining or changing Fmax. Before any
DEV outcome, Agent B froze DEV anchor forces 4/6/7 N. This F4 row is therefore
the first valid confirmatory anchor cell and supplies the LOCAL_FAILURE label
for root 7500. The existing Utility hash remains unchanged and no Utility
claim is made for the new long-horizon task.
