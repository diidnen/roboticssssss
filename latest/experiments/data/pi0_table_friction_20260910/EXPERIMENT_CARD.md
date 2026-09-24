# Frozen table-friction sweep — `pi0_libero`, LIBERO Goal task 5

Question: across fixed official task-5 initial states, does higher table/plate sliding resistance reduce native frozen-policy feasibility?

The only runtime edit is `model.geom_friction[table_collision][0]`; its other components remain `[0.005, 0.0001]`. Levels and roots are frozen at `[0.6, 1.2, 2.0] × [0,1,2,3,4]`. The primary endpoint is native success rate per level. All new outcomes use online websocket `pi0_libero`, native `OffScreenRenderEnv` success, 300-step horizon plus 10 stabilization steps, and 5-action replanning.

The init-0 0.6 and 2.0 pilot receipts are retained as reused evidence only after validation. The authoritative 2.0 receipt records `before=[0.6,0.005,0.0001]` and runtime readback `[2.0,0.005,0.0001]`; an older pilot `FROZEN_PROTOCOL.json` text summary incorrectly said `1.0` and is explicitly preserved only as a stale-provenance discrepancy. Server/RNG comparability is limited because the stock server exposes no seed API.

This is a five-root pilot and a one-property robustness intervention, not ActiveForcing or force-adaptation evidence.
