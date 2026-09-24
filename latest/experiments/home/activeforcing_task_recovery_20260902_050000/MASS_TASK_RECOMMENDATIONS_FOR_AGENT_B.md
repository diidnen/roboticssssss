# Mass-task recommendations for Agent B

1. Qualify one base task first: frozen VLA reach, exact pre-probe state, force frontier, full-task labels, and post-lift checkpoint labels must all pass.
2. Run mass-only low/mid/high while holding object friction, geometry, appearance, initial state, controller, route, and query policy fixed. The actual mass values must be extracted from asset metadata and pre-registered; no absolute kg values are recovered in this lane.
3. Then run a 3×3 factorial: friction LOW/MID/HIGH × mass LOW/MID/HIGH. Record requested and realized force, measured force, acceleration/turning, contact state, slip/drop/placement events, and full success.
4. Treat mass as a real dynamical intervention: report inertia/load effects, not just a change in an object label. Articulated fixtures such as stove/microwave/cabinet may not support the same mass configuration, so use a rigid object/receptacle task for the first mass result.

Recommended order: current task 1 or 6 as a narrow-force stress control → `LIBERO-10 task 7`/LongTransportBasket for delayed transport → 3×3 joint physics. This is a recommendation for new collection, not an existing result.
