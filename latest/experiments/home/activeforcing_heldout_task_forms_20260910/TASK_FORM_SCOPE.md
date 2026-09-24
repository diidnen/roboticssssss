# Compatible task scope

Full-task feasibility means terminal success of the complete downstream frozen-VLA behavior under the maintained grasp. The original lift, transport, placement and release stages are one instantiation, not a universal definition.

A compatible task must have (1) an established grasp before intervention, (2) the same grasp throughout the force-sensitive downstream portion, (3) a material dependence on grip-force sufficiency, (4) online frozen-VLA arm control, (5) a meaningful single grip-force setpoint, and (6) an objective, frozen task-specific terminal evaluator. Final release is permitted; repeated regrasp or manipulation of a second object is outside this experiment. An object can contact a surface while grasped, but ungrasped pushing is not grip-force adaptation.

There are three separate gates: a task definition exists; the current runtime can faithfully represent it; and Fixed-5 demonstrates VLA competence. Config files, generic pretrained-model capabilities, and demonstrations from a different checkpoint do not pass the last gate.

The current feasibility representation additionally requires a frozen unseen-task encoding before AF inference. Its four-way categorical code rejects new IDs. This does not prevent VLA-only qualification. No architecture or training labels may change during the primary test.

Keep existing native mass per task fixed across friction conditions; do not scale mass or claim mass transfer. New objects' different native masses remain an object-transfer confound and must be reported in category D. Use friction conditions within validated support, with no post-outcome tuning.

Environment definitions for opening drawers and inserting pegs exist. Drawer handles are articulated links, whereas the current AF probe/state contract targets rigid objects. Factory tasks use a different controller/observation configuration. These require demonstrated interface compatibility before inclusion. Plate pushing and multi-object/drawer-close composites are excluded from the complete-task benchmark; slicing them into a new invented task is not permitted.
