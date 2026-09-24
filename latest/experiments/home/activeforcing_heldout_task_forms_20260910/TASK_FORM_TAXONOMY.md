# Evidence-based task-form taxonomy

The inventory is not a count of tasks supported by the frozen VLA. `ALL_AVAILABLE_TASKS.csv` records availability tiers; `TASK_FORM_DIVERSITY_MATRIX.csv` keeps form novelty separate from object novelty. All unmeasured motion distances and rotations remain unknown.

| Form | Repository evidence | Training overlap | Present qualification status |
|---|---|---|---|
| A: lift/transport/place | All ten libero_object definitions; bowl-to-plate and bowl-to-stove variants | Yes: tasks 0,1,5,6 | Only original four have current matched online evidence |
| B: reorientation-heavy maintained grasp | Nut threading is a candidate; wine rack placement might rotate the object | No established instance | Rotation and retained grasp must be observed; do not count it yet |
| C: insertion/constrained placement | Native peg insertion and gear meshing; book-to-caddy and wine-to-rack candidates | Absent | Other native controller or unqualified task; position-only evaluator is insufficient evidence of insertion |
| D: extraction/pulling | Drawer opening; bowl-from-drawer candidate | Absent | Articulated-handle compatibility or post-grasp extraction loading must be established |
| E: surface-contact dragging while grasped | No supported maintained-grasp instance found; plate pushing exists | Absent | Plate pushing is outside the grip-force scope, not evidence for E |
| F: other maintained-grasp behavior | Stove knob operation is an unconfirmed candidate | Absent | Hand-control mode and grip dependence unresolved |

Multiple task definitions can instantiate one form. Scene/language variants are not independent forms. The six new libero_object identities are category A (`SAME_FORM_NEW_OBJECT`), not held-out task forms. Cream-cheese-to-bowl is conservatively category B (`NEW_TASK_SAME_FORM`): a tighter target alone does not establish a qualitatively different loading regime. Existing multi-object tasks that contain training objects require multiple grasps or other hand modes.

No clean category C (`HELDOUT_TASK_FORM` with a training-overlap manipulated object) is yet established. Book, wine bottle, bowl, drawer handle, peg and gear candidates also change the manipulated object, so any eligible experiment belongs to category D (`HELDOUT_TASK_FORM_AND_OBJECT`). Do not merge that result with isolated task-form transfer.

Native `libero_tactile.json` declares seven tasks but contains six entries with IDs 1,2,3,4,5,7. The current loader indexes the list by ID. Inventory preserves both declared ID and array index; silently using the wrong task would invalidate qualification.

The 785 additional archived BDDL definitions are explicitly marked as external definitions and provisionally classified from language. They require simulator, evaluator, controller, checkpoint and asset checks before use. Their count does not increase qualified task diversity.
