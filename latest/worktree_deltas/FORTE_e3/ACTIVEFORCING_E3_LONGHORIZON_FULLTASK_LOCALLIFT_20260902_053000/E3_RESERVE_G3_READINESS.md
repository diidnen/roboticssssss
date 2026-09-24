# E3 reserve g3 launch readiness

Status: `STATIC_READY_RESOURCE_GATED_WAIT_EXPLICIT_GO`

Task2 and task8 both completed as valid frozen-pi0 nominal failures before grasp. The next candidate in the frozen reserve order is `libero_goal/task3`; this document prepares it offline and does not authorize launch.

- Exact authoritative instruction: `open the top drawer and put the bowl inside`.
- FullTask goals: `akita_black_bowl_1 in wooden_cabinet_1` AND wooden-cabinet top drawer open.
- LocalLift object: `akita_black_bowl_1`.
- TRAIN/root7600, friction0.6, one episode, max80 ten-step chunks, fixed diagnostic8N.
- Frozen pi0 checkpoint step49999; nominal motion/controller unchanged.
- New timestamped output root required.
- No correct local g3 HDF5/demo exists. Do not substitute `libero_goal/task1` or any short basket demo.

Launch exactly one foreground `g3` cell only after explicit root GO and immediate resource/duplicate audit. No Utility/Fmax/TEST/pi0-update claim is authorized.
