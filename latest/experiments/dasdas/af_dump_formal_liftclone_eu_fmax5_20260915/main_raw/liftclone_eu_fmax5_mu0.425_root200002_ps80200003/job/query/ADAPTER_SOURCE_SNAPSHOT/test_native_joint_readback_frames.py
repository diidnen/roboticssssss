"""Known-load sensor qualification; not task data and not a method change."""
import json
from pathlib import Path
import numpy as np
import sapien
import af_native_joint_readback

scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
scene.set_timestep(0.004)
builder = scene.create_articulation_builder()
root = builder.create_link_builder()
root.set_name('fixed_base')
root.add_box_collision(half_size=[.05, .05, .05])
tip = builder.create_link_builder(root)
tip.set_name('instrumented_finger')
tip.add_box_collision(half_size=[.01, .01, .01])
tip.set_joint_properties('prismatic', [[-.1, .1]], sapien.Pose([0, 0, 1]), sapien.Pose())
art = builder.build(fix_root_link=True)
link = art.get_links()[1]
rows = []
for rotation in ([1., 0., 0., 0.], [2**-.5, 0., 0., 2**-.5]):
    for force in ([0., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [.1, .4, .2]):
        art.set_root_pose(sapien.Pose(q=rotation))
        art.set_qpos([0.])
        art.set_qvel([0.])
        for _ in range(5):
            link.add_force_torque(force, [0, 0, 0])
            scene.step()
        state_before = scene.physx_system.pack()
        got = af_native_joint_readback.read_link(link)
        state_after = scene.physx_system.pack()
        prediction = np.asarray(got['unqualified_contact_balance_world_n'])
        rows.append({'root_rotation_wxyz': rotation, 'known_external_force_world_n': force, 'readback': got,
                     'max_error_n': float(np.max(abs(prediction - force))),
                     'readback_preserves_physics_state': state_before == state_after})
report = {'scope': 'known-load native sensor UNIT TEST; not dump or AF data', 'rows': rows,
          'passed': all(r['max_error_n'] < .001 and r['readback_preserves_physics_state'] for r in rows)}
Path(__file__).with_name('NATIVE_KNOWN_LOAD_TEST.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if not report['passed']:
    raise SystemExit(1)
