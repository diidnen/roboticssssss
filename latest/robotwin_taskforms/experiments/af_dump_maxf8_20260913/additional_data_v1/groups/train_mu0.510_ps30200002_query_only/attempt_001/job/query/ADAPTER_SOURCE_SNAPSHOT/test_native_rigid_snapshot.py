"""Audit what SAPIEN pack/unpack and pose restore do to actual velocities."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sapien
from qualify_native_interfaces import Snapshot,readback

scene=sapien.Scene([sapien.physx.PhysxCpuSystem()])
builder=scene.create_actor_builder()
builder.add_box_collision(half_size=[.1,.1,.1])
actor=builder.build(name='snapshot_known_moving_box')
body=actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
actor.set_pose(sapien.Pose([0,0,2]))
body.set_linear_velocity([.13,-.27,.31])
body.set_angular_velocity([.7,.2,-.1])
env=SimpleNamespace(scene=scene,now_obs={},robot=SimpleNamespace())
snapshot=Snapshot(env)
expected=readback(env)
for _ in range(20):scene.step()
scene.physx_system.unpack(snapshot.physics)
scene.unpack_poses(snapshot.poses)
pack_only=readback(env)
snapshot.restore()
report={'scope':'engineering snapshot test only','pack_and_pose_restore_exact':pack_only==expected,
        'explicit_velocity_restore_exact':readback(env)==expected,
        'expected':expected,'pack_only':pack_only}
Path(__file__).with_name('RIGID_VELOCITY_SNAPSHOT_TEST.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert report['explicit_velocity_restore_exact']
