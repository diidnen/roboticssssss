"""Moving-chain and contact UNIT TESTS; never AF training/evaluation data."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import sapien
from scipy.spatial.transform import Rotation
import af_native_joint_readback as native
from native_contact_balance import contact_force


def quat(angles):
    return Rotation.from_euler('xyz', angles).as_quat()[[3,0,1,2]]


def make_scene(mode, offset):
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(.004)
    builder = scene.create_articulation_builder()
    root = builder.create_link_builder()
    root.set_name('fixed_root')
    root.add_box_collision(half_size=[.01]*3)
    parent = root
    if mode == 'chain':
        parent = builder.create_link_builder(root)
        parent.set_name('rotating_arm')
        parent.set_joint_properties('revolute', [[-3,3]], sapien.Pose([0,0,1],quat([0,.7,.2])), sapien.Pose())
        parent.set_mass_and_inertia(.2, sapien.Pose([.08,0,0]), [.001]*3)
    tip = builder.create_link_builder(parent)
    tip.set_name('instrumented_tip')
    material = scene.create_physical_material(0.,0.,0.)
    if mode == 'contact':
        tip.add_box_collision(half_size=[.01]*3, material=material)
    tip.set_mass_and_inertia(.02648,
        sapien.Pose([.004,-.003,.002], quat([.3,.5,.7])) if offset else sapien.Pose(),
        [1e-5,1.2e-5,1.5e-5])
    tip.set_joint_properties('prismatic', [[-.02,.02]],
        sapien.Pose([.14,0,0] if mode=='chain' else [0,0,1]), sapien.Pose())
    art = builder.build(fix_root_link=True)
    link = art.get_links()[-1]
    if mode == 'contact':
        wall = scene.create_actor_builder()
        wall.add_box_collision(half_size=[.01,.05,.05], material=material)
        wall.build_static('wall').set_pose(sapien.Pose([.021,0,1]))
    joints = art.get_active_joints()
    for joint in joints:
        joint.set_drive_properties(1000., 200., 2. if joint.type=='prismatic' else 100.)
    return scene, art, link, joints


def run(mode, offset):
    scene, art, link, joints = make_scene(mode, offset)
    rows = []
    for step in range(1000):
        t = step*.004
        if mode == 'chain':
            joints[0].set_drive_target(.3*np.sin(3*t))
            joints[-1].set_drive_target(.004*np.sin(5*t))
        else:
            joints[-1].set_drive_target(.005 + .0006*np.sin(8*t))
        external = np.array([.13,-.27,.31]) if step >= 500 else np.zeros(3)
        link.add_force_torque(external, [0,0,0])
        old_velocity = np.array(link.get_linear_velocity())
        scene.step()
        before = scene.physx_system.pack()
        reading = native.read_link(link)
        assert before == scene.physx_system.pack()
        normal = np.zeros(3)
        for c in scene.get_contacts():
            entities = [b.entity for b in c.bodies]
            if link.entity not in entities: continue
            sign = 1 if entities[0] == link.entity else -1
            for p in c.points: normal += sign*np.array(p.impulse)/.004
        expected = external+normal
        measured = np.array(reading['unqualified_contact_balance_world_n'])
        corrected = contact_force(reading,old_velocity,.004,applied_link_force_world=external)
        finite_acceleration = (np.array(reading['com_linear_velocity_world'])-old_velocity)/.004
        alternative = finite_acceleration*reading['mass_kg']-np.array(reading['gravity_world'])*reading['mass_kg']-np.array(reading['joint_force_world_n'])
        rows.append({'step': step, 'external': external.tolist(), 'normal_contact': normal.tolist(),
                     'expected': expected.tolist(), 'reading': reading,
                     'balance_error': (measured-expected).tolist(),
                     'corrected_contact_error': (corrected-normal).tolist(),
                     'finite_difference_balance_error': (alternative-expected).tolist(),
                     'finite_difference_acceleration': finite_acceleration.tolist()})
    chosen = [r for r in rows if 100 <= r['step'] < 490 or r['step'] >= 600]
    errors = np.array([r['balance_error'] for r in chosen])
    fd_errors = np.array([r['finite_difference_balance_error'] for r in chosen])
    corrected_errors = np.array([r['corrected_contact_error'] for r in chosen])
    report = {'scope': 'instrumentation unit test only', 'mode': mode, 'offset_inertia': offset,
              'cache_error_p95_n': np.percentile(abs(errors),95,axis=0).tolist(),
              'cache_error_max_n': abs(errors).max(axis=0).tolist(),
              'fd_error_p95_n': np.percentile(abs(fd_errors),95,axis=0).tolist(),
              'fd_error_max_n': abs(fd_errors).max(axis=0).tolist(),
              'corrected_contact_error_max_n':abs(corrected_errors).max(axis=0).tolist(),
              'passed': bool(abs(corrected_errors).max() < .001)}
    name = Path(__file__).with_name('FORCE_CACHE_DAMPING_CORRECTED_' + mode.upper() + ('_OFFSET' if offset else '_CENTERED'))
    with gzip.open(str(name)+'.json.gz','wt') as f: json.dump(rows,f)
    Path(str(name)+'_SUMMARY.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['chain','contact'],required=True)
    parser.add_argument('--offset',action='store_true')
    args = parser.parse_args()
    run(args.mode,args.offset)
