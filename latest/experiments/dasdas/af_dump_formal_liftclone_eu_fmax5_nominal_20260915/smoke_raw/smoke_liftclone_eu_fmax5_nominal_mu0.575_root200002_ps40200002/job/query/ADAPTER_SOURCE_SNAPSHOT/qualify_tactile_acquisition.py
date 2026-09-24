"""Real-scene sensor acquisition qualification; never emits AF training labels."""
import json
from pathlib import Path
import numpy as np
import sapien
from scipy.spatial.transform import Rotation
import qualify_native_interfaces as base
from original_tactile_core import OriginalTactileCore
import af_native_joint_readback as native


class Acquisition:
    def __init__(self, env, arm):
        self.env = env
        self.fingers = [j.child_link for j, _, _ in getattr(env.robot, arm + '_gripper')]
        self.cameras, self.frames = [], []
        original_frames = json.loads(Path(__file__).with_name('ORIGINAL_GEL_CAMERA_FRAMES.json').read_text())
        # Fixed URDF/mesh-derived diagnostic mounting, not fitted to outcomes.
        # Original Franka camera intrinsics; ARX mounting requires qualification.
        for i, finger in enumerate(self.fingers):
            sign = 1 if i == 0 else -1
            gel_rotation = np.array([[1, 0, 0], [0, 0, -sign], [0, sign, 0.]])
            center = np.array([.055, -sign * .024494417, .000103516])
            origin = center - .0285 * gel_rotation[:, 2]
            camera_rotation = np.column_stack((gel_rotation[:, 2], -gel_rotation[:, 0], -gel_rotation[:, 1]))
            quat = Rotation.from_matrix(camera_rotation).as_quat()[[3, 0, 1, 2]]
            camera = env.scene.add_mounted_camera('af_diagnostic_tactile_' + str(i), finger.entity,
                sapien.Pose(origin, quat), 160, 120, 2*np.arctan(18/40), .024, .029)
            camera.set_perspective_parameters(.024, .029, 160., 120*20/18, 80., 60., 0.)
            self.cameras.append(camera)
            # Preserve the actual original gelpad-to-camera transform (gelpad
            # origin is ~24 mm in front of camera, NOT at the camera origin).
            native_from_opengl = np.eye(4)
            native_from_opengl[:3,:3] = [[0,0,-1],[-1,0,0],[0,1,0]]
            opengl_from_gel = np.asarray(original_frames[i]['gel_in_opengl_camera']).T
            finger_from_gel = camera.local_pose.to_transformation_matrix() @ native_from_opengl @ opengl_from_gel
            gel_pose = sapien.Pose(finger_from_gel)
            self.frames.append(gel_pose)
        self.core = OriginalTactileCore()
        self.collision_core = OriginalTactileCore()

    def capture(self, output, tag):
        state_before = base.readback(self.env)
        visibility = []
        # Exclude the sensor's own opaque backing from the diagnostic camera,
        # then restore it before any policy rendering. No collision edits.
        for finger in self.fingers:
            for component in finger.entity.components:
                if isinstance(component, sapien.render.RenderBodyComponent):
                    visibility.append((component, component.visibility))
                    component.visibility = 0.
        depths, collisions, positions, quats, frame_info = [], [], [], [], []
        rendered_positions = []
        try:
            self.env.scene.update_render()
            target = self.env.deskbin.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            for finger, camera, local_gel in zip(self.fingers, self.cameras, self.frames):
                camera.take_picture()
                xyz = camera.get_picture('Position')
                rendered_positions.append(np.asarray(xyz).copy())
                if hasattr(self,'validate_pixel_centers'):
                    self.validate_pixel_centers(xyz)
                depth = -xyz[..., 2]
                valid = (xyz[..., 3] < 1) & (depth >= .024) & (depth <= .029)
                depth = np.where(valid, depth, .029)
                depths.append(depth * 1000)
                world = camera.global_pose.to_transformation_matrix()
                yy, xx = np.mgrid[0:120, 0:160]
                pixel_offset=getattr(self,'ray_pixel_offset',(.5,.5))
                rays = np.stack((np.ones_like(xx), -(xx+pixel_offset[0]-80)/160,
                                 -(yy+pixel_offset[1]-60)/(120*20/18)), -1)
                lengths = np.linalg.norm(rays, axis=-1)
                directions = (rays / lengths[..., None]) @ world[:3, :3].T
                origins = np.broadcast_to(world[:3, 3], directions.shape)
                distances = native.target_rays(target, origins.reshape(-1, 3), directions.reshape(-1, 3), .04)
                physical_depth = distances.reshape(120, 160) / lengths
                collisions.append(physical_depth)
                gel_world = finger.pose * local_gel
                rel = gel_world.inv() * self.env.deskbin.get_pose()
                positions.append(rel.p)
                quats.append(rel.q)
                frame_info.append({'finger_pose': base.vec(np.r_[finger.pose.p, finger.pose.q]),
                    'camera_pose': base.vec(np.r_[camera.global_pose.p, camera.global_pose.q]),
                    'gel_pose': base.vec(np.r_[gel_world.p, gel_world.q])})
        finally:
            for component, value in visibility:
                component.visibility = value
            self.env.scene.update_render()
        assert base.readback(self.env) == state_before, 'Sensor capture modified physics'
        depths = np.asarray(depths)
        markers = self.core.observe(depths, positions, quats, source='SAPIEN_RENDER_CAMERA_DEPTH')
        collisions = np.asarray(collisions)
        collision_depth_mm = np.where((collisions >= .024) & (collisions <= .029), collisions, .029) * 1000
        collision_markers = self.collision_core.observe(collision_depth_mm, positions, quats,
            source='PHYSX_COLLISION_DEPTH_DIAGNOSTIC_NOT_QUALIFIED')
        np.savez_compressed(output / (tag + '_sensor.npz'), depth_mm=depths,
                            rendered_camera_positions=rendered_positions,
                            collision_axial_depth_m=collisions, markers=markers,
                            collision_markers_diagnostic=collision_markers,
                            relative_positions_m=positions, relative_quaternions_wxyz=quats)
        row = {'tag': tag, 'depth_source': 'SAPIEN_RENDER_CAMERA_DEPTH',
            'mounting_qualified': False, 'physics_unchanged_by_capture': True,
            'depth_min_mm': depths.min(axis=(1, 2)).tolist(),
            'visible_contact_fraction': (depths < 28.5).mean(axis=(1, 2)).tolist(),
            'marker_displacement_max_px': np.linalg.norm(markers[:, 1]-markers[:, 0], axis=-1).max(axis=1).tolist(),
            'collision_marker_displacement_max_px': np.linalg.norm(collision_markers[:, 1]-collision_markers[:, 0], axis=-1).max(axis=1).tolist(),
            'frames': frame_info, 'contacts': base.contacts(self.env, self.env._af_grasp_arm_tag)}
        (output / (tag + '.json')).write_text(json.dumps(row, indent=2))
        print('TACTILE_CAPTURE ' + json.dumps({k:v for k,v in row.items() if k not in ('contacts','frames')}), flush=True)
        return row


def qualify(env, out):
    out.mkdir(parents=True, exist_ok=False)
    arm = env._af_grasp_arm_tag
    acquisition = Acquisition(env, arm)
    snap = base.Snapshot(env)
    servo, constants = base.original_servo()
    joints = getattr(env.robot, arm + '_gripper')
    scale = getattr(env.robot, arm + '_gripper_scale')
    command = float(joints[0][0].drive_target[0])
    report = {'scope': 'engineering sensor acquisition only', 'root': 200002,
              'model_training_started': False, 'task_success_evaluated': False,
              'formal_collection_gate_passed': False,
              'source_hashes': acquisition.core.source_hashes,
              'calibration_hashes': acquisition.core.calibration_hashes, 'captures': []}
    report['captures'].append(acquisition.capture(out, 'scripted_grasp'))
    env.robot.set_gripper_force_limit(2., arm)
    trace = []
    for step in range(500):
        if step % 5 == 0:
            command = servo(command, base.contacts(env, arm)['measured_squeeze_n'], 4.)
            env.robot.set_gripper((command-scale[0])/(scale[1]-scale[0]), arm, gripper_eps=0)
        env.scene.step()
        trace.append(base.contacts(env, arm))
        if step in (249, 499):
            report['captures'].append(acquisition.capture(out, 'preload_' + str(step+1)))
    (out / 'preload_trace.json').write_text(json.dumps(trace))
    # Deliberately open for a negative control; do not count this as a task trial.
    env.robot.set_gripper_force_limit(5., arm)
    env.robot.set_gripper((.04-scale[0])/(scale[1]-scale[0]), arm, gripper_eps=0)
    for _ in range(250):
        env.scene.step()
    report['captures'].append(acquisition.capture(out, 'open_negative_control'))
    snap.restore()
    report['restored_initial_exposed_physics'] = True
    (out / 'qualification.json').write_text(json.dumps(report, indent=2))
    print('TACTILE_ACQUISITION_DIAGNOSTIC_DONE', flush=True)


if __name__ == '__main__':
    base.qualify = qualify
    base.main()
