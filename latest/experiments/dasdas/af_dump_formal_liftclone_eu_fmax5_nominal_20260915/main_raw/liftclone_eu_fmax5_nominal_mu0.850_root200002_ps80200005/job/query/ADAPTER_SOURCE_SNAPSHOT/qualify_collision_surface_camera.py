"""Camera/raycast parity on actual physical surfaces; no physical edits."""
import json
import numpy as np
from scipy.ndimage import binary_erosion
import qualify_native_interfaces as base
from native_visual_mirror import observation_identity
from collision_surface_render_acquisition import CollisionSurfaceRenderAcquisition


def qualify(env,out):
    out.mkdir(parents=True,exist_ok=False)
    arm=env._af_grasp_arm_tag
    before=base.readback(env)
    observation_before=observation_identity(env.get_obs())
    acquisition=CollisionSurfaceRenderAcquisition(env,arm)
    report={'scope':'sensor camera surface parity, not an AF task trial',
            'formal_collection_gate_passed':False,'captures':[]}
    if base.readback(env)!=before:raise RuntimeError('Visual-only sensor proxy changed physical topology/state')
    def capture(tag):
        physical=base.readback(env)
        row=acquisition.capture(out,tag)
        with np.load(out/(tag+'_sensor.npz')) as data:
            reference=data['collision_axial_depth_m']
            rendered=data['depth_mm']/1000
            raster_centers=[]
            yy,xx=np.mgrid[:120,:160]
            for xyz in data['rendered_camera_positions']:
                valid=(xyz[...,3]<1)&(-xyz[...,2]>.024)&(-xyz[...,2]<.029)
                z=xyz[...,2]
                u=-xyz[...,0]/np.where(z!=0,z,1)*160+80
                v=xyz[...,1]/np.where(z!=0,z,1)*(120*20/18)+60
                raster_centers.append({'u_minus_pixel_index_median':float(np.median((u-xx)[valid])) if valid.any() else None,
                                       'v_minus_pixel_index_median':float(np.median((v-yy)[valid])) if valid.any() else None})
            errors=[];counts=[]
            for a,b in zip(reference,rendered):
                mask=binary_erosion((a>=.024)&(a<.0285),iterations=2)
                counts.append(int(mask.sum()))
                errors.append(float(np.max(abs(a[mask]-b[mask]))) if mask.any() else None)
        result={'tag':tag,'interior_contact_pixels':counts,'interior_camera_raycast_max_error_m':errors,
                'camera_marker_displacement_max_px':row['marker_displacement_max_px'],
                'ray_marker_displacement_max_px':row['collision_marker_displacement_max_px'],
                'measured_raster_pixel_center_convention':raster_centers,
                'capture_preserves_physics':base.readback(env)==physical}
        report['captures'].append(result)
    capture('grasp')
    servo,_=base.original_servo()
    scale=getattr(env.robot,arm+'_gripper_scale')
    joints=getattr(env.robot,arm+'_gripper')
    command=float(joints[0][0].drive_target[0])
    env.robot.set_gripper_force_limit(2.,arm)
    for step in range(500):
        if step%5==0:
            command=servo(command,base.contacts(env,arm)['measured_squeeze_n'],4.)
            env.robot.set_gripper((command-scale[0])/(scale[1]-scale[0]),arm,gripper_eps=0)
        env.scene.step()
    capture('hold4N')
    env.robot.set_gripper_force_limit(5.,arm)
    env.robot.set_gripper((.04-scale[0])/(scale[1]-scale[0]),arm,gripper_eps=0)
    for _ in range(250):env.scene.step()
    capture('open')
    # Visibility isolation at an unchanged current physical state, rather than
    # comparing policy images before and after physically opening the hand.
    observation_a=observation_identity(env.get_obs())
    capture('open_visibility_repeat')
    observation_b=observation_identity(env.get_obs())
    report['native_policy_observation_unchanged_by_sensor_capture']=observation_a==observation_b
    errors=[v for row in report['captures'] for v in row['interior_camera_raycast_max_error_m'] if v is not None]
    report['camera_raycast_interior_within_0_1mm']=bool(errors) and max(errors)<1e-4
    hold=report['captures'][1];opened=report['captures'][2]
    report['both_fingers_camera_contact_response']=all(v>0 for v in hold['camera_marker_displacement_max_px'])
    report['open_camera_markers_zero']=all(v==0 for v in opened['camera_marker_displacement_max_px'])
    report['passed']=all([report['native_policy_observation_unchanged_by_sensor_capture'],
        report['camera_raycast_interior_within_0_1mm'],report['both_fingers_camera_contact_response'],
        report['open_camera_markers_zero'],all(r['capture_preserves_physics'] for r in report['captures'])])
    (out/'qualification.json').write_text(json.dumps(report,indent=2))
    print('COLLISION_SURFACE_CAMERA_QUALIFICATION '+json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('Actual camera surface qualification failed')


if __name__=='__main__':
    base.qualify=qualify
    base.main()
