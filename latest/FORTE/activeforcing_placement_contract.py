"""Isolated task0 placement repair; no controller, probe or planner changes.

The nominal path clears the actual basket and gripper collision geometry.
The evaluator is conservative geometric containment, not native contact-goal.
It is task0/asset specific and is NOT a general-purpose production evaluator.
"""
import ast
import hashlib
import inspect
import itertools
import json
import numpy as np

CONFIG = {
    'transport_clearance_m': .03,
    'release_clearance_m': .01,
    'minimum_lift_m': .142,
    'geometry_tolerance_m': .003,
    'top_margin_m': .003,
    'stable_window_steps': 20,
    'maximum_linear_speed_m_s': .02,
    'maximum_angular_speed_rad_s': .2,
    'maximum_relative_position_range_m': .005,
    'maximum_finger_object_contact_N': .05,
    'maximum_basket_tilt_deg': 15.,
}
PHASE_COUNTS = {'branch_hold':20, 'lift':50, 'transit':110,
                'over_basket':30, 'place':40, 'release':50, 'settle':50}


def rotation(q):
    q=np.asarray(q,float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:
        raise ValueError('Invalid wxyz quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def transform(points,pose):
    pose=np.asarray(pose,float)
    if pose.shape!=(7,) or not np.isfinite(pose).all():raise ValueError('Invalid pose')
    return np.asarray(points)@rotation(pose[3:]).T+pose[:3]


def corners(low,high):
    return np.array(list(itertools.product(*zip(low,high))),float)


class Geometry:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as f:self.data={k:f[k] for k in f.files}
        self.basket=self.data['basket_vertices']
        self.object=self.data['object_collision_surface']
        self.robot={str(k):self.data['robot_bounds'][i] for i,k in enumerate(self.data['robot_names'])}

    def top_surface(self,xy):
        """Conservative 1 mm mesh raster upper envelope; no-hit is not inside."""
        indices=np.floor((xy-self.data['height_origin'])/float(self.data['height_resolution'])).astype(int)
        field=self.data['height_field']; ix,iy=indices.T
        valid=(ix>=0)&(iy>=0)&(ix+1<field.shape[1])&(iy+1<field.shape[0])
        result=np.full(len(xy),np.nan)
        xx,yy=ix[valid],iy[valid]
        result[valid]=np.maximum.reduce([field[yy,xx],field[yy+1,xx],field[yy,xx+1],field[yy+1,xx+1]])
        return result

    def containment(self,object_pose,basket_pose):
        rb=rotation(np.asarray(basket_pose)[3:])
        points=(transform(self.object,object_pose)-np.asarray(basket_pose)[:3])@rb
        lo,hi=self.basket.min(0),self.basket.max(0)
        xy_margin=float(np.minimum(points[:,:2]-lo[:2],hi[:2]-points[:,:2]).min())
        top_margin=float(hi[2]-points[:,2].max())
        floor=self.top_surface(points[:,:2]); finite=bool(np.isfinite(floor).all())
        clearance=float((points[:,2]-floor).min()) if finite else None
        tilt=float(np.degrees(np.arccos(np.clip(rb[2,2],-1,1))))
        # A vertical free column above the actual woven basket surface, below
        # its rim, is sufficient but not necessary containment. Overhang/edge
        # cases can be conservatively rejected; no contact signal is a label.
        inside=bool(xy_margin>=0 and top_margin>=CONFIG['top_margin_m'] and finite
            and clearance>=-CONFIG['geometry_tolerance_m'] and tilt<=CONFIG['maximum_basket_tilt_deg'])
        return {'inside':inside,'outer_xy_margin_m':xy_margin,'below_top_margin_m':top_margin,
            'surface_clearance_m':clearance,'all_surface_columns_present':finite,'basket_tilt_deg':tilt,
            'object_center_basket_frame':((np.asarray(object_pose)[:3]-np.asarray(basket_pose)[:3])@rb).tolist()}


def evaluate_placement(frames,phase_counts,geometry):
    if len(frames)<CONFIG['stable_window_steps']:
        return {'place_success':0,'reason':'incomplete_stability_window','validated':False}
    tail=frames[-CONFIG['stable_window_steps']:]
    checks=[geometry.containment(f['object_pose_w'],f['basket_pose_w']) for f in tail]
    speeds=[np.linalg.norm(f['object_velocity_w'][:3]) for f in tail]
    angular=[np.linalg.norm(f['object_velocity_w'][3:]) for f in tail]
    positions=np.array([c['object_center_basket_frame'] for c in checks])
    movement=float(np.linalg.norm(np.ptp(positions,axis=0)))
    unheld=all(max(f['finger_object_contact_norms_N'])<=CONFIG['maximum_finger_object_contact_N'] for f in tail)
    settled=bool(max(speeds)<=CONFIG['maximum_linear_speed_m_s'] and max(angular)<=CONFIG['maximum_angular_speed_rad_s']
        and movement<=CONFIG['maximum_relative_position_range_m'])
    complete=phase_counts==PHASE_COUNTS
    inside=all(c['inside'] for c in checks)
    return {'place_success':int(complete and inside and unheld and settled),'validated':True,
        'all_phases_completed':complete,'all_final_window_inside':inside,'unheld':unheld,'settled':settled,
        'window_steps':len(tail),'max_linear_speed_m_s':float(max(speeds)),
        'max_angular_speed_rad_s':float(max(angular)),'relative_position_range_m':movement,
        'max_finger_object_contact_N':float(max(max(f['finger_object_contact_norms_N']) for f in tail)),
        'final_geometry':checks[-1],
        'definition':'all phases; final 20 steps collision-surface inside current-basket free-column volume below rim; unheld and stable',
        'scope':'conservative task0 asset-specific geometric evaluator; validate with final scene images'}


class PlacementPath:
    def __init__(self,env,p4,obj_name,basket_name,geometry):
        self.env=env;self.p4=p4;self.obj_name=obj_name;self.basket_name=basket_name
        self.geometry=geometry;self.trace=[];self.release_target=None

    def target(self,phase,cmd_pos):
        if phase=='branch_hold':return np.array(cmd_pos,copy=True)
        if phase in ('release','settle'):
            if self.release_target is None:raise RuntimeError('No pre-release target')
            return self.release_target.copy()
        env=self.env;p4=self.p4
        def cpu(t):return t.detach().cpu().numpy()
        eef=cpu(env.observation_manager.compute()['policy']['eef_pose'][0])
        robot=env.scene['robot']; root=cpu(robot.data.root_state_w[0,:7]);rr=rotation(root[3:])
        obj=cpu(env.scene[self.obj_name].data.root_state_w[0,:7])
        basket=cpu(env.scene[self.basket_name].data.root_state_w[0,:7])
        basket_world=transform(self.geometry.basket,basket)
        basket_base=(basket_world-root[:3])@rr
        basket_center=(basket[:3]-root[:3])@rr
        object_base=(transform(self.geometry.object,obj)-root[:3])@rr
        carried=[object_base]
        for name,bounds in self.geometry.robot.items():
            index=robot.body_names.index(name)
            pose=np.r_[cpu(robot.data.body_pos_w[0,index]),cpu(robot.data.body_quat_w[0,index])]
            carried.append((transform(bounds,pose)-root[:3])@rr)
        lowest=float(np.concatenate(carried)[:,2].min()); offset_low=lowest-float(eef[2])
        top=float(basket_base[:,2].max())
        clearance=CONFIG['release_clearance_m'] if phase=='place' else CONFIG['transport_clearance_m']
        height=top-offset_low+clearance
        target=np.array(cmd_pos,copy=True)
        obj_offset=(obj[:3]-root[:3])@rr-eef[:3]
        if phase=='lift':target[2]=max(height,float(cmd_pos[2])+CONFIG['minimum_lift_m'])
        else:
            # Refresh destination pose at each transport/placement boundary.
            # Align held object, not the EEF origin, with the basket center.
            if np.linalg.norm(obj_offset)>.12:
                raise RuntimeError('Object no longer near gripper; do not chase a fallen object')
            target[:2]=basket_center[:2]-obj_offset[:2]
            target[2]=height
        if phase=='place':self.release_target=target.copy()
        self.trace.append({'phase':phase,'target_base_m':target.tolist(),'eef_base_m':eef[:3].tolist(),
            'basket_pose_w':basket.tolist(),'basket_top_base_m':top,'lowest_carried_base_m':lowest,
            'carried_min_relative_eef_m':offset_low,'clearance_m':clearance,'object_offset_from_eef_m':obj_offset.tolist()})
        return target


def adapt_placement_runner(module):
    """AST edits only handoff and phase target construction; servo untouched."""
    original=inspect.getsource(module.downstream_branch)
    tree=ast.parse(original);f=tree.body[0]
    matches=[i for i,n in enumerate(f.body) if isinstance(n,ast.Try) and 'gripper_ids' in ast.unparse(n)]
    if len(matches)!=1:raise ValueError('Unrecognized handoff')
    f.body[matches[0]]=ast.parse('d_pred = float(handoff_cmd)').body[0]
    start=next(i for i,n in enumerate(f.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
        and [ast.unparse(e) for e in n.targets[0].elts]==['basket_b','_'])
    end=next(i for i,n in enumerate(f.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='phases')
    phases=[(k,v,None,'track' if k in ('branch_hold','lift') else 'open' if k in ('release','settle') else 'freeze') for k,v in PHASE_COUNTS.items()]
    f.body[start:end+1]=ast.parse('phases = '+repr(phases)).body
    loops=[n for n in ast.walk(f) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
        and [ast.unparse(e) for e in n.target.elts]==['phase','n_steps','target','mode']]
    if len(loops)!=1:raise ValueError('Unrecognized phase loop')
    loops[0].body.insert(0,ast.parse('target = placement_path.target(phase, cmd_pos)').body[0])
    f.name='downstream_branch_repaired_placement'
    for arg in ('handoff_cmd','placement_path'):
        f.args.kwonlyargs.append(ast.arg(arg=arg));f.args.kw_defaults.append(None)
    ast.fix_missing_locations(tree)
    def semantic(n):
        if isinstance(n,ast.AST):return {'node':type(n).__name__,**{k:semantic(v) for k,v in ast.iter_fields(n) if not(k=='type_params' and not v)}}
        if isinstance(n,list):return [semantic(v) for v in n]
        return n
    namespace=dict(vars(module));exec(compile(tree,'<placement_contract_adapter>','exec'),namespace)
    return namespace[f.name],{'original_source_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'adapted_semantic_ast_sha256':hashlib.sha256(json.dumps(semantic(tree),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'source':ast.unparse(tree),'changed':'checked command handoff; geometry-cleared refreshed phase targets',
        'unchanged':'phase durations, force inputs/modes, inner action construction, force servo, controller, probe, posterior, utility'}
