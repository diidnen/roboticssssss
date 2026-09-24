"""Offline USD/terminal-pose inspection; no simulation or label rewriting.

Run with the existing omni.usd.libs Python/library directories in the environment.
Outer AABB inclusion is only a necessary containment diagnostic, not a new label.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pxr import Usd,UsdGeom

ROOT=Path('/home/exouser/FORTE')
OUT=ROOT/'analysis/results/current_contract_restore_and_matched_pilot_20260905/history_reconstruction/placement_geometry_diagnostic'
ASSETS=Path('/home/exouser/Tabero/benchmarks/datasets/libero/USD')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rotation(q):
    w,x,y,z=np.asarray(q,float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def vertices(name,scale):
    path=ASSETS/name/(name+'.usd');stage=Usd.Stage.Open(str(path));cache=UsdGeom.XformCache()
    chunks=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
            transform=np.asarray(cache.GetLocalToWorldTransform(prim))
            chunks.append((np.column_stack([points,np.ones(len(points))])@transform)[:,:3])
    value=np.concatenate(chunks)*np.array(scale)
    return value,{'path':str(path),'sha256':sha(path),'vertex_count':len(value),
        'configured_spawn_scale':scale,'asset_stage_metersPerUnit':UsdGeom.GetStageMetersPerUnit(stage),
        'extra_unit_conversion_applied':False,'min':value.min(axis=0).tolist(),'max':value.max(axis=0).tolist()}


def main():
    if not (OUT/'PILOT_COMPLETE.json').exists():raise RuntimeError('Geometry diagnostic incomplete')
    basket,basket_meta=vertices('basket',[1.,1.,1.])
    obj,obj_meta=vertices('alphabet_soup',[.01,.01,.01])
    rows=[];details=[]
    for band,f in [('high',3.),('high',4.),('low',5.)]:
        job=OUT/'branches'/band/f'F{f:g}_R0';geo=read(job/'FINAL_GEOMETRY.json');gate=read(job/'FULL_BRANCH_REPLAY_GATE.json')
        if not gate['passed']:raise RuntimeError('Geometry replay mismatch')
        trace=read(job/'BRANCH_TRACE.json');r=read(job/'RESULT.json')
        b=np.array(geo['rigid_objects']['basket_1']['root_pose_w'][0]);o=np.array(geo['rigid_objects']['alphabet_soup_1']['root_pose_w'][0])
        snapshot=torch.load(job/'DECISION_STATE.pt',map_location='cpu',weights_only=False)
        initial_basket=np.asarray(snapshot['state']['rigid_object']['basket_1']['root_pose'][0,:3])
        shift=b[:3]-initial_basket
        rb,ro=rotation(b[3:7]),rotation(o[3:7])
        obj_world=obj@ro.T+o[:3]
        obj_in_basket=(obj_world-b[:3])@rb
        bmin,bmax=basket.min(axis=0),basket.max(axis=0)
        omin,omax=obj_in_basket.min(axis=0),obj_in_basket.max(axis=0)
        margins=np.minimum(omin-bmin,bmax-omax)
        center=(o[:3]-b[:3])@rb
        contact=float(trace[-1]['basket_contact_force_N'])
        row={'context':band.upper(),'candidate_F':f,'raw_contact_criterion_label':r['full_task_success_y'],
            'final_basket_contact_N':contact,'final_native_goal':trace[-1]['terminations']['success'],
            'object_center_basket_local_x':center[0],'object_center_basket_local_y':center[1],'object_center_basket_local_z':center[2],
            'minimum_xy_outer_bbox_margin_m':float(margins[:2].min()),
            'all_mesh_vertices_inside_outer_xy_bbox':bool(np.all(margins[:2]>=0)),
            'all_mesh_vertices_inside_outer_xyz_bbox':bool(np.all(margins>=0)),
            'basket_displacement_x_m':float(shift[0]),'basket_displacement_y_m':float(shift[1]),
            'basket_xy_displacement_m':float(np.linalg.norm(shift[:2])),
            'replay_gate_passed':True,'ground_truth_label_rewritten':False}
        rows.append(row);details.append({'case':row,'final_basket_pose_wxyz':b.tolist(),'final_object_pose_wxyz':o.tolist(),
            'object_vertices_in_basket_frame_min':omin.tolist(),'object_vertices_in_basket_frame_max':omax.tolist(),
            'goal_params':geo['task_goal_params'],'scene_state_sha256':sha(job/'FINAL_SCENE_STATE.pt'),
            'geometry_source_sha256':sha(job/'FINAL_GEOMETRY.json'),
            'images':[str(p) for p in sorted(job.glob('FINAL_*.png'))]})
    with (OUT/'TERMINAL_GEOMETRY_DIAGNOSTIC.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'scope':'geometry/contact-proxy diagnostic; not a new label builder','basket_asset':basket_meta,'object_asset':obj_meta,
        'contains_no_new_threshold_fitted_to_outcomes':True,
        'outer_bbox_inclusion_is_not_sufficient_for_true_containment':True,
        'formal_labels_admitted':False,'cases':details}
    (OUT/'GEOMETRY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
