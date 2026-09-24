"""Independent engineering checks on real original-P4 capture, no model fit."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from native_original_evidence_adapter import original_evidence
from original_p4_native_protocol import load_protocol


def audit(root):
    raw=json.loads((root/'original_raw_rows.json').read_text())
    readbacks=json.loads((root/'patch_readbacks.json').read_text())
    controls=json.loads((root/'native_controls.json').read_text())
    features=np.load(root/'original58_engineering.npy')
    reconstructed=np.asarray(original_evidence(raw,readbacks))
    checks={'feature_rebuild_exact':np.array_equal(features,reconstructed),
            'finite_58D':features.shape==(len(raw),58) and bool(np.isfinite(features).all()),
            'exact_row_count_alignment':len(raw)==len(readbacks)==len(controls),
            'contiguous_steps':all(r['step']==i for i,r in enumerate(raw,1)),
            'timestamps_match_actual_simulation':all(r['t_s']==c['elapsed_s'] for r,c in zip(raw,controls)),
            'clock_cumulative_error_under_half_physics_step':
                all(abs(c['elapsed_s']-i*.05)<=.00200001 for i,c in enumerate(controls,1)),
            'all_tactile_readouts_available':all(r['tactile_ok']==1 for r in raw)}
    altered=deepcopy(raw)
    for row in altered:
        row['friction']=913.7;row['seed_idx']=-100;row['trial_id']='leakage_sentinel'
        for key in row:
            if key.startswith('object_') and key.endswith('_priv'):row[key]=12345.67
    checks['privileged_labels_and_object_pose_do_not_enter58']=np.array_equal(features,original_evidence(altered,readbacks))
    differences=[];normal_differences=[];marker_differences=[]
    from scipy.spatial.transform import Rotation
    for row,control in zip(raw,controls):
        tag=f"query_{row['step']:04d}"
        capture=json.loads((root/(tag+'.json')).read_text())
        normal=np.stack([np.asarray(rot).T@np.asarray(finger['force_world_n'])
            for rot,finger in zip(control['force_frame_rotations_world'],capture['contacts']['fingers'])]).astype(np.float32)
        observed=np.array([[row[side+'_f'+axis] for axis in 'xyz'] for side in ('left','right')])
        normal_differences.append(float(abs(normal-observed).max()))
        with np.load(root/(tag+'_sensor.npz')) as data:
            source=readbacks[row['step']-1]['sensor_depth_provenance']
            mm=data['markers' if source.startswith('SAPIEN_RENDER') else 'collision_markers_diagnostic']
            displacement=mm[:,1]-mm[:,0]
            marker=float(np.linalg.norm(displacement,axis=-1).mean())
        marker_differences.append(abs(marker-row['marker_motion']))
        if row['step']>1:
            prev=raw[row['step']-2]
            expected=(row['marker_motion']-prev['marker_motion'])/control['interval_s']
            differences.append(abs(expected-row['marker_velocity']))
    checks['normal_vector_reprojection_exact']=max(normal_differences)==0
    checks['real_saved_markers_reproduce_features_exactly']=max(marker_differences)==0
    checks['marker_velocity_uses_true_interval']=max(differences,default=0)<1e-9
    inner_path=root/'original_squeeze_inner_trace.json'
    if inner_path.exists():
        inner=json.loads(inner_path.read_text())
        if inner:
            boundaries=np.cumsum([c['native_steps'] for c in controls])-1
            checks['outer_feedback_uses_original_inner_filtered_measurement']=all(
                r['measured_squeeze']==inner[int(i)]['measured_filtered_N'] for r,i in zip(raw,boundaries))
    report={'scope':'original P4 raw measurement and leakage audit; no task score',
            'checks':checks,'all_checks_passed':all(checks.values()),
            'rows':len(raw),'source_query_completed':True,
            'sensor_depth_contract':readbacks[0]['sensor_depth_provenance'],
            'formal_collection_gate_passed':False,
            'unresolved':['native depth acquisition admission','full online controller/replay qualification'],
            'max_normal_reprojection_error_n':max(normal_differences),
            'max_marker_feature_error_px':max(marker_differences),
            'max_marker_velocity_error':max(differences,default=0)}
    (root/'INDEPENDENT_CAPTURE_AUDIT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['all_checks_passed']:raise RuntimeError('Original P4 capture audit failed')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path)
    audit(ap.parse_args().root)
