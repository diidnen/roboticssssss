"""Aggregate measured native gates; never infer qualification from a job exit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from audit_original_p4_capture import audit
from native_original_evidence_adapter import original_evidence


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def admit(query, *, require_material=False):
    query=Path(query)
    sys.path.insert(0,'/home/exouser/FORTE')
    from current_contract_physical_belief import verify_decision_prefix
    raw=read(query/'original_raw_rows.json');patch=read(query/'patch_readbacks.json')
    verify_decision_prefix(raw)
    if {r['task_id'] for r in raw}!={'dump_bin_bigbin'}:
        raise ValueError('Native task may not alias LIBERO task ID')
    if {int(r['seed_idx']) for r in raw}!={200002}:raise ValueError('Wrong root')
    x=original_evidence(raw,patch)
    np.testing.assert_array_equal(x,np.load(query/'original58_engineering.npy',allow_pickle=False))
    audit(query)
    report=read(query/'qualification.json');capture=read(query/'INDEPENDENT_CAPTURE_AUDIT.json')
    if not report.get('completed') or report.get('probe_failure') or not capture['all_checks_passed']:
        raise ValueError('Query/capture failed')
    if report.get('sensor_depth_binding_still_diagnostic') is not False:
        raise ValueError('Diagnostic depth is not formal training evidence')
    if report.get('squeeze_controller_binding')!='ORIGINAL_OUTER_AND_ORIGINAL_INNER_POSITION_FEEDBACK':
        raise ValueError('Full original squeeze controller required')
    if not capture['checks'].get('outer_feedback_uses_original_inner_filtered_measurement'):
        raise ValueError('No exact original filtered-feedback evidence')
    gates=Path(__file__).parent
    camera=read(gates/'collision_surface_camera_v3/qualification.json')
    replay=read(gates/'original_online_full_squeeze_reference_v3/online_qualification.json')
    if not camera.get('passed'):raise ValueError('Camera qualification failed')
    if not replay.get('duplicate_force_replays') or not all(r['full_trace_exact'] for r in replay['duplicate_force_replays']):
        raise ValueError('Full original-controller replay unqualified')
    if len(set(replay['first_chunk_hashes']))!=1:raise ValueError('Policy pairing unqualified')
    if any(r['kind']!='done' or r['result']['squeeze_controller_binding']!='ORIGINAL_OUTER_AND_ORIGINAL_INNER_POSITION_FEEDBACK'
           for r in replay['results']):raise ValueError('Wrong replay controller')
    material=query/'ACTUAL_OBJECT_MATERIAL.json'
    if require_material:
        values=read(material)
        if not values['shape_materials'] or not np.allclose(values['shape_materials'],report['friction'],atol=1e-6,rtol=0):
            raise ValueError('Actual applied material mismatch')
    result={'admitted':True,'root':200002,'task':'dump_bin_bigbin','features_shape':list(x.shape),
            'scope':'sensor/probe/pairing engineering admission; no learned-model or performance claim',
            'original_decision_prefix_passed':True,'original58_exact':True,
            'material_readback_required':require_material,
            'gate_hashes':{str(p):sha(p) for p in [query/'qualification.json',query/'INDEPENDENT_CAPTURE_AUDIT.json',
                gates/'collision_surface_camera_v3/qualification.json',
                gates/'original_online_full_squeeze_reference_v3/online_qualification.json']}}
    if require_material:result['gate_hashes'][str(material)]=sha(material)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('query',type=Path);ap.add_argument('--require-material',action='store_true')
    args=ap.parse_args();print(json.dumps(admit(args.query,require_material=args.require_material),indent=2))
