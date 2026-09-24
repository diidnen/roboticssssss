"""Validate paired raw probe sensor evidence without inferring an OOD score."""
import argparse
import json
from pathlib import Path
from common import read,sha
from behavior_audit import probe_shift

REQUIRED = ('rgb__agentview_cam','rgb__eye_in_hand_cam','policy__eef_pose',
            'policy__gripper_pos','policy__gripper_marker_motion','policy__gripper_net_force')


def review(out):
    out=Path(out);report=read(out/'PROBE_OBSERVATION_SHIFT_REPORT.json')
    expected={p['id'] for p in read(out/'DEV_PLAN.json')['contexts']}
    rows=report['rows']
    if len(rows)!=12 or {r['context'] for r in rows}!=expected:
        raise RuntimeError('Incomplete sensor diagnostic coverage')
    result=[]
    for row in rows:
        job=Path(row['original_reference_job']);shift=read(job/'PROBE_OBSERVATION_SHIFT.json')
        if sha(job/'PROBE_OBSERVATION_SHIFT.json')!=row['artifact_sha256'] or shift!=row['shift']:
            raise RuntimeError('Probe sensor summary changed')
        process=read(job/'PROCESS_EXIT.json')
        if process['exit_code']!=0 or not process['completion']['logical_success']:
            raise RuntimeError('Probe diagnostic did not finish successfully')
        if not read(job/'DIAGNOSTIC_REFERENCE_PARITY.json')['passed']:
            raise RuntimeError('Probe diagnostic changed physical reference')
        for part in ('pre_probe_artifact','post_probe_artifact'):
            artifact=shift[part]
            if sha(artifact['path'])!=artifact['sha256']:
                raise RuntimeError('Raw sensor evidence changed')
        if not all(k in shift['metrics'] for k in REQUIRED):
            raise RuntimeError('Required raw image/tactile/state input missing')
        if not all(v.get('shape_parity') and v.get('all_values_finite') for v in shift['metrics'].values()):
            raise RuntimeError('Sensor input invalid or shape changed')
        result.append(dict(context=row['context'],job=str(job),raw_sensor_evidence_valid=True,
            same_physical_probe_reference=True,physical_shift=probe_shift(job),sensor_metrics=shift['metrics'],
            reused_reference_without_new_physics=row['reference_reused_without_new_physics'],
            summary_sha256=sha(job/'PROBE_OBSERVATION_SHIFT.json')))
    return dict(role='COMPLETE_PAIRED_PROBE_SENSOR_EVIDENCE_REVIEW',contexts=12,
        all_raw_image_tactile_state_valid=True,all_physical_probe_references_equal=True,
        rows=result,report_sha256=sha(out/'PROBE_OBSERVATION_SHIFT_REPORT.json'),
        implementation_sha256=sha(__file__),final_roots_used=False,
        inference='Measured input perturbations only. Postprobe policy acceptability must be judged together with complete online continuation/behavior/force qualification; no raw-MAE threshold substitutes for that evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--save',required=True);a=p.parse_args()
    d=review(a.out);Path(a.save).write_text(json.dumps(d,indent=2)+'\n')
    print('RAW_PROBE_SENSOR_EVIDENCE_VALID',d['contexts'])
