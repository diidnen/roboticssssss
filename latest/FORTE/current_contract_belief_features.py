"""Current probe evidence: legacy observable proxies + actual friction patches.

No label, band, root, simulator material, object privileged state, candidate
force or task outcome is read. Online and saved-data paths use the same builder.
World-frame friction vectors are explicitly named; no hidden frame conversion.
"""
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

ADAPTER=Path('/home/exouser/Tabero/analysis/activeforcing_historical_transfer_20260904/posterior/evidence_46d.py')
LEGACY_SCHEMA=ADAPTER.with_name('NORMALIZATION_46D.json')
SCHEMA_ID='CURRENT_PROBE_OBSERVABLES_PLUS_TRUE_FRICTION_58D_V1'
EXTRA_NAMES=[f'true_friction_{side}_world_{axis}_N' for side in ('left','right') for axis in 'xyz']+[
    'normal_patch_left_magnitude_N','normal_patch_right_magnitude_N',
    'actual_aperture_m','actual_left_joint_m','actual_right_joint_m',
    'incremental_directional_friction_stop_ratio']
PHASES=('approach','descend','close','hold','probe_out','probe_back','probe_hold')


class ProbeEvidence:
    def __init__(self):
        spec=importlib.util.spec_from_file_location('current_belief_legacy_observables',ADAPTER)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.legacy=mod.Evidence46D(LEGACY_SCHEMA)
        self.feature_names=['legacy_observable/'+x for x in self.legacy.feature_names]+EXTRA_NAMES
        assert len(self.feature_names)==58

    def rows(self, raw, readback):
        if not raw or len(raw)!=len(readback):raise ValueError('Unmatched raw/patch observation count')
        extra=[]
        for i,(r,b) in enumerate(zip(raw,readback),1):
            if int(r['step'])!=i or b['step']!=i:raise ValueError('Noncontiguous/stale observations')
            if r['probe_phase'] not in PHASES:raise ValueError('Candidate/task execution in belief input')
            if b['termination_signal']['phase']!=r['probe_phase']:raise ValueError('Phase epoch mismatch')
            if b['termination_signal']['contract_id']!='PHYSX_FRICTION_PATCH_INCREMENT_ALONG_PROBE_WORLD_V1':
                raise ValueError('Wrong probe signal contract')
            n=b['patches']['normal'];f=b['patches']['friction']
            if [x['pair'] for x in n]!=[[0,0],[1,0]] or [x['pair'] for x in f]!=[[0,0],[1,0]]:
                raise ValueError('Unexpected sensor/filter pair ordering')
            forces=[v for row in f for v in row['sum_friction_vector_world']]
            normals=[sum(float(v[0]) for v in row['normal_forces']) for row in n]
            joints=np.asarray(b['finger_joints'],float).reshape(-1)
            if joints.shape!=(2,) or not np.isclose(joints.sum(),b['aperture_m'],rtol=0,atol=1e-7):
                raise ValueError('Actual aperture/joint mismatch')
            extra.append(forces+normals+[b['aperture_m'],*joints,
                b['termination_signal']['incremental_directional_friction_ratio']])
        x=np.concatenate([self.legacy.rows(raw),np.asarray(extra,np.float32)],axis=1)
        if x.shape!=(len(raw),58) or not np.isfinite(x).all():raise ValueError('Invalid evidence tensor')
        return x.astype(np.float32)

    def load(self, directory):
        directory=Path(directory)
        with (directory/'RAW_PROBE.csv').open() as stream:raw=list(csv.DictReader(stream))
        readback=json.loads((directory/'CONTACT_PATCH_READBACK.json').read_text())
        return self.rows(raw,readback)

    def schema(self):
        return {'schema_id':SCHEMA_ID,'dim':len(self.feature_names),'feature_names':self.feature_names,
            'legacy_fields_semantics':'normal-contact-vector projections are observable proxies, NOT friction force',
            'new_force_source':'PhysX RigidContactView.get_friction_data / get_contact_data; physics dt',
            'sequence_time':'real observations from initial approach through post-probe return/hold only',
            'forbidden_inputs':['hidden_friction_analysis_only','friction_band','root_id','material_properties',
                                'object_position','object_velocity','candidate_force','full_task_success_y']}
