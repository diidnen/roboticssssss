"""Original-data exact parity and native force-bridge negative controls."""
from copy import deepcopy
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from native_original_evidence_adapter import measured_patch_record,normal_force_local,original_evidence,SOURCE

HERE=Path(__file__).parent
REF=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1/references/t0_r170052_mu5')

class OriginalAdapterTests(unittest.TestCase):
    def original_rows(self):
        with (REF/'RAW_PROBE.csv').open() as f:raw=list(csv.DictReader(f))
        return raw,json.loads((REF/'CONTACT_PATCH_READBACK.json').read_text())

    def test_unchanged_builder_exact_parity_on_original_observations(self):
        raw,patches=self.original_rows()
        spec=importlib.util.spec_from_file_location('_reference58_test',SOURCE)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        np.testing.assert_array_equal(original_evidence(raw,patches),module.ProbeEvidence().rows(raw,patches))

    def test_missing_tactile_or_numeric_data_cannot_default_to_zero(self):
        raw,patches=self.original_rows()
        raw[0].pop('marker_velocity')
        with self.assertRaisesRegex(ValueError,'Missing real observation'):
            original_evidence(raw,patches)
        raw,patches=self.original_rows()
        raw[0]['tactile_ok']='0'
        with self.assertRaisesRegex(ValueError,'tactile'):
            original_evidence(raw,patches)

    def test_true_friction_zero_negative_control_all_actual_task_samples(self):
        rows=json.loads(gzip.open(HERE/'force_zero_friction_v2/zero_friction.json.gz','rt').read())
        for i,row in enumerate(rows,1):
            velocities=[f['prestep_com_velocity_world'] for f in row['fingers']]
            record=measured_patch_record(row,velocities,.004,i)
            for friction in record['patches']['friction']:
                self.assertLess(max(abs(np.asarray(friction['sum_friction_vector_world']))),.001)

    def test_non_target_contact_rejected_not_mislabeled_as_target_friction(self):
        row=json.loads(gzip.open(HERE/'force_zero_friction_v2/original.json.gz','rt').read())[0]
        row=deepcopy(row)
        row['fingers'][0]['points'].append({'is_target':False,'impulse_ns':[.001,0,0]})
        with self.assertRaisesRegex(ValueError,'non-target'):
            measured_patch_record(row,[f['prestep_com_velocity_world'] for f in row['fingers']],.004,1)

    def test_legacy_normal_channels_never_use_full_friction_force(self):
        row=json.loads(gzip.open(HERE/'force_zero_friction_v2/original.json.gz','rt').read())[0]
        rotations=np.array([[[0,-1,0],[1,0,0],[0,0,1]],[[1,0,0],[0,0,-1],[0,1,0]]],float)
        expected=np.stack([r.T@f['force_world_n'] for r,f in zip(rotations,row['fingers'])]).astype(np.float32)
        np.testing.assert_array_equal(normal_force_local(row,rotations),expected)
        mutated=deepcopy(row)
        for finger in mutated['fingers']:
            finger['damping_corrected_contact_world_n']=[999,999,999]
        np.testing.assert_array_equal(normal_force_local(mutated,rotations),expected)

if __name__=='__main__':unittest.main()
