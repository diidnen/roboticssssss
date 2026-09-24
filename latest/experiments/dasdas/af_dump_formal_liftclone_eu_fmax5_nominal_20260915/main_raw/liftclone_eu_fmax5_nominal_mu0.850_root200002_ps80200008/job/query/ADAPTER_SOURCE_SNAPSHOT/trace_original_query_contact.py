"""Passive high-rate diagnostic around the unchanged original native query."""
import gzip
import hashlib
import json
import os
import numpy as np
import qualify_native_interfaces as base
import qualify_original_p4_native as query
from native_p4_query_env import NativeP4QueryEnv
from rootlocal_collection_contract import write


class TracedQuery(NativeP4QueryEnv):
    def __init__(self, env, out):
        super().__init__(env, out)
        self.diagnostic_stream = gzip.open(out/'DIAGNOSTIC_PHYSICS_TRACE.jsonl.gz', 'wt', encoding='utf-8')
        self.diagnostic_hash = hashlib.sha256()
        self.diagnostic_steps = 0

    def step(self, action):
        native_step = self.native.scene.step

        def traced_step():
            native_step()
            before_q = np.asarray(self.kin.art.get_qpos()).copy()
            before_v = np.asarray(self.kin.art.get_qvel()).copy()
            contact = base.contacts(self.native, self.arm)
            np.testing.assert_array_equal(self.kin.art.get_qpos(), before_q)
            np.testing.assert_array_equal(self.kin.art.get_qvel(), before_v)
            self.diagnostic_steps += 1
            row = {'physics_step': self.diagnostic_steps, 'query_step': self.step_count+1,
                'physics_dt_s': float(self.native.physics_timestep),
                'inner_before_step': dict(self.inner.last),
                'original_action13': action.detach().cpu().numpy().reshape(-1).tolist(),
                'qpos': before_q.tolist(), 'qvel': before_v.tolist(),
                'arm_indices': self.kin.arm_indices,
                'arm_drive_targets': [float(j.drive_target[0]) for j in getattr(self.native.robot,self.arm+'_arm_joints')],
                'finger_drive_targets': [float(j.drive_target[0]) for j,_,_ in self.joints],
                'finger_drive_properties': [{'stiffness': float(j.stiffness), 'damping': float(j.damping),
                                            'force_limit': float(j.force_limit)} for j,_,_ in self.joints],
                'object_pose': np.r_[self.native.deskbin.get_pose().p, self.native.deskbin.get_pose().q].tolist(),
                'contact': contact, 'readout_preserved_qpos_qvel_exactly': True}
            line = json.dumps(row, separators=(',', ':'))
            self.diagnostic_hash.update(line.encode()); self.diagnostic_stream.write(line+'\n')

        self.native.scene.step = traced_step
        try: return super().step(action)
        finally: self.native.scene.step = native_step

    def save(self):
        try: super().save()
        finally:
            self.diagnostic_stream.close()
            write(self.out/'DIAGNOSTIC_TRACE_RECEIPT.json', {'physics_steps': self.diagnostic_steps,
                'trace_rows_sha256': self.diagnostic_hash.hexdigest(),
                'formal_data_admission': False, 'changed_controller_or_query_parameters': False})


if __name__ == '__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP'] = '1'
    query.NativeP4QueryEnv = TracedQuery
    base.qualify = query.qualify
    base.main()
