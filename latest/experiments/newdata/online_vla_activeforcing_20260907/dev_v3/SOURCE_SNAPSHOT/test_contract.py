import unittest
import numpy as np
from arbitration import Arbitration
from worker import make_sequence
from common import payload_sha

class ContractTests(unittest.TestCase):
    def test_force_injection_cannot_change_arm_or_squeeze(self):
        arb=Arbitration(4.25,.012)
        raw=np.array([.4,.2,.5,1.,2.,3.,-.5,1e5,-1e5,1e5,-1e5,1e5,-1e5],np.float32)
        a,r=arb.action(raw)
        np.testing.assert_array_equal(a[:6],raw[:6])
        self.assertEqual(a[6],np.float32(.012))
        np.testing.assert_array_equal(a[7:],[0,0,2.125,0,0,2.125])
        self.assertFalse(r['vla_gripper_override_after_af_handoff'])

    def test_vla_release_is_semantic_not_force_head_passthrough(self):
        arb=Arbitration(5,.013);raw=np.zeros(13,np.float32);raw[6]=.04;raw[9]=900
        a,r=arb.action(raw)
        self.assertTrue(r['vla_release_intent']);self.assertEqual(a[6],np.float32(.04))
        self.assertEqual(a[9],0);self.assertEqual(arb.command,.013)

    def test_real_soup_width_is_not_a_release_request(self):
        arb=Arbitration(4.7,.0262);raw=np.zeros(13,np.float32);raw[6]=.0258967
        action,record=arb.action(raw)
        self.assertFalse(record['vla_release_intent'])
        self.assertEqual(action[6],np.float32(.0262))
        self.assertEqual(action[9],np.float32(4.7/2))

    def test_nonfinite_policy_fails_loudly(self):
        a=np.zeros(13);a[2]=np.nan
        with self.assertRaises(ValueError):Arbitration(3,.01).action(a)

    def test_sequence_uses_real_online_chunk_not_scripted_prefix(self):
        raw=[dict(left_fx=0,left_fy=0,left_fz=2,right_fx=0,right_fy=0,right_fz=2,gripper_opening=.012,object_id='obj')]
        saved={'state':{'articulation':{'robot':{'joint_position':[0]*7+[.012,.012]}},'rigid_object':{'obj':{'root_velocity':[0]*6}}}}
        chunk=np.zeros((50,13));chunk[:8,:3]=np.arange(24).reshape(8,3)*.001
        x=make_sequence(raw,saved,None,None,5,chunk)
        np.testing.assert_allclose(x[:,:3],chunk[:8,:3]-chunk[0,:3],atol=1e-8)
        np.testing.assert_allclose(x[1:,3:6],np.diff(chunk[:8,:3],axis=0),atol=1e-8)
        np.testing.assert_array_equal(x[:,6:13],0)
        changed=chunk.copy();changed[3,1]+=.02
        self.assertFalse(np.array_equal(x,make_sequence(raw,saved,None,None,5,changed)))

    def test_observation_hash_covers_prompt_dtype_and_shape(self):
        a={'state':np.zeros(7,np.float32),'prompt':'task0'}
        self.assertNotEqual(payload_sha(a),payload_sha({**a,'prompt':'task1'}))
        self.assertNotEqual(payload_sha(a),payload_sha({**a,'state':np.zeros(7,np.float64)}))

if __name__=='__main__':unittest.main()
