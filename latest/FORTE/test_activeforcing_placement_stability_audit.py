import copy
import unittest
from activeforcing_placement_stability_audit import audit_stability
from activeforcing_placement_label_v2 import evaluate
from activeforcing_placement_contract import PHASE_COUNTS


class StabilityAuditTests(unittest.TestCase):
    def trace(self):return [dict(episode_step=i,object_pose_w=[0,0,.07,1,0,0,0],
        basket_pose_w=[0,0,0,1,0,0,0],object_velocity_w=[0]*6) for i in range(21)]
    def test_rest(self):
        result=audit_stability(self.trace())
        self.assertTrue(result['pose_motion_within_frozen_numeric_thresholds'])
        self.assertFalse(result['instantaneous_velocity_only_rejection'])
    def test_raw_velocity_disagreement_does_not_promote_label(self):
        trace=self.trace()
        for t in trace:t['object_velocity_w'][-1]=.3
        result=audit_stability(trace)
        self.assertTrue(result['instantaneous_velocity_only_rejection'])
        self.assertFalse(result['label_rewritten']);self.assertIsNone(result['new_success_label'])
    def test_actual_pose_motion_detected(self):
        trace=self.trace()
        for i,t in enumerate(trace):t['object_pose_w'][0]=.01*i
        self.assertFalse(audit_stability(trace)['pose_motion_within_frozen_numeric_thresholds'])
    def test_quaternion_sign_is_not_motion(self):
        trace=self.trace()
        for i,t in enumerate(trace):t['object_pose_w'][3]=(-1)**i
        self.assertEqual(audit_stability(trace)['max_pose_derived_angular_speed_rad_s'],0.)
    def test_missing_frame_rejected(self):
        trace=self.trace();trace[10]['episode_step']+=1
        with self.assertRaises(ValueError):audit_stability(trace)
    def test_inputs_unchanged(self):
        trace=self.trace();before=copy.deepcopy(trace);audit_stability(trace)
        self.assertEqual(trace,before)
    def test_v2_retains_geometry_release_and_motion_requirements(self):
        class Inside:
            def containment(self,*args):return {'inside':True}
        trace=self.trace()
        for t in trace:
            t['finger_object_contact_norms_N']=[0,0]
            t['object_velocity_w'][-1]=.3
        before=copy.deepcopy(trace)
        result=evaluate(trace,PHASE_COUNTS,Inside())
        self.assertEqual(result['place_success'],1)
        self.assertFalse(result['formal_training_admitted'])
        self.assertFalse(result['pre_registered_before_entire_pilot'])
        self.assertEqual(trace,before)
        trace[-1]['finger_object_contact_norms_N']=[1,0]
        self.assertEqual(evaluate(trace,PHASE_COUNTS,Inside())['place_success'],0)


if __name__=='__main__':unittest.main()
