import unittest
from types import SimpleNamespace
import numpy as np
from activeforcing_probe_friction_contract import FrictionProbeBudget


class NumpyTensor:
    def __getitem__(self, key): return self
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.array([1.,0.,0.,0.])


class FrictionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.records=[]
        self.budget=FrictionProbeBudget(self.records,lambda q,v:v)
        self.env=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(root_quat_w=NumpyTensor()))})

    def call(self, phase='hold', force=(0.,0.,1.), legacy_rho=.9):
        step=len(self.records)+1
        self.records.append({'step':step,'patches':{
            'normal':[{'normal_forces':[[2.]]}]*2,
            'friction':[{'sum_friction_vector_world':(np.array(force)/2).tolist()}]*2}})
        return self.budget(env=self.env,phase=phase,tangent_base=[1.,0.,0.],step=step,legacy_rho=legacy_rho)

    def test_normal_projection_not_friction(self):
        for _ in range(10):self.call()
        self.assertEqual(self.call('probe_out'),0.)

    def test_project_increment_in_one_frame(self):
        for _ in range(10):self.call(force=(.4,.1,1.))
        self.assertAlmostEqual(self.call('probe_out',(.6,.1,1.)),.05)

    def test_transverse_baseline_not_probe_shear(self):
        for _ in range(10):self.call()
        self.assertEqual(self.call('probe_out',(0.,.5,1.5)),0.)

    def test_missing_hold_rejected(self):
        with self.assertRaises(RuntimeError):self.call('probe_out')

    def test_baseline_does_not_update_after_probe_start(self):
        for _ in range(10):self.call()
        self.call('probe_out',(.4,0.,1.))
        np.testing.assert_array_equal(self.budget.baseline,[0.,0.,1.])
        self.assertAlmostEqual(self.call('probe_out',(.8,0.,1.)),.2)

    def test_inputs_not_rewritten(self):
        for _ in range(10):self.call()
        self.call('probe_out',(.4,0.,1.))
        self.assertEqual(self.records[-1]['patches']['friction'][0]['sum_friction_vector_world'],[.2,0.,.5])


if __name__=='__main__':unittest.main()
