"""Offline-only ablation semantics and exact phase-free Local-Lift parity."""
import copy
import unittest
import numpy as np
import torch
from online_ablation_feasibility import AblationFeasibility, LOCAL, load_local_checkpoint
from phase_free_feasibility import KEEP_INPUT
from current_fulltask_feasibility_runtime import Network
from common import read


class AblationSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.models={name:AblationFeasibility(name,device='cpu') for name in
            ('POSTERIOR_MEAN','COARSE_GRID','PRIOR_NO_POSTERIOR','LOCAL_LIFT')}

    def test_mean_is_single_point_and_does_not_modify_evidence(self):
        original={'posterior_moments':{'mean':.4,'std':.2},'integration_nodes':[.2,.6],'integration_weights':[.5,.5]}
        before=copy.deepcopy(original)
        point=self.models['POSTERIOR_MEAN'].ablated_posterior(original)
        self.assertEqual(original,before)
        self.assertEqual(point['integration_nodes'],[.4]);self.assertEqual(point['integration_weights'],[1.])
        self.assertEqual(point['posterior_moments']['std'],0.)

    def test_prior_replaces_posterior_with_only_training_prior(self):
        model=self.models['PRIOR_NO_POSTERIOR']
        a=model.ablated_posterior({'integration_nodes':[.01],'integration_weights':[1.]})
        b=model.ablated_posterior({'integration_nodes':[100.],'integration_weights':[1.]})
        self.assertEqual(a,b);self.assertEqual(len(a['integration_nodes']),48)
        self.assertAlmostEqual(sum(a['integration_weights']),1.)

    def test_coarse_force_support_is_exactly_three_candidates(self):
        np.testing.assert_array_equal(self.models['COARSE_GRID'].force_grid,[3.,4.,5.])

    def test_local_lift_phase_removal_preserves_original_model_forward(self):
        raw64=np.zeros((3,8,64),np.float32)
        raw64[:,:,6]=1.;raw64[:,:,10]=np.array([3.,4.,5.])[:,None]/8.
        raw64[:,:,11]=.4
        # Nonzero causal chunk kinematics exercise the changed input layout.
        raw64[:,:,:3]=np.arange(8)[None,:,None]*np.array([.002,-.001,.005])[None,None,:]
        for i,item in enumerate(read(LOCAL)['checkpoints']):
            old=load_local_checkpoint(item['path']);legacy=Network().eval();legacy.load_state_dict(old['state_dict'])
            mean=np.asarray(old['normalization_mean'],np.float32);std=np.asarray(old['normalization_std'],np.float32)
            raw71=np.zeros((3,8,71),np.float32);raw71[:,:,KEEP_INPUT]=raw64;raw71[:,:,6:13]=mean[None,None,6:13]
            x=torch.as_tensor((raw71-mean)/std);new=self.models['LOCAL_LIFT']
            y=torch.as_tensor((raw64-new.mean)/new.std)
            with torch.no_grad():
                before=legacy(x[:,:,:17],x[:,0,17:]);after=new.models[i](y[:,:,:10],y[:,0,10:])
            torch.testing.assert_close(before,after,rtol=0,atol=2e-6)
            self.assertEqual(new.models[i].command_gru.input_size,10)


if __name__=='__main__':
    unittest.main()
