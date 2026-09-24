import sys, unittest, numpy as np
sys.path.insert(0,"/media/volume/data/exouser/activeforcing_table_push_20260910/code")
from push_force import PushForceController
class TestPushForce(unittest.TestCase):
 def test_disabled_exact_parity(self):
  a=np.array([.1,-.2,.3,.4,.5,.6,-1.]); b,log=PushForceController(25,enabled=False).apply(a,[1,0,0],3,True)
  np.testing.assert_array_equal(a,b); self.assertFalse(log['active'])
 def test_only_xy_mutate(self):
  a=np.array([0,0,.3,.4,.5,.6,-1.]); b,log=PushForceController(25).apply(a,[1,1,0],0,True)
  np.testing.assert_array_equal(a[2:],b[2:]); self.assertGreater(np.linalg.norm(b[:2]-a[:2]),0)
 def test_bound(self):
  a=np.zeros(7); b,log=PushForceController(1000,residual_limit=.02).apply(a,[1,0,0],0,True)
  self.assertLessEqual(np.linalg.norm(log['residual_xy']),.020000001); self.assertTrue(log['clipped'])
if __name__=='__main__': unittest.main()
