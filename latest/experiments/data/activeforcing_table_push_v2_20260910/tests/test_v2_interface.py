import sys,unittest,numpy as np
sys.path.insert(0,'/media/volume/data/exouser/activeforcing_table_push_v2_20260910/code')
from v2_interface import direction_from_fresh_chunk,ResidualRamp
class TestDirection(unittest.TestCase):
 def test_median_direction(self):
  x=np.zeros((20,7));x[:15,:2]=[1,.1];d=direction_from_fresh_chunk(x);self.assertEqual(d['selected_indices'],list(range(15)));self.assertGreater(d['direction_world_xy'][0],.99)
 def test_degenerate(self):
  with self.assertRaises(ValueError):direction_from_fresh_chunk(np.zeros((20,7)))
 def test_conflicting_median_degenerate(self):
  x=np.zeros((4,7));x[:,:2]=[[1,0],[-1,0],[1,0],[-1,0]]
  with self.assertRaises(ValueError):direction_from_fresh_chunk(x)
 def test_xy_cap_ramp_only(self):
  a=np.array([0,0,.3,.4,.5,.6,-1.]);r=ResidualRamp(.12);b,l=r.apply(a,[1,0],True);self.assertAlmostEqual(l['residual_scalar'],.02);np.testing.assert_array_equal(a[2:],b[2:])
if __name__=='__main__':unittest.main()
