import unittest
import subprocess
import sys
import numpy as np
from activeforcing_command_handoff import command_from_probe


class HandoffTests(unittest.TestCase):
    def test_preserves_command_not_measured_joint(self):
        action=np.zeros((1,13));action[0,6]=.0028
        self.assertEqual(command_from_probe(action,.0028,0.,.04),.0028)

    def test_mismatched_controller_rejected(self):
        action=np.zeros((1,13));action[0,6]=.0028
        with self.assertRaises(ValueError):command_from_probe(action,.02225,0.,.04)

    def test_out_of_range_rejected(self):
        action=np.zeros((1,13));action[0,6]=.2
        with self.assertRaises(ValueError):command_from_probe(action,.2,0.,.04)

    def test_semantic_guard_matches_host_and_isaac_python(self):
        code="import sys;sys.path.insert(0,'/home/exouser/FORTE');import activeforcing_e2e_task0_smoke_20260905 as s;from activeforcing_command_handoff import adapt_branch_runner;m=s.load_module('guard_unit_test_p5',s.P5_PATH);a=adapt_branch_runner(m)[1];print(a['original_source_sha256'],a['adapted_semantic_ast_sha256'])"
        host=subprocess.check_output([sys.executable,'-c',code],text=True)
        isaac=subprocess.check_output(['/media/volume/newdata/exouser/tabero/env_isaaclab51/bin/python','-c',code],text=True)
        self.assertEqual(host,isaac)


if __name__=='__main__':unittest.main()
