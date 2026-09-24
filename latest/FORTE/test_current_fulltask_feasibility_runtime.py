import csv,json,unittest
from pathlib import Path
import numpy as np
from current_fulltask_feasibility_runtime import CurrentFeasibility

TRAIN=Path('/home/exouser/FORTE/analysis/results/current_fulltask_feasibility_baseline_v1_20260906')
STAGE=Path('/home/exouser/FORTE/analysis/results/current_matched_resumption_v1_20260906')

class TestRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime=CurrentFeasibility(device='cpu')
        with (TRAIN/'HELDOUT_PLANNER.csv').open() as f:cls.expected=list(csv.DictReader(f))
        cls.rows={json.loads(p.read_text())['context_id']:json.loads(p.read_text()) for p in (STAGE/'rows').glob('*.json')}
    def test_all_heldout_selected_force_parity(self):
        for expected in self.expected:
            row=self.rows[expected['context_id']];reference=Path(row['reference'])
            got=self.runtime.select(np.load(reference/'PREACTION_SEQUENCE.npy'),json.loads((reference/'PREACTION_POSTERIOR.json').read_text()))
            self.assertEqual(float(expected['selected_force']),got['selected_force_N'])
            self.assertAlmostEqual(float(expected['p3']),got['p_success'][0],places=5)
            self.assertAlmostEqual(float(expected['p5']),got['p_success'][-1],places=5)
    def test_rejects_postaction_posterior(self):
        expected=self.expected[0];row=self.rows[expected['context_id']];reference=Path(row['reference'])
        posterior=json.loads((reference/'PREACTION_POSTERIOR.json').read_text());posterior['candidate_actions_executed']=1
        with self.assertRaises(ValueError):self.runtime.select(np.load(reference/'PREACTION_SEQUENCE.npy'),posterior)
if __name__=='__main__':unittest.main()
