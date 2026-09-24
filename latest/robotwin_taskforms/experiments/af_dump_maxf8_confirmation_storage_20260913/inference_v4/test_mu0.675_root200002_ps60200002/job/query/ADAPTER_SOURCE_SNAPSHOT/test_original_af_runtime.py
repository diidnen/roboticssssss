import json
from pathlib import Path
import unittest
from original_af_runtime import OriginalActiveForcing

class OriginalMethodGuardTests(unittest.TestCase):
    def test_rejects_old_dump_query_instead_of_silent_conversion(self):
        path = Path('/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments/af_dump_bin_bigbin_rootlocal_forcecal_20260912/branches.jsonl')
        row = json.loads(path.read_text().splitlines()[0])
        query = row['evidence']['query_info']['trace']
        with self.assertRaisesRegex(ValueError, 'CONTACT_PATCH_READBACK'):
            OriginalActiveForcing.validate_inputs(query, None, {})

    def test_rejects_fabricated_empty_contact_rows(self):
        with self.assertRaisesRegex(ValueError, 'Missing original sensor fields'):
            OriginalActiveForcing.validate_inputs([{'actor_position': [0, 0, 0]}], [{}], {})

    def test_rejects_missing_probe(self):
        with self.assertRaisesRegex(ValueError, 'RAW_PROBE'):
            OriginalActiveForcing.validate_inputs([], [], {})

    def test_original_ensembles_load_without_replacement(self):
        runtime = OriginalActiveForcing()
        self.assertEqual(len(runtime.belief.models), 3)
        self.assertTrue(all(hasattr(m, 'log_sigma_head') for m in runtime.belief.models))
        self.assertEqual(len(runtime.feasibility.models), 3)
        self.assertEqual(runtime.feasibility.models[0].command_gru.input_size, 10)
        self.assertEqual(runtime.feasibility.models[0].condition[0].in_features, 54)

if __name__ == '__main__':
    unittest.main()
