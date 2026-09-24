import unittest
from matched_seeds import matched_native_prefixes


class SeedMatching(unittest.TestCase):
    def setUp(self):
        self.full=[dict(step=1+10*i,noise_seed=i,noise_sha256=str(i)) for i in range(35)]

    def test_valid_native_early_termination_preserves_pairing(self):
        self.assertTrue(matched_native_prefixes([self.full,self.full[:3]]))

    def test_changed_shared_seed_rejected(self):
        wrong=[dict(x) for x in self.full[:3]];wrong[-1]['noise_seed']=100
        self.assertFalse(matched_native_prefixes([self.full,wrong]))

    def test_skipped_native_query_rejected(self):
        self.assertFalse(matched_native_prefixes([self.full,self.full[1:3]]))

    def test_absent_inference_cannot_pass(self):
        self.assertFalse(matched_native_prefixes([self.full,[]]))


if __name__=='__main__':
    unittest.main()
