import unittest
from types import SimpleNamespace
import numpy as np
import torch
from activeforcing_execution_snapshot import attributes,apply_attributes,clone_value,max_difference


class SnapshotTests(unittest.TestCase):
    def test_deep_copy_no_live_alias(self):
        value={'x':torch.tensor([1.]),'nested':[np.array([2.])]}
        saved=clone_value(value);value['x'].add_(10);value['nested'][0][:]=30
        self.assertEqual(float(saved['x'][0]),1.)
        self.assertEqual(float(saved['nested'][0][0]),2.)

    def test_restore_preserves_live_tensor_alias(self):
        o=SimpleNamespace(filter=torch.tensor([2.]),latch=True,debug={'d':torch.tensor([3.])})
        alias=o.filter;saved=attributes(o)
        o.filter.zero_();o.latch=False;o.debug.clear()
        apply_attributes(o,saved,'cpu')
        self.assertIs(o.filter,alias);self.assertEqual(float(alias[0]),2.)
        self.assertTrue(o.latch);self.assertEqual(float(o.debug['d'][0]),3.)

    def test_references_not_serialized(self):
        reference=object();o=SimpleNamespace(environment=reference,filter=torch.tensor([4.]))
        self.assertNotIn('environment',attributes(o))

    def test_difference_detects_changed_layout(self):
        self.assertEqual(max_difference({'a':[torch.tensor([2.])]}, {'a':[torch.tensor([2.])]}),0.)
        self.assertEqual(max_difference({'a':1}, {'b':1}),float('inf'))
        self.assertEqual(max_difference(np.array([1.,3.]),np.array([2.,3.])),1.)


if __name__=='__main__':unittest.main()
