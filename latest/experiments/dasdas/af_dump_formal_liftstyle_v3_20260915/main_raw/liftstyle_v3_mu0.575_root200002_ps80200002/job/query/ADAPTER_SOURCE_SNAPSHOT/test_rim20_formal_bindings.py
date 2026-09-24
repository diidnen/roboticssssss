"""Binding-only tests; no new formal experiment or checkpoint is implied."""
import os
import unittest
from unittest.mock import patch
import canonical_open_ready_query as ready
import diagnose_canonical_rim_alignment as candidate
import rim20_formal_binding as binding
import qualify_original_p4_native as query
import run_original_rootlocal_collection as collection
import run_original_rootlocal_inference as inference
import complete_original_rootlocal_pipeline as pipeline
from canonical_queue_bindings import bind


class Bindings(unittest.TestCase):
    def test_original_methods_preserved(self):
        for name in ['ready','step','save']:
            self.assertIs(getattr(binding.FormalRim20Query,name),getattr(candidate.AlignedDiagnosticQuery,name))

    def test_super_class_cell_preserved(self):
        for name in ['__init__','canonicalize_open_ready','geometry_pose']:
            method=getattr(binding.BoundRimQuery,name)
            cells=dict(zip(method.__code__.co_freevars,method.__closure__ or []))
            self.assertIs(cells['__class__'].cell_contents,binding.BoundRimQuery)

    def test_geometry_calls_native_parent_with_only_fixed_offset(self):
        instance=object.__new__(binding.BoundRimQuery);instance.annotation_offset=.02
        with patch.object(ready.CanonicalOpenReadyQuery,'geometry_pose',return_value='native_pose') as parent:
            self.assertEqual(instance.geometry_pose(.12),'native_pose')
            self.assertAlmostEqual(parent.call_args.args[0],.14)

    def test_formal_context_missing_rejected_before_env_access(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'Exactly one formal context'):
                binding.FormalRim20Query(None,None)

    def test_install_fixed_geometry_and_restore_globals(self):
        previous=query.NativeP4QueryEnv
        try:
            with patch.dict(os.environ,{},clear=True):
                binding.install()
                self.assertIs(query.NativeP4QueryEnv,binding.FormalRim20Query)
                self.assertEqual(os.environ['AF_DIAGNOSTIC_RIM_OFFSET_M'],'.02')
        finally:query.NativeP4QueryEnv=previous

    def test_queue_fidelity(self):
        for module,old,new,count in [(collection,'collect_original_rootlocal.py','collect_rim20_original_rootlocal.py',1),
                (inference,'infer_original_rootlocal.py','infer_rim20_original_rootlocal.py',2),
                (pipeline,'run_original_rootlocal_inference.py','run_rim20_original_inference.py',2)]:
            method=bind(module,old,new,expected_count=count)
            self.assertIn(new,method.__code__.co_consts);self.assertNotIn(old,method.__code__.co_consts)


if __name__=='__main__':unittest.main()
