"""No experimental result: only binding invariants checked before v2 freeze."""
import ast
import inspect
import unittest
from canonical_formal_binding import FormalCanonicalQuery, install
import canonical_open_ready_query as candidate
import qualify_original_p4_native as query
import run_original_rootlocal_collection as collection
import run_original_rootlocal_inference as inference
import complete_original_rootlocal_pipeline as pipeline
from canonical_queue_bindings import bind


class BindingTests(unittest.TestCase):
    def test_original_ready_inherited(self):
        self.assertIs(FormalCanonicalQuery.ready,candidate.CanonicalOpenReadyQuery.ready)

    def test_original_step_save_inherited(self):
        self.assertIs(FormalCanonicalQuery.step,candidate.CanonicalOpenReadyQuery.step)
        self.assertIs(FormalCanonicalQuery.save,candidate.CanonicalOpenReadyQuery.save)

    def test_query_installed_globally(self):
        previous=query.NativeP4QueryEnv
        try:
            install();self.assertIs(query.NativeP4QueryEnv,FormalCanonicalQuery)
        finally:query.NativeP4QueryEnv=previous

    def test_exact_queue_bindings(self):
        for module,old,new,count in [(collection,'collect_original_rootlocal.py','collect_canonical_original_rootlocal.py',1),
                              (inference,'infer_original_rootlocal.py','infer_canonical_original_rootlocal.py',2),
                              (pipeline,'run_original_rootlocal_inference.py','run_canonical_original_inference.py',2)]:
            changed=bind(module,old,new,expected_count=count)
            self.assertIn(new,changed.__code__.co_consts)
            self.assertNotIn(old,changed.__code__.co_consts)

    def test_drift_rejected(self):
        with self.assertRaises(ValueError):bind(collection,'not_an_entrypoint.py','changed.py')


if __name__=='__main__':unittest.main()
