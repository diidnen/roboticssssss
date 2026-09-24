import tempfile
import unittest
from pathlib import Path

import torch

import mass_runtime_core as core


class FakeView:
    def __init__(self):
        self.mass = torch.tensor([[0.25]], dtype=torch.float32)
        self.inertia = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32)

    def get_masses(self):
        return self.mass.clone()

    def get_inertias(self):
        return self.inertia.clone()

    def set_masses(self, value, _ids):
        self.mass = value.clone()

    def set_inertias(self, value, _ids):
        self.inertia = value.clone()


class FakeP4:
    def __init__(self, view):
        self.calls = []
        self.view = view

    def _apply_friction(self, env, name, mu):
        self.calls.append((env, name, mu))
        return mu


class MassRuntimeCoreTests(unittest.TestCase):
    def test_mass_and_inertia_scale_together(self):
        view = FakeView()
        p4 = FakeP4(view)
        env = type("Env", (), {"scene": {"object": type("Object", (), {"root_physx_view": view})()}})()
        state = core.install_mass_intervention(p4, 0.1)
        self.assertEqual(p4._apply_friction(env, "object", 0.5), 0.5)
        self.assertAlmostEqual(float(view.mass.sum()), 0.1, places=7)
        torch.testing.assert_close(view.inertia, torch.tensor([[0.8, 1.2, 1.6]]))
        self.assertAlmostEqual(state["ratio"], 0.4)
        self.assertEqual(len(p4.calls), 1)

    def test_plan_requires_fixed_friction_and_positive_mass(self):
        plan = {"task": 0, "root": 1, "id": "x", "object": "o", "target": "t", "mu": 0.5, "mass_kg": 0.1}
        with tempfile.TemporaryDirectory() as folder:
            core.validate_call(plan, Path(folder), "0" * 64)
            bad = dict(plan, mu=0.6)
            with self.assertRaises(ValueError):
                core.validate_call(bad, Path(folder), "0" * 64)
            bad = dict(plan, mass_kg=0.0)
            with self.assertRaises(ValueError):
                core.validate_call(bad, Path(folder), "0" * 64)


if __name__ == "__main__":
    unittest.main()
