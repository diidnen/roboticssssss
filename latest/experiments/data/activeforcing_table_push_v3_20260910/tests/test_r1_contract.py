import importlib.util
from pathlib import Path
import numpy as np
P=Path('/media/volume/data/exouser/activeforcing_table_push_v3_20260910/code/r1_contract.py')
spec=importlib.util.spec_from_file_location('r1_contract',P); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_symmetric_grid_is_inside_final_bounds():
    g=m.safe_symmetric_grid(np.array([.2,-.3]), .01)
    assert np.all(np.abs(np.array([.2,-.3])+g[:,None]*np.array([1.,0.]))<1)
    assert np.allclose(g,-g[::-1])
def test_final_clip_is_detected():
    assert m.final_action_safe(np.array([.90,0]),np.array([.08,0]),.01)
    assert not m.final_action_safe(np.array([.99,0]),np.array([.01,0]),.01)
