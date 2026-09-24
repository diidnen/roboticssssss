"""Load the unchanged original Taxim/FOTS kernels without Isaac acquisition.

This module requires actual depth and relative-pose sensor inputs. It does not
invent a depth image from force, privileged friction, success or candidate force.
The sensor mounting and depth acquisition are separately qualification-gated.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import torch
import torchvision.transforms.functional as F

ROOT = Path('/home/exouser/Tabero/source/tac_manip/tac_manip/core/sensors/tacex/simulation_approaches')
CALIB = Path('/home/exouser/Tabero/source/tac_manip/tac_manip/assets/data/Sensors/GelSight_Mini/calibs/640x480')
DEPENDENCIES = Path(__file__).parent / 'sensor_dependencies'
ORIGINAL_MATH = Path('/media/volume/newdata/exouser/tabero/IsaacLab/source/isaaclab/isaaclab/utils/math.py')

def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def load_original_euler_conversion():
    # The actual installed IsaacLab defaults to signed Euler angles, not
    # wrapping to [0,2pi). Execute its exact function bodies, without importing
    # the Isaac application. Removing only TorchScript decorators avoids its
    # source-inspection dependency; equations/defaults remain unchanged.
    names = {'copysign', 'euler_xyz_from_quat'}
    nodes = [node for node in ast.parse(ORIGINAL_MATH.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError('Original Euler conversion source unavailable')
    for node in nodes: node.decorator_list = []
    namespace = {'torch':torch}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ORIGINAL_MATH),'exec'),namespace)
    return namespace['euler_xyz_from_quat']

euler_xyz_from_quat = load_original_euler_conversion()

class OriginalTactileCore:
    def __init__(self, fingers=2, device='cuda'):
        sys.path.insert(0, str(DEPENDENCIES))
        package_name = '_af_original_taxim_sim'
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            package.__path__ = [str(ROOT / 'gpu_taxim/sim')]
            sys.modules[package_name] = package
        mod = load_file(package_name + '.taxim_torch', ROOT / 'gpu_taxim/sim/taxim_torch.py')
        marker_mod = load_file('_af_original_fots_marker', ROOT / 'fots/sim/marker_motion.py')
        self._taxim = mod.TaximTorch(calib_folder=CALIB, device=device)
        self._device = device
        self.marker_motion_sim = marker_mod.MarkerMotion(
            frame0_blur=self._taxim.background_img.movedim(0, 2).cpu().numpy(),
            mm2pix=19.58, num_markers_col=11, num_markers_row=9,
            tactile_img_width=320, tactile_img_height=240,
            lamb=[.00125, .00021, .00038], x0=15, y0=26)
        self.init_marker_pos = np.stack((self.marker_motion_sim.init_marker_x_pos,
                                        self.marker_motion_sim.init_marker_y_pos), axis=-1).reshape(-1, 2)
        self.marker_data = torch.zeros((fingers, 2, 99, 2), device=device)
        self.marker_data[:, 0] = torch.as_tensor(self.init_marker_pos, device=device)
        self.cfg = SimpleNamespace(tactile_img_res=(320, 240), target_select_mode='manual',
                                   distance_metric='3d', switch_margin=.005, switch_hysteresis_steps=3)
        self._active_target_idx = torch.zeros(fingers, dtype=torch.long, device=device)
        self._stable_steps = torch.zeros(fingers, dtype=torch.long, device=device)
        self.sensor = SimpleNamespace(_data=SimpleNamespace(output={'traj': [[] for _ in range(fingers)]}))
        self.frame_transformer = SimpleNamespace(update=lambda dt: None, data=SimpleNamespace())
        wrapper = ROOT / 'fots/fots_marker_sim.py'
        cls = next(n for n in ast.parse(wrapper.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'FOTSMarkerSimulator')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'marker_motion_simulation')
        namespace = {'torch': torch, 'np': np, 'F': F, 'euler_xyz_from_quat': euler_xyz_from_quat}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(wrapper), 'exec'), namespace)
        self.original_observe = namespace[fn.name]
        self.source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [wrapper, ROOT / 'fots/sim/marker_motion.py', ROOT / 'gpu_taxim/sim/taxim_torch.py', ORIGINAL_MATH]}
        self.calibration_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in CALIB.iterdir() if p.is_file()}

    def observe(self, depth_mm, relative_positions_m, relative_quaternions_wxyz, *, source):
        if source not in ('SAPIEN_RENDER_CAMERA_DEPTH', 'RECORDED_ORIGINAL_DEPTH', 'UNIT_TEST_DEPTH_ONLY',
                          'PHYSX_COLLISION_DEPTH_DIAGNOSTIC_NOT_QUALIFIED'):
            raise ValueError('Actual sensor depth with explicit provenance is required')
        depth = torch.as_tensor(depth_mm, dtype=torch.float32, device=self._device)
        if depth.ndim != 3 or depth.shape[0] != len(self._active_target_idx) or not torch.isfinite(depth).all():
            raise ValueError('Invalid sensor depth')
        if depth.min() < 0: raise ValueError('Depth must be nonnegative millimeters')
        pos = torch.as_tensor(relative_positions_m, dtype=torch.float32, device=self._device)
        quat = torch.as_tensor(relative_quaternions_wxyz, dtype=torch.float32, device=self._device)
        if pos.shape != (depth.shape[0], 3) or quat.shape != (depth.shape[0], 4):
            raise ValueError('Invalid measured sensor-relative poses')
        self.sensor._data.output['height_map'] = depth
        # Exact original TaximSimulator.compute_indentation_depth calculation:
        distance = torch.clamp(depth.amin((1, 2)) / 1000 - .024, min=0)
        self.sensor._indentation_depth = torch.where(distance <= .0045, (.0045 - distance) * 1000, 0)
        self.frame_transformer.data.target_pos_source = pos[:, None, :]
        self.frame_transformer.data.target_quat_source = quat[:, None, :]
        value = self.original_observe(self)
        if not torch.isfinite(value).all(): raise ValueError('Original tactile simulator produced invalid markers')
        return value.detach().cpu().numpy().copy()

if __name__ == '__main__':
    core = OriginalTactileCore()
    markers = core.observe(np.full((2, 120, 160), 29., dtype=np.float32),
                           [[0,0,.029]] * 2, [[1,0,0,0]] * 2, source='UNIT_TEST_DEPTH_ONLY')
    assert np.array_equal(markers[:, 0], markers[:, 1])
    print('ORIGINAL_TACTILE_KERNEL_IMPORT_AND_NO_CONTACT_UNIT_TEST_PASSED', markers.shape)
