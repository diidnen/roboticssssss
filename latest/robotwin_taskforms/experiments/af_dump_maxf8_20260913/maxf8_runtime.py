"""Same original model/curve, corrected support-derived maxF utility contract."""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np

OLD = Path('/media/volume/data/exouser/activeforcing_robotwin_taskforms_20260911/experiments/af_dump_original_restore_20260912')
sys.path.insert(0, str(OLD))
import native_original_feasibility_binding as original
from max_force_utility import select_force


def corrected_initializer():
    tree = ast.parse(inspect.getsource(original))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeOriginalFeasibility')
    function = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'))
    baseline = ast.dump(function)
    comparisons = [n for n in ast.walk(function) if isinstance(n, ast.Compare)
                   and ast.unparse(n.left) == "m.get('utility_normalization_N')"]
    if len(comparisons) != 1 or ast.literal_eval(comparisons[0].comparators[0]) != 5.0:
        raise ValueError('Original initializer contract changed')
    old = comparisons[0].comparators[0]
    comparisons[0].comparators[0] = ast.parse("float(m['force_support'][1])", mode='eval').body
    executable = deepcopy(function)
    comparisons[0].comparators[0] = old
    if ast.dump(function) != baseline: raise ValueError('Unexpected model-loader change')
    namespace = dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[executable], type_ignores=[])), str(__file__), 'exec'), namespace)
    return namespace['__init__']


class MaxF8Feasibility(original.NativeOriginalFeasibility):
    def __init__(self, path, *, manifest_sha256, device='cpu'):
        content = Path(path).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest_sha256: raise ValueError('Manifest changed')
        manifest = json.loads(content)
        if manifest.get('force_support') != [0.5, 8.0] or manifest.get('utility_normalization_N') != 8.0:
            raise ValueError('Expected user-authorized maxF8 support/utility')
        for file in [Path(__file__).resolve(), Path(__file__).with_name('max_force_utility.py').resolve()]:
            if manifest['source_hashes'].get(str(file)) != hashlib.sha256(file.read_bytes()).hexdigest():
                raise ValueError('Corrected runtime source changed')
        corrected_initializer()(self, path, manifest_sha256=manifest_sha256, device=device)

    def select(self, feature, posterior):
        if not isinstance(feature, dict) or feature.get('task_binding') != original.TASK_BINDING:
            raise ValueError('Explicit native feature/task binding required')
        if feature.get('source') != 'ONLINE_VLA_ACTION_CHUNK' or feature.get('candidate_actions_executed') != 0:
            raise ValueError('Pre-action feature required')
        curve = self.curve(np.asarray(feature['sequence'], np.float32), posterior)
        result = select_force(self.force_grid, curve, self.manifest['force_support'])
        result.update(native_task_binding=deepcopy(original.TASK_BINDING), root_scope=[200002],
                      phase_representation='NONE', posterior_interface=self.manifest['posterior_interface'],
                      posterior_sigma_used=True, feasibility_manifest=str(self.manifest_path),
                      feasibility_manifest_sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())
        return result
