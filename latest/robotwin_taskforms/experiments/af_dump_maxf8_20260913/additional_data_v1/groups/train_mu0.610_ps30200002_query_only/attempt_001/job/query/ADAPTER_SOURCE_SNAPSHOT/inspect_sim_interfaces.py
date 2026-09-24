"""Inspect installed APIs and original sensor dependencies without a rollout."""
import importlib
import json
import sapien

def methods(cls):
    return {n: str(getattr(cls, n).__doc__)[:1600] for n in dir(cls)
            if any(s in n.lower() for s in ('pack', 'state', 'velocity', 'qpos', 'force', 'camera'))}

report = {}
for name, cls in [('Scene', sapien.Scene), ('PhysxCpuSystem', sapien.physx.PhysxCpuSystem),
                  ('PhysxArticulation', sapien.physx.PhysxArticulation),
                  ('RenderCameraComponent', sapien.render.RenderCameraComponent)]:
    report[name] = methods(cls)
for name in ['torch', 'torchvision', 'torch_scatter', 'cv2', 'trimesh']:
    try:
        mod = importlib.import_module(name)
        report[name] = {'path': mod.__file__, 'version': getattr(mod, '__version__', None)}
    except Exception as e:
        report[name] = {'error': repr(e)}
print(json.dumps(report, indent=2))
