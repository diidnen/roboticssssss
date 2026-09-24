import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'sensor_dependencies'))
from pxr import Usd, UsdGeom

asset_root = Path('/home/exouser/Tabero/source/tac_manip/tac_manip/assets/data')
paths = [asset_root / 'Sensors/GelSight_Mini/Sensor.usd']
robot = asset_root / 'Robots/Franka_gsmini'
paths.extend(robot.glob('*.usd'))
rows = []
for path in paths:
    stage = Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera): continue
        cam = UsdGeom.Camera(prim)
        rows.append({'asset': str(path), 'camera': str(prim.GetPath()),
                     'focalLength': cam.GetFocalLengthAttr().Get(),
                     'horizontalAperture': cam.GetHorizontalApertureAttr().Get(),
                     'verticalAperture': cam.GetVerticalApertureAttr().Get(),
                     'clippingRange': list(cam.GetClippingRangeAttr().Get()),
                     'projection': cam.GetProjectionAttr().Get(),
                     'local_transform': [list(r) for r in UsdGeom.Xformable(prim).GetLocalTransformation()]})
print(json.dumps(rows, indent=2))
Path(__file__).with_name('ORIGINAL_CAMERA_ASSET.json').write_text(json.dumps(rows, indent=2))
