"""Native preflight with only the source-specified drawer default restored."""
from pathlib import Path
P=Path(__file__).resolve().parent
source=(P/'preflight_worker.py').read_text()
needle="    receipt['config_loaded']=True"
assert source.count(needle)==1
source=source.replace(needle,"    from extraction_reset_compatibility import repair\n    repair(cfg,a.suite,a.task,job/'RESET_COMPATIBILITY.json')\n"+needle)
needle="    receipt['scene_articulations']=list(env.scene.articulations)"
assert source.count(needle)==1
source=source.replace(needle,needle+"\n    receipt['cabinet_joint_names']=env.scene['wooden_cabinet_1'].joint_names\n    receipt['cabinet_joint_positions']=env.scene['wooden_cabinet_1'].data.joint_pos.cpu().tolist()")
exec(compile(source,str(P/'preflight_worker.py')+'::extraction_reset_compatibility','exec'))
