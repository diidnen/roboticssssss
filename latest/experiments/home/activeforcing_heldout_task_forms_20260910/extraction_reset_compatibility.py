"""Config-only repair for ignored native open-drawer initial state.

No model, force control, task terminal criterion, object pose, or root changes.
"""
import json
from pathlib import Path

def repair(cfg, suite, task, receipt_path):
    if (suite,task)!=('libero_spatial',4):raise ValueError('Repair applies only to the audited extraction task')
    source=Path('/home/exouser/Tabero/benchmarks/datasets/libero/config/libero_spatial.json')
    record=json.loads(source.read_text())['tasks'][4]
    value=record['regions']['main_table_cabinet_region']['pose_range']['joint_pos_range']['0']
    assert value[0]==value[1]
    # The source task explicitly identifies the TOP drawer. Resolve by joint
    # name, not by the ordering of an articulation's runtime joint tensor.
    previous=dict(cfg.scene.wooden_cabinet_1.init_state.joint_pos)
    cfg.scene.wooden_cabinet_1.init_state.joint_pos['top_level']=value[0]
    Path(receipt_path).write_text(json.dumps({'source':str(source),'task':4,'named_joint':'top_level',
        'source_value':value,'before':previous,'after':cfg.scene.wooden_cabinet_1.init_state.joint_pos,
        'reason':'randomize_object_pose ignores joint_pos_range; honor native specified initial condition',
        'VLA_or_AF_outcomes_used':False,'terminal_evaluator_changed':False,'object_pose_changed':False},indent=2))
