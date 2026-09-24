import json
from pathlib import Path
def test_frozen_grid_and_targets():
 p=Path('/media/volume/data/exouser/activeforcing_table_push_v5_20260910')
 assert json.load(open(p/'V5_R2H_PROTOCOL.json'))['hybrid_axis_grid_n']==[10,15,20,25,30]
 assert json.load(open(p/'A3_TRACKING_CONTRACT.json'))['targets_n']==[19.5,20.5,21.5]
