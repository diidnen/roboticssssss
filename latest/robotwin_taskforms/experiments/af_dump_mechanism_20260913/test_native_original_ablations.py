"""Bounded CPU operator tests and offline-only selections; no TEST fitting."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import torch
from native_original_ablations import NativeOriginalAblation,MaxF8Feasibility,V4,BASE
torch.set_num_threads(1)
path=V4/'models_v4/feasibility/FEASIBILITY_MANIFEST.json'
model=MaxF8Feasibility(path,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
controls=[NativeOriginalAblation(m,model) for m in ['POSTERIOR_MEAN','PRIOR_NO_POSTERIOR']]
rows=[]
for job in sorted((BASE/'af_dump_maxf8_confirmation_storage_20260913/inference_v4').glob('test_*/job')):
    p=json.loads((job/'PREACTION_POSTERIOR.json').read_text());f=json.loads((job/'PREACTION_FEATURE.json').read_text())
    before=copy.deepcopy(p);r={'id':job.parent.name,'offline_diagnostic_only':True}
    for c in controls:
        used=c.operator(c,p);assert p==before
        if c.method=='POSTERIOR_MEAN':
            assert used['integration_nodes']==[p['posterior_moments']['mean']]
            assert used['integration_weights']==[1.] and used['posterior_moments']['std']==0
        else:
            assert len(used['integration_nodes'])==18
            np.testing.assert_allclose(used['integration_weights'],np.full(18,1/18),atol=0,rtol=0)
        assert all(used[k]==p[k] for k in p if k not in ['integration_nodes','integration_weights','posterior_moments'])
        d=c.select(f,p);assert .5<=d['selected_force_N']<=8 and not d['legal_no_probe_pathway']
        r[c.method]={'force':d['selected_force_N'],'p':d['predicted_success']}
    rows.append(r)
print(json.dumps({'passed':True,'original_operator_unchanged':True,'model_fitting':False,
    'legal_no_probe':False,'original_prior_rule':'empirical full TRAIN contexts',
    'native_prior_rows':controls[1].prior_rows,'source_hashes':controls[1].sources,'rows':rows}))
