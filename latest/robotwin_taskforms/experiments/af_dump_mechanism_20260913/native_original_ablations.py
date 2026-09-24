"""Original transport point/prior operators with native dump data binding.

This is NOT a no-probe implementation. All controls use the common post-probe
state. Primary AF is never modified. No simulator or model fitting on import.
"""
import ast,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
V4=BASE/'af_dump_maxf8_20260913'
ORIGINAL=Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_ablation_confirmatory_v1/SOURCE_SNAPSHOT/online_ablation_feasibility.py')
sys.path.insert(0,str(V4))
from maxf8_runtime import MaxF8Feasibility

def original_operator():
    tree=ast.parse(ORIGINAL.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AblationFeasibility')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='ablated_posterior')
    namespace={'copy':copy,'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[])),str(ORIGINAL),'exec'),namespace)
    return namespace['ablated_posterior']

def empirical_train_context_prior():
    rows=[];sources={}
    for dataset in [BASE/'af_dump_original_restore_20260912/original_rootlocal_dataset_v3_rim20',V4/'additional_data_v1']:
        file=dataset/'COLLECTION_COMPLETE.json';content=file.read_bytes()
        sources[str(file)]=hashlib.sha256(content).hexdigest()
        for entry in json.loads(content)['contexts']:
            ctx=entry['context']
            if ctx['split']!='TRAIN' or ctx.get('collection_mode','FULL_GROUP')!='FULL_GROUP':continue
            rows.append({'id':ctx['id'],'mu':ctx['friction']})
    if len(rows)!=18 or len({r['id'] for r in rows})!=18:raise ValueError('Unexpected native TRAIN-context prior')
    return np.asarray([r['mu'] for r in rows],float),np.full(len(rows),1/len(rows)),rows,sources

class NativeOriginalAblation:
    def __init__(self,method,model):
        if method not in ('POSTERIOR_MEAN','PRIOR_NO_POSTERIOR'):raise ValueError('Unimplemented original ablation')
        self.method=method;self.model=model;self.operator=original_operator()
        self.sources={str(ORIGINAL):hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()}
        self.prior_rows=[]
        if method=='PRIOR_NO_POSTERIOR':
            self.prior_nodes,self.prior_weights,self.prior_rows,paths=empirical_train_context_prior()
            self.sources.update(paths)
    def select(self,feature,posterior):
        used=self.operator(self,posterior)
        decision=self.model.select(feature,used)
        decision.update(ablation_method=self.method,probe_executed=True,legal_no_probe_pathway=False,
            scientific_label='Prior / No-Posterior' if self.method=='PRIOR_NO_POSTERIOR' else 'Posterior mean',
            posterior_sigma_used=False,original_operator_AST_unchanged=True,
            native_prior_binding='empirical full TRAIN contexts,18' if self.prior_rows else None,
            prior_contexts=self.prior_rows,ablation_source_hashes=self.sources,
            used_integration_nodes=used['integration_nodes'],used_integration_weights=used['integration_weights'])
        return decision
