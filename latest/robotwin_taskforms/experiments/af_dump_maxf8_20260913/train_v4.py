"""Original AF architectures/recipes, added audited data, maxF8 and bounded longer fit."""
import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'af_dump_original_restore_20260912'
sys.path.insert(0, str(OLD))
import train_original_rootlocal as original
from rootlocal_collection_contract import read, write, sha, now, verify_runtime
from native_original_evidence_adapter import original_evidence
from current_contract_physical_belief import verify_decision_prefix
from maxf8_runtime import MaxF8Feasibility
from diagnose_existing_models import extended_trainer


def load_added():
    dataset = HERE/'additional_data_v1'; lock = read(dataset/'FREEZE_LOCK.json')
    verify_runtime(dataset/'RUNTIME_MANIFEST.json', lock['runtime_sha256'])
    if sha(dataset/'CONTEXTS.json') != lock['contexts_sha256'] or sha(dataset/'COLLECTION_PROTOCOL.json') != lock['protocol_sha256']:
        raise ValueError('Added dataset plan changed')
    done = read(dataset/'COLLECTION_COMPLETE.json'); contexts = read(dataset/'CONTEXTS.json')
    if not done['completed'] or done['new_downstream_labels'] != 64 or done['query_only_groups'] != 16:
        raise ValueError('Incomplete added collection')
    if [r['context'] for r in done['contexts']] != contexts: raise ValueError('Incomplete context schedule')
    records, inputs = [], {}
    for item in done['contexts']:
        context = item['context']; attempt = Path(item['attempt']); job = attempt/'job'
        if not item['accepted'] or read(attempt/'PROCESS_EXIT.json')['exit_code'] != 0: raise ValueError('Invalid process')
        sources = dict(item['artifact_hashes'])
        query_audit = job/'query/INDEPENDENT_NATIVE_FORCE_REBUILD.json'
        audit = read(query_audit)
        if not audit['passed'] or audit['audit_source_sha256'] != sha(OLD/'audit_original_collected_group.py'):
            raise ValueError('Query audit invalid')
        sources.update(audit['source_hashes']); sources[str(query_audit)] = sha(query_audit)
        branches, feature = [], None
        if context['collection_mode'] == 'FULL_GROUP':
            group = read(job/'INDEPENDENT_GROUP_AUDIT.json')
            if not group['passed'] or group['branches'] != 8: raise ValueError('Group audit invalid')
            if group['query_audit_sha256'] != sha(query_audit): raise ValueError('Query audit changed')
            for branch, digest in group['branch_audit_hashes'].items():
                p = Path(branch)/'INDEPENDENT_BRANCH_AUDIT.json'; sources[str(p)] = digest
                report = read(p)
                if not report['passed'] or report['audit_source_sha256'] != sha(OLD/'audit_original_collected_group.py'):
                    raise ValueError('Branch audit invalid')
                sources.update(report['source_hashes'])
            branches = read(job/'COLLECTED_ROWS.json')
            if [r['force'] for r in branches] != context['forces_N'] or len(branches) != 8:
                raise ValueError('Missing candidate force labels')
            feature = read(Path(branches[0]['job'])/'original_motion_feature.json')
        for file, digest in sources.items():
            if sha(file) != digest: raise ValueError('Audited input changed: '+file)
        inputs.update(sources)
        if not read(job/'QUERY_ADMISSION.json')['admitted']: raise ValueError('Query not admitted')
        materials = read(job/'query/ACTUAL_OBJECT_MATERIAL.json')['shape_materials']
        if not materials or not np.allclose(materials, context['friction'], atol=1e-6, rtol=0):
            raise ValueError('Training target material mismatch')
        raw = read(job/'query/original_raw_rows.json'); patch = read(job/'query/patch_readbacks.json')
        verify_decision_prefix(raw); values = original_evidence(raw, patch)
        np.testing.assert_array_equal(values, np.load(job/'query/original58_engineering.npy', allow_pickle=False))
        records.append({'id':context['id'],'root':200002,'split':context['split'],'y':context['friction'],
                        'raw':values,'query_raw':raw,'patch':patch,'branches':branches,'feature':feature,
                        'reference':str(job),'context':context})
    return records, inputs, lock


def extended_feasibility():
    tree = ast.parse(inspect.getsource(original.train_feasibility)); baseline = ast.dump(tree); changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 81:
            changes.append((node,81)); node.value = 401
        elif isinstance(node, ast.Constant) and isinstance(node.value,float) and node.value == 5.0:
            changes.append((node,5.0)); node.value = 8.0
    if len(changes) != 3: raise ValueError('Unexpected training budget/utility constant layout')
    executable = deepcopy(tree)
    for node, value in changes: node.value = value
    if ast.dump(tree) != baseline: raise ValueError('Unexpected training algorithm changes')
    namespace = dict(original.__dict__)
    namespace['NativeOriginalFeasibility'] = MaxF8Feasibility
    def write_with_corrected_sources(path, value):
        if Path(path).name == 'FEASIBILITY_MANIFEST.json':
            value = deepcopy(value)
            for name in ['maxf8_runtime.py','max_force_utility.py']:
                value['source_hashes'][str(HERE/name)] = sha(HERE/name)
            value['utility_definition'] = 'p*(maxF-F)/maxF-(1-p)'
        write(path, value)
    namespace['write'] = write_with_corrected_sources
    exec(compile(ast.fix_missing_locations(executable), str(__file__), 'exec'), namespace)
    return namespace['train_feasibility']


def main():
    previous, inputs, oldlock = original.load_dataset(OLD/'original_rootlocal_dataset_v3_rim20')
    added, new_inputs, newlock = load_added(); inputs.update(new_inputs)
    records = previous + added
    full = [r for r in records if r['branches']]
    if sum(len(r['branches']) for r in full) != 192: raise ValueError('Wrong total label count')
    unique, hashes = {}, {}
    for record in records:
        key = hashlib.sha256(record['raw'].tobytes()).hexdigest(); hashes[record['id']] = key
        if key in unique:
            if (record['split'],record['y']) != (unique[key]['split'],unique[key]['y']):
                raise ValueError('Exact query duplicate crosses split or label')
        else: unique[key] = record
    out = HERE/'models_v4'; out.mkdir(exist_ok=False)
    protocol = {'version':'MAXF8_ORIGINAL_AF_ADDED_DATA_V4','created_utc':now(),
                'root_scope':[200002],'input_hashes':inputs,'data_locks':[oldlock,newlock],
                'belief_architecture':[58,16,16],'feasibility_feature_shape':[8,64],
                'belief_max_updates':400,'feasibility_max_epochs':400,'seeds':[0,1,2],
                'selection':'per seed first minimum VAL NLL; original Gaussian / posterior-marginal criteria',
                'loss_optimizer_normalization_original':True,'query_deduplication':hashes,
                'unique_belief_queries':len(unique),'feasibility_labels':192,
                'force_support':[0.5,8.0],'planner_grid_step':0.05,'utility_normalization_N':8.0,
                'no_test_for_training_or_selection':True,
                'source_hashes':{str(p):sha(p) for p in [Path(__file__),HERE/'maxf8_runtime.py',HERE/'max_force_utility.py',
                     HERE/'diagnose_existing_models.py',OLD/'train_original_rootlocal.py']}}
    write(out/'TRAINING_PROTOCOL.json',protocol)
    write(out/'TRAINING_PROTOCOL_LOCK.json',{'sha256':sha(out/'TRAINING_PROTOCOL.json')})
    extended_trainer()(list(unique.values()),out/'belief',out/'TRAINING_PROTOCOL.json',newlock['runtime_sha256'])
    for record in records:
        record['posterior'] = unique[hashes[record['id']]]['posterior']
    extended_feasibility()(full,out/'feasibility',out/'TRAINING_PROTOCOL.json')
    for path,digest in inputs.items():
        if sha(path) != digest: raise ValueError('Training input changed')
    write(out/'TRAINING_COMPLETE.json',{'completed':True,'protocol_sha256':sha(out/'TRAINING_PROTOCOL.json'),
          'belief_manifest_sha256':sha(out/'belief/BELIEF_MANIFEST.json'),
          'feasibility_manifest_sha256':sha(out/'feasibility/FEASIBILITY_MANIFEST.json'),
          'test_groups_executed':0,'utility_normalization_N':8.0,'finished_utc':now()})
    print('V4_TRAINING_COMPLETE',flush=True)


if __name__ == '__main__': main()
