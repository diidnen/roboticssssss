"""Freeze a separate v3 only after complete independent engineering evidence."""
from copy import deepcopy
from pathlib import Path
import subprocess
from rootlocal_collection_contract import HERE,BASE,read,write,sha,now,verify_runtime,source_files
from admit_original_query import admit
from run_rim20_online_gate import prerequisites


def qualified_evidence():
    qualification=prerequisites()
    gate=HERE/'rim20_online_gate_v1'
    if not read(gate/'COMPLETE.json')['completed']:raise ValueError('Incomplete online handoff')
    audit=read(gate/'INDEPENDENT_ENGINEERING_AUDIT.json')
    if not audit['passed'] or not audit['full_duplicate_trace_exact'] or not audit['all_three_independently_audited']:
        raise ValueError('Unqualified handoff')
    if audit['source_sha256']!=sha(HERE/'audit_rim20_online_gate.py'):raise ValueError('Independent gate auditor changed')
    protocol=read(gate/'PROTOCOL.json')
    for path,digest in {**protocol['source_hashes'],**protocol['query_qualification_hashes']}.items():
        if sha(path)!=digest:raise ValueError('Qualified source/evidence changed')
    full=read(qualification/'INDEPENDENT_FULL_QUALIFICATION.json')
    if full['source_sha256']!=sha(HERE/'audit_rim20_query_qualification.py'):
        raise ValueError('Full query auditor changed')
    for item in full['cases']:
        if sha(qualification/item['case']/'ORIGINAL_FULL_ADMISSION.json')!=item['admission_sha256']:
            raise ValueError('Query admission receipt changed')
        if admit((qualification/item['case']/'query').resolve())!=read(qualification/item['case']/'ORIGINAL_FULL_ADMISSION.json'):
            raise ValueError('Original query evidence no longer reproduces admission')
    return qualification,gate


def main():
    qualification,gate=qualified_evidence()  # No v3 directory before qualification.
    old=HERE/'original_rootlocal_dataset_v2_canonical';oldlock=read(old/'FREEZE_LOCK.json')
    verify_runtime(old/'RUNTIME_MANIFEST.json',oldlock['runtime_sha256'])
    if sha(old/'COLLECTION_PROTOCOL.json')!=oldlock['protocol_sha256']:raise ValueError('Prior protocol changed')
    if sha(old/'CONTEXTS.json')!=oldlock['contexts_sha256']:raise ValueError('Prior contexts changed')
    admission=admit((gate/'job/query').resolve())
    tests={};python=str(BASE/'venv_robotwin/bin/python3')
    for name in ['test_rim20_formal_bindings.py','test_original_rootlocal_training.py','test_original_inference_audit.py']:
        result=subprocess.run([python,str(HERE/name)],cwd=HERE,capture_output=True,text=True)
        if result.returncode:raise ValueError('Pre-freeze tests failed '+name+'\n'+result.stdout+result.stderr)
        tests[name]={'source_sha256':sha(HERE/name),'exit_code':0,'stdout':result.stdout,'stderr':result.stderr}
    out=HERE/'original_rootlocal_dataset_v3_rim20';out.mkdir(exist_ok=False)
    write(out/'PRE_FREEZE_TESTS.json',tests)
    snapshot=out/'SOURCE_SNAPSHOT';snapshot.mkdir()
    sources={str(path):sha(path) for path in source_files()}
    for path,digest in sources.items():
        target=snapshot/digest
        if not target.exists():target.write_bytes(Path(path).read_bytes())
    paths=[qualification/name for name in ['COMPLETE.json','PROTOCOL.json','REPLAY_AUDIT.json','INDEPENDENT_FULL_QUALIFICATION.json']]
    paths += [gate/name for name in ['COMPLETE.json','PROTOCOL.json','INDEPENDENT_ENGINEERING_AUDIT.json']]
    paths += [HERE/'mu070_rim_alignment_diagnostics_v1/INDEPENDENT_GEOMETRY_AUDIT.json',HERE/'RIM20_FORMAL_V3_AMENDMENT.md']
    runtime=deepcopy(read(old/'RUNTIME_MANIFEST.json'))
    runtime.update(version='DUMP_ROOT200002_ORIGINAL_AF_NATIVE_V3_RIM20',created_utc=now(),source_hashes=sources,
        engineering_admission=admission,initialization_binding='CANONICAL_OPEN_READY_RIM20_V3',
        native_rim_pre_dis_offset_m=.02,rim20_engineering_gate_hashes={str(p):sha(p) for p in paths},
        prior_runtime_manifest_path=str(old/'RUNTIME_MANIFEST.json'),prior_runtime_manifest_sha256=oldlock['runtime_sha256'])
    write(out/'RUNTIME_MANIFEST.json',runtime)
    protocol=deepcopy(read(old/'COLLECTION_PROTOCOL.json'))
    protocol.update(version='DUMP_ORIGINAL_AF_ROOTLOCAL_COLLECTION_V3_RIM20',created_utc=now(),
        runtime_manifest_path=str(out/'RUNTIME_MANIFEST.json'),runtime_manifest_sha256=sha(out/'RUNTIME_MANIFEST.json'),
        initialization_binding='CANONICAL_OPEN_READY_RIM20_V3',native_rim_pre_dis_offset_m=.02,
        amendment_sha256=sha(HERE/'RIM20_FORMAL_V3_AMENDMENT.md'),
        prior_collection_protocol_path=str(old/'COLLECTION_PROTOCOL.json'),prior_collection_protocol_sha256=oldlock['protocol_sha256'],
        prior_v1_v2_labels_preserved_not_mixed_into_v3=True)
    write(out/'COLLECTION_PROTOCOL.json',protocol)
    contexts=deepcopy(read(old/'CONTEXTS.json'))
    for context in contexts:
        context.update(runtime_manifest_path=str(out/'RUNTIME_MANIFEST.json'),runtime_manifest_sha256=sha(out/'RUNTIME_MANIFEST.json'),
            collection_protocol_path=str(out/'COLLECTION_PROTOCOL.json'),collection_protocol_sha256=sha(out/'COLLECTION_PROTOCOL.json'))
    write(out/'CONTEXTS.json',contexts)
    for name in ['ROOTLOCAL_COLLECTION_PROTOCOL.md','RIM20_FORMAL_V3_AMENDMENT.md']:
        (out/name).write_bytes((HERE/name).read_bytes())
    write(out/'FREEZE_LOCK.json',{'runtime_sha256':sha(out/'RUNTIME_MANIFEST.json'),
        'protocol_sha256':sha(out/'COLLECTION_PROTOCOL.json'),'contexts_sha256':sha(out/'CONTEXTS.json')})
    write(out/'VERSIONED_GEOMETRY_BINDING.json',{'old_dataset_preserved':str(old),
        'prior_versions_not_modified_or_mixed':True,'no_diagnostic_query_admitted_as_formal_sample':True,
        'new_formal_queries_before_TEST':16,'new_formal_TRAIN_VAL_labels_planned':128,
        'same_task_specific_grasp_preparation_for_all_splits':True,'created_utc':now()})
    print({'frozen':str(out),'labels_planned':128,'prior_versions_retained':True},flush=True)


if __name__=='__main__':main()
