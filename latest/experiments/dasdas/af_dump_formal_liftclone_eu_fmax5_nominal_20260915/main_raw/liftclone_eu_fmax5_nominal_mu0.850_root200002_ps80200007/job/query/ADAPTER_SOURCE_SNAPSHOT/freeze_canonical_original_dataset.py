"""Create a new, explicit initialization version; never mutate dataset_v1."""
from copy import deepcopy
from pathlib import Path
import subprocess
from rootlocal_collection_contract import HERE, read, write, sha, now, verify_runtime, source_files
from admit_original_query import admit


def main():
    old=HERE/'original_rootlocal_dataset_v1'
    oldlock=read(old/'FREEZE_LOCK.json')
    verify_runtime(old/'RUNTIME_MANIFEST.json',oldlock['runtime_sha256'])
    if sha(old/'COLLECTION_PROTOCOL.json')!=oldlock['protocol_sha256']:raise ValueError('Old protocol changed')
    if sha(old/'CONTEXTS.json')!=oldlock['contexts_sha256']:raise ValueError('Old contexts changed')
    gate=HERE/'canonical_ready_online_gate_v1'
    check=read(gate/'INDEPENDENT_ENGINEERING_AUDIT.json')
    if not check['passed'] or not check['full_duplicate_trace_exact']:raise ValueError('Unqualified handoff/replay')
    if not read(gate/'COMPLETE.json')['completed']:raise ValueError('Unfinished engineering')
    for source,digest in read(gate/'PROTOCOL.json')['source_hashes'].items():
        if sha(source)!=digest:raise ValueError('Qualified initializer source changed')
    if not read(HERE/'mu060_canonical_ready_diagnostics_v1/CONTACT_ORDER_COMPARISON.json')['all_fields_exact_after_contact_point_sort']:
        raise ValueError('Residual diagnostic trajectory mismatch')
    admission=admit((gate/'job/query').resolve())
    python=str(HERE.parent.parent/'venv_robotwin/bin/python3')
    tests={}
    for name in ['test_canonical_formal_bindings.py','test_original_rootlocal_training.py','test_original_inference_audit.py']:
        result=subprocess.run([python,str(HERE/name)],cwd=HERE,capture_output=True,text=True)
        if result.returncode:raise ValueError('Pre-freeze tests failed: '+name+'\n'+result.stdout+result.stderr)
        tests[name]={'source_sha256':sha(HERE/name),'exit_code':0,'stdout':result.stdout,'stderr':result.stderr}
    out=HERE/'original_rootlocal_dataset_v2_canonical';out.mkdir(exist_ok=False)
    write(out/'PRE_FREEZE_TESTS.json',tests)
    snapshot=out/'SOURCE_SNAPSHOT';snapshot.mkdir()
    sources={str(p):sha(p) for p in source_files()}
    for source,digest in sources.items():
        target=snapshot/digest
        if not target.exists():target.write_bytes(Path(source).read_bytes())
    paths=[gate/'COMPLETE.json',gate/'PROTOCOL.json',gate/'INDEPENDENT_ENGINEERING_AUDIT.json',
           HERE/'mu060_canonical_ready_diagnostics_v1/INITIALIZATION_COMPARISON.json',
           HERE/'mu060_canonical_ready_diagnostics_v1/CONTACT_ORDER_COMPARISON.json']
    runtime=deepcopy(read(old/'RUNTIME_MANIFEST.json'))
    runtime.update(version='DUMP_ROOT200002_ORIGINAL_AF_NATIVE_V2_CANONICAL',created_utc=now(),
        source_hashes=sources,engineering_admission=admission,initialization_binding='CANONICAL_OPEN_READY_V2',
        canonical_engineering_gate_hashes={str(p):sha(p) for p in paths},
        prior_runtime_manifest_path=str(old/'RUNTIME_MANIFEST.json'),prior_runtime_manifest_sha256=oldlock['runtime_sha256'])
    write(out/'RUNTIME_MANIFEST.json',runtime)
    protocol=deepcopy(read(old/'COLLECTION_PROTOCOL.json'))
    protocol.update(version='DUMP_ORIGINAL_AF_ROOTLOCAL_COLLECTION_V2_CANONICAL',created_utc=now(),
        runtime_manifest_path=str(out/'RUNTIME_MANIFEST.json'),runtime_manifest_sha256=sha(out/'RUNTIME_MANIFEST.json'),
        initialization_binding='CANONICAL_OPEN_READY_V2',
        amendment_sha256=sha(HERE/'CANONICAL_FORMAL_V2_AMENDMENT.md'),
        prior_collection_protocol_path=str(old/'COLLECTION_PROTOCOL.json'),
        prior_collection_protocol_sha256=oldlock['protocol_sha256'],
        old_40_labels_retained_as_prior_version_not_mixed_into_v2=True)
    write(out/'COLLECTION_PROTOCOL.json',protocol)
    contexts=deepcopy(read(old/'CONTEXTS.json'))
    for context in contexts:
        context.update(runtime_manifest_path=str(out/'RUNTIME_MANIFEST.json'),runtime_manifest_sha256=sha(out/'RUNTIME_MANIFEST.json'),
                       collection_protocol_path=str(out/'COLLECTION_PROTOCOL.json'),collection_protocol_sha256=sha(out/'COLLECTION_PROTOCOL.json'))
    write(out/'CONTEXTS.json',contexts)
    for name in ['ROOTLOCAL_COLLECTION_PROTOCOL.md','CANONICAL_FORMAL_V2_AMENDMENT.md']:
        (out/name).write_bytes((HERE/name).read_bytes())
    write(out/'FREEZE_LOCK.json',{'runtime_sha256':sha(out/'RUNTIME_MANIFEST.json'),
          'protocol_sha256':sha(out/'COLLECTION_PROTOCOL.json'),'contexts_sha256':sha(out/'CONTEXTS.json')})
    write(out/'VERSIONED_INITIALIZATION_REPAIR.json',{'old_dataset_preserved':str(old),
        'old_labels_not_overwritten':40,'old_failure_retained':'train_mu0.600_root200002/attempt_001',
        'no_diagnostic_query_admitted_as_formal_training_sample':True,'new_formal_queries_before_TEST':16,
        'new_formal_TRAIN_VAL_labels_planned':128,'created_utc':now()})
    print({'frozen':str(out),'labels_planned':128,'old_labels_preserved':40},flush=True)


if __name__=='__main__':main()
