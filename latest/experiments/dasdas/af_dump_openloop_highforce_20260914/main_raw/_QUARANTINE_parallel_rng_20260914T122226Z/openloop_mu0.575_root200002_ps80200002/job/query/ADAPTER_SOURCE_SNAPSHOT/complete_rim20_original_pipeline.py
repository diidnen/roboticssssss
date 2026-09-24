"""Original train/infer pipeline, identical geometry on TEST, independent audit."""
import argparse
from pathlib import Path
import complete_original_rootlocal_pipeline as original
from canonical_queue_bindings import bind
from rootlocal_collection_contract import HERE,BASE,read,write,sha,now

run_original=bind(original,'run_original_rootlocal_inference.py','run_rim20_original_inference.py',expected_count=2)


def run(dataset,models,inference,out):
    run_original(dataset,models,inference,out)
    out=Path(out).resolve();inference=Path(inference).resolve()
    sources={str(p):sha(p) for p in [HERE/'audit_original_inference_results.py',HERE/'test_original_inference_audit.py']}
    python=str(BASE/'venv_robotwin/bin/python3')
    original.execute('final_audit_tests',[python,str(HERE/'test_original_inference_audit.py')],out,sources)
    original.execute('independent_final_audit',[python,str(HERE/'audit_original_inference_results.py'),str(inference)],out,sources)
    if not read(inference/'INDEPENDENT_FINAL_RESULT_AUDIT.json')['passed']:raise ValueError('Final audit failed')
    write(out/'RIM20_PIPELINE_AUDITED.json',{'completed':True,'finished_utc':now(),
          'final_audit_sha256':sha(inference/'INDEPENDENT_FINAL_RESULT_AUDIT.json'),'Anvil_backup_still_required':True})


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('dataset','models','inference','out'):ap.add_argument(name,type=Path)
    args=ap.parse_args();run(args.dataset,args.models,args.inference,args.out)
