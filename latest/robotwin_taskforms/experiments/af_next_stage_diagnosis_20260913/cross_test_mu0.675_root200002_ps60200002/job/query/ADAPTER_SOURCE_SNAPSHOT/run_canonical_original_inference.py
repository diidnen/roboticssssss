import argparse
from pathlib import Path
import run_original_rootlocal_inference as original
from canonical_queue_bindings import bind

run=bind(original,'infer_original_rootlocal.py','infer_canonical_original_rootlocal.py',expected_count=2)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);ap.add_argument('models',type=Path);ap.add_argument('out',type=Path)
    args=ap.parse_args();run(args.dataset,args.models,args.out)
