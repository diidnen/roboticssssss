import argparse
from pathlib import Path
import run_original_rootlocal_collection as original
from canonical_queue_bindings import bind

run=bind(original,'collect_original_rootlocal.py','collect_canonical_original_rootlocal.py')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);ap.add_argument('--max-groups',type=int)
    args=ap.parse_args();run(args.dataset,args.max_groups)
