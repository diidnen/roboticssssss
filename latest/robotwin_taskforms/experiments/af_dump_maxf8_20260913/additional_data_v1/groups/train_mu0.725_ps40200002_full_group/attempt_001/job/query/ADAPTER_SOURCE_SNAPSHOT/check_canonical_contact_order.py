"""Check whether raw query differences are solely contact point permutations."""
import gzip
import json
from itertools import zip_longest
from rootlocal_collection_contract import HERE, write, sha


def normalize(row):
    for finger in row['contact']['fingers']:
        finger['points'].sort(key=lambda point:json.dumps(point,sort_keys=True))
    return row


def main():
    out=HERE/'mu060_canonical_ready_diagnostics_v1'
    paths=[out/f'repeat_{i:02d}/query/DIAGNOSTIC_PHYSICS_TRACE.jsonl.gz' for i in (1,2)]
    mismatched=[];rows=0
    with gzip.open(paths[0],'rt') as left,gzip.open(paths[1],'rt') as right:
        for rows,(a,b) in enumerate(zip_longest(left,right),1):
            if a is None or b is None:raise ValueError('Different trace lengths')
            if normalize(json.loads(a))!=normalize(json.loads(b)):mismatched.append(rows)
    report={'rows':rows,'all_fields_exact_after_contact_point_sort':not mismatched,
            'mismatched_steps':mismatched,'only_transformation':'sort points within each original finger; no fields removed or rounded',
            'original_raw_files_unchanged':True,'source_hashes':{str(p):sha(p) for p in paths},'source_sha256':sha(__file__)}
    write(out/'CONTACT_ORDER_COMPARISON.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
