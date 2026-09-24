"""Locate every differing high-rate field without discarding physical fields."""
import gzip
import json
from itertools import zip_longest
from rootlocal_collection_contract import HERE, read, write, sha


def differences(a,b,path=''):
    if type(a)!=type(b):yield path,a,b;return
    if isinstance(a,dict):
        for key in sorted(a.keys()|b.keys()):
            if key not in a or key not in b:yield path+'/'+key,a.get(key),b.get(key)
            else:yield from differences(a[key],b[key],path+'/'+key)
    elif isinstance(a,list):
        if len(a)!=len(b):yield path+'/length',len(a),len(b)
        for index,(left,right) in enumerate(zip(a,b)):
            yield from differences(left,right,path+'/*')
    elif a!=b:yield path,a,b


def main():
    out=HERE/'mu060_canonical_ready_diagnostics_v1'
    paths=[out/f'repeat_{i:02d}/query/DIAGNOSTIC_PHYSICS_TRACE.jsonl.gz' for i in (1,2)]
    changed={};rows=0
    with gzip.open(paths[0],'rt') as left,gzip.open(paths[1],'rt') as right:
        for rows,(a,b) in enumerate(zip_longest(left,right),1):
            if a is None or b is None:raise ValueError('Mismatched trace length')
            a,b=json.loads(a),json.loads(b)
            for key,x,y in differences(a,b):
                record=changed.setdefault(key,{'count':0,'first_physics_step':rows,'first_values':[x,y],
                                               'max_numeric_abs_difference':0.})
                record['count']+=1
                if type(x) in (int,float) and type(y) in (int,float):
                    record['max_numeric_abs_difference']=max(record['max_numeric_abs_difference'],abs(x-y))
    report={'rows':rows,'different_field_paths':changed,'no_fields_omitted':True,
            'source_hashes':{str(p):sha(p) for p in paths},'source_sha256':sha(__file__)}
    write(out/'HIGH_RATE_FIELD_DIFFERENCES.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
