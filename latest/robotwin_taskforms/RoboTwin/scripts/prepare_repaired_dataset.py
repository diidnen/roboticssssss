#!/usr/bin/env python3
"""Combine legacy and repeated rows, assign balanced root-disjoint splits."""
from __future__ import annotations
import argparse, csv, itertools, json
from collections import defaultdict, Counter
from pathlib import Path


def read_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def ctx(r):
    return (str(r['task']), int(r['root_slot']), round(float(r['friction']), 8))


def root_types(rows):
    groups=defaultdict(list)
    for r in rows: groups[ctx(r)].append(r)
    per_root=defaultdict(Counter)
    for (task,slot,mu), rs in groups.items():
        ok=[bool((r.get('evidence') or {}).get('success',False)) for r in rs]
        typ='EASY' if all(ok) else ('UNRESCUED' if not any(ok) else 'TRANSITION')
        per_root[(task,slot)][typ]+=1
    return per_root


def best_split(rows):
    roots=sorted(root_types(rows))
    # Six train roots, three validation roots, three offline-test roots.
    total=Counter()
    for c in root_types(rows).values(): total.update(c)
    def score(assign):
        out=[]
        for name, chosen in assign.items():
            c=Counter()
            for root in chosen: c.update(root_types(rows)[root])
            n=sum(c.values()) or 1
            # Keep each split's type proportions close to the global mix.
            out.append(sum(abs(c[k]/n-total[k]/sum(total.values())) for k in ('EASY','TRANSITION','UNRESCUED')))
        return sum(out)
    best=None
    for val in itertools.combinations(roots,3):
        rem=[r for r in roots if r not in val]
        for test in itertools.combinations(rem,3):
            train=[r for r in rem if r not in test]
            a={'train':train,'validation':list(val),'test':list(test)}
            s=score(a)
            if best is None or s<best[0]: best=(s,a)
    return best[1]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--experiment',type=Path,required=True); ap.add_argument('--repeated',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); old=read_jsonl(a.experiment/'branches.jsonl'); rep=read_jsonl(a.repeated)
    rows=old+rep
    split=best_split(old)
    split_by_root={root:name for name, roots in split.items() for root in roots}
    counts=Counter(); per_ctx=Counter()
    for r in rows:
        root=(str(r['task']),int(r['root_slot']))
        r['split']=split_by_root[root]; counts[r['split']]+=1; per_ctx[ctx(r)]+=1
        r['source_directory']=str(a.experiment)
    a.out.mkdir(parents=True,exist_ok=True)
    combined=a.out/'repaired_records.jsonl'
    combined.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows))
    manifest={'schema_id':'AF_FORCE_SUPERVISION_REPAIR_SPLIT_V1','root_disjoint':True,'root_assignment':{k:[list(x) for x in v] for k,v in split.items()},'rows':len(rows),'split_counts':dict(counts),'context_counts':len(per_ctx),'repeated_rows':len(rep),'source_rows':len(old)}
    (a.out/'new_split_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    summary=[]
    groups=defaultdict(list)
    for r in rows: groups[ctx(r)].append(r)
    for name in ('train','validation','test'):
        g={k:v for k,v in groups.items() if v[0]['split']==name}; rs=[r for v in g.values() for r in v]
        summary.append({'split':name,'independent_contexts':len(g),'branches':len(rs),'task_distribution':json.dumps(dict(Counter(k[0] for k in g))),'friction_distribution':json.dumps(dict(Counter(str(k[2]) for k in g))),'EASY_count':sum(1 for k,v in g.items() if len([r for r in v if bool((r.get('evidence') or {}).get('success'))])==len(v)),'TRANSITION_count':sum(1 for k,v in g.items() if any(bool((r.get('evidence') or {}).get('success')) for r in v) and not all(bool((r.get('evidence') or {}).get('success')) for r in v)),'UNRESCUED_count':sum(1 for k,v in g.items() if not any(bool((r.get('evidence') or {}).get('success')) for r in v)),'overall_success_rate':sum(bool((r.get('evidence') or {}).get('success')) for r in rs)/max(1,len(rs)),'retention_success_rate':'unknown'})
    with (a.out/'split_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print(json.dumps(manifest,indent=2,sort_keys=True))

if __name__=='__main__': main()
