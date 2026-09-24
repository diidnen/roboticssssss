"""Prospective root collision check. Reads identifiers, never selects on outcomes."""
import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from common import HERE, read, write, sha

LOCATIONS = [Path('/home/exouser/FORTE'), Path('/home/exouser/Tabero'),
             Path('/home/exouser/E3_E6_E7_LANES'), Path('/media/volume/newdata/exouser')]
EXCLUDES = ['!**/.git/**', '!**/.venv/**', '!**/env_isaaclab51/**', '!**/site-packages/**',
            '!**/node_modules/**', '!**/torchinductor_cache/**', '!**/triton_cache/**']

def scan(roots, out):
    prior_path = HERE/'ROOT_IDENTIFIER_EXPOSURE_AUDIT_V3_NOIGNORE.json'
    prior = read(prior_path)
    if len(set(roots)) != 4 or any(r in prior['excluded_root_or_seed_ids'] for r in roots):
        raise RuntimeError('Candidate roots already exposed or not exactly four')
    # Decimal delimiters exclude coincidental digits inside continuous sensor
    # measurements while accepting CSV cells, quoted JSON, command-line seeds,
    # source literal lists, and root/context names.
    number = '(?:'+'|'.join(map(str, roots))+')'
    pattern = r'(^|[^0-9.])'+number+r'([^0-9.]|$)'
    locations = [p for p in LOCATIONS if p.exists()]
    globs = sum((['-g', g] for g in EXCLUDES), [])
    content_globs = sum((['-g', '*.'+ext] for ext in
        ('json','jsonl','csv','tsv','log','txt','md','yaml','yml','toml','py','sh','json.gz','jsonl.gz','csv.gz')), [])
    command = ['rg','--hidden','--no-ignore','--search-zip','--files-with-matches',
               *globs,*content_globs,pattern,*map(str, locations)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode not in (0,1):
        raise RuntimeError('Incomplete content scan: '+result.stderr)
    content_matches = result.stdout.splitlines()
    proc = subprocess.Popen(['rg','--files','--hidden','--no-ignore',*globs,*map(str,locations)],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    import re
    path_pattern = re.compile(r'(?:_r|root|_s)'+number+r'(?=[_/\.\-]|$)')
    digest=hashlib.sha256();count=0;path_matches=[]
    for line in proc.stdout:
        digest.update(line);count+=1
        value=line.decode().rstrip('\n')
        if path_pattern.search(value):path_matches.append(value)
    error=proc.stderr.read().decode()
    if proc.wait() not in (0,1):raise RuntimeError('Incomplete path scan: '+error)
    evidence = {str(prior_path):sha(prior_path)}
    old = read(HERE/'ROOT_EXPOSURE_AUDIT.json')
    for path, expected in old['source_manifest_sha256'].items():
        if sha(path)!=expected:raise RuntimeError('Primary split source changed: '+path)
        evidence[path]=expected
    report=dict(role='PROSPECTIVE_FOUR_ROOT_COLLISION_AUDIT',created_utc=datetime.now(timezone.utc).isoformat(),
        roots=roots,root_selection_used_outcomes=False,physics_branches_started_before_freeze=0,
        candidate_selection_rule='One prospective contiguous numeric block supplied before any candidate physics; reject on identifier collision, never on outcome.',
        scan_locations=list(map(str,locations)),excludes=EXCLUDES,content_pattern=pattern,
        content_command=command,content_return_code=result.returncode,content_matches=content_matches,
        path_matches=path_matches,path_count=count,path_stream_sha256=digest.hexdigest(),
        FRESH_ROOT_NONEXPOSURE_VERIFIED=not content_matches and not path_matches,
        primary_split_and_exposure_evidence_sha256=evidence,
        excluded_root_or_seed_ids=prior['excluded_root_or_seed_ids'],implementation_sha256=sha(__file__),
        scope='Current and archived local experiment text/CSV/compressed CSV metadata, source seed literals, all artifact path identifiers, primary model split manifests. Binary arrays are tied to those named/context manifests; arbitrary undocumented external experiments cannot be ruled out by a local audit.')
    write(out,report)
    print('FRESH_CANDIDATE_SCAN',report['FRESH_ROOT_NONEXPOSURE_VERIFIED'],len(content_matches),len(path_matches),count,flush=True)
    if content_matches or path_matches:raise RuntimeError('Potential root exposure requires identifier-only review; no physics allowed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs=4,type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();scan(a.roots,a.out)
