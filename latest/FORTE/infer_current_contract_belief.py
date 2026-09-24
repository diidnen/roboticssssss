"""Read-only saved-decision inference; never executes a simulation action."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch
from current_contract_physical_belief import CurrentPhysicalBelief


def infer_saved_decision(manifest,probe_dir,*,diagnostic_only=False):
    directory=Path(probe_dir)
    with (directory/'RAW_PROBE.csv').open() as stream:raw=list(csv.DictReader(stream))
    patches=json.loads((directory/'CONTACT_PATCH_READBACK.json').read_text())
    snapshot_path=directory/'DECISION_STATE.pt'
    snapshot=torch.load(snapshot_path,map_location='cpu',weights_only=False)
    if snapshot.get('candidate_actions_already_executed')!=0:raise ValueError('Post-action snapshot forbidden')
    if snapshot.get('observation_step')!=len(raw):raise ValueError('Probe/snapshot step mismatch')
    if snapshot.get('context_id')!=raw[-1]['trial_id']:raise ValueError('Probe/snapshot context mismatch')
    runtime=CurrentPhysicalBelief(manifest,diagnostic_only=diagnostic_only)
    result=runtime.rows(raw,patches)
    if not diagnostic_only and not result['positive_support']:raise ValueError('Nonpositive physical support')
    result.update(context_id=snapshot['context_id'],decision_step=len(raw),
        decision_snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        posterior_source='real pre-action probe evidence only',diagnostic_only=diagnostic_only,
        candidate_actions_executed=0,task_branches_executed=0)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--probe-dir',required=True)
    ap.add_argument('--diagnostic-only',action='store_true')
    args=ap.parse_args()
    print(json.dumps(infer_saved_decision(args.manifest,args.probe_dir,diagnostic_only=args.diagnostic_only),indent=2))
