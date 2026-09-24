"""Verified E1-schema CPU inference on saved inputs, with explicit refusal gate.

This is an offline recovery interface, not a physical runner or final-model
approval. It never substitutes ground-truth mu, pads probes, or clips members.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import e1_verified_inference as rt


def infer(probe, commands, fold, task, mode):
    with Path(probe).open() as f:
        rows = list(csv.DictReader(f))
    raw = rt.raw_features(probe)
    members = rt.probe_members(raw[None], fold)[0]
    gate = rt.validate_live_probe(rows, members)
    result = {'status':'DIAGNOSTIC_ONLY', 'physical_rerun_allowed':False,
              'mode':mode,'fold':fold,'probe_members':members.tolist(),
              'probe_mean':float(members.mean()),'probe_std_population':float(members.std()),
              'input_gate':gate,'selected_setpoint':None,
              'caveat':'OOF historical model; no final deployment eligibility inferred',
              'source_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                               for p in [probe,commands,Path(rt.__file__),Path(__file__)]}}
    if not gate['admitted']:
        result['status']='REJECTED_CURRENT_PROBE_SUPPORT'
        return result
    if task != 0:
        raise ValueError('Current frozen [3,5] candidate domain is verified for task0 only')
    base = rt.e1_nominal_from_saved(probe,commands,task,3,float(members.mean()))
    fs = np.round(np.arange(3,5.0001,.01),2)
    support = [float(members.mean())] if mode=='point' else members
    x = np.repeat(base[None],len(fs)*len(support),axis=0)
    x[:,:,17]=np.repeat(fs/8,len(support))[:,None]
    x[:,:,18]=np.tile(support,len(fs))[:,None]
    p=rt.probabilities(x,fold).reshape(len(fs),len(support)).mean(1)
    result.update(rt.select(fs,p,5))
    result['curves']=[{'setpoint':float(f),'p_success':float(v),'utility':float(v*(2-f/5)-1)} for f,v in zip(fs,p)]
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe',required=True,type=Path)
    ap.add_argument('--commands',required=True,type=Path)
    ap.add_argument('--fold',required=True,type=int,choices=[0,1,2])
    ap.add_argument('--task',type=int,default=0)
    ap.add_argument('--mode',required=True,choices=['point','posterior'])
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args(); torch.set_num_threads(2)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing result')
    result=infer(args.probe,args.commands,args.fold,args.task,args.mode)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='curves'},indent=2))
    return 2 if result['status'].startswith('REJECTED') else 0


if __name__=='__main__':
    raise SystemExit(main())
