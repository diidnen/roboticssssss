"""Read-only summary of archived pre/post physical-probe sensor arrays."""
import argparse
import csv
from pathlib import Path
import numpy as np
from common import write,sha


def summarize(job):
    job=Path(job);rows=list(csv.DictReader((job/'RAW_PROBE.csv').open()))
    before=next(int(r['step']) for r in rows if r['probe_phase'].startswith('probe_'))-1
    after=int(rows[-1]['step'])
    paths=[job/f'PROBE_OBSERVATION_{step:04d}.npz' for step in (before,after)]
    metrics={}
    with np.load(paths[0]) as a,np.load(paths[1]) as b:
        for key in a.files:
            if key not in b.files or a[key].shape!=b[key].shape:
                metrics[key]={'shape_parity':False};continue
            delta=b[key].astype(float)-a[key].astype(float)
            metrics[key]=dict(shape=list(a[key].shape),mae=float(np.mean(np.abs(delta))),
                maximum_absolute_change=float(np.max(np.abs(delta))),fraction_elements_changed=float(np.mean(delta!=0)),
                all_values_finite=bool(np.isfinite(a[key]).all() and np.isfinite(b[key]).all()),shape_parity=True)
    report=dict(pre_probe_step=before,post_probe_step=after,
        pre_probe_artifact={'step':before,'path':str(paths[0]),'sha256':sha(paths[0])},
        post_probe_artifact={'step':after,'path':str(paths[1]),'sha256':sha(paths[1])},metrics=metrics,
        interpretation='Paired observed change, not a learned OOD score; combine with actual online continuation evidence.',
        force_strategy_unchanged=True,extra_simulator_steps=0,scripted_downstream_actions=0,
        summary_source_sha256=sha(__file__))
    write(job/'PROBE_OBSERVATION_SHIFT.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');a=p.parse_args();r=summarize(a.job)
    print('PROBE_SENSOR_SUMMARY',r['pre_probe_step'],r['post_probe_step'],list(r['metrics']))
