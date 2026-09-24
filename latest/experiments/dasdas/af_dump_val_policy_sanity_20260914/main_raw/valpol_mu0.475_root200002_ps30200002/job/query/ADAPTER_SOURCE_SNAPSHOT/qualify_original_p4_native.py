"""Execute original P4 query on actual dump root; retain raw engineering data."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
import qualify_native_interfaces as base
from original_p4_native_protocol import load_protocol
from native_p4_query_env import NativeP4QueryEnv
from native_original_evidence_adapter import original_evidence


def qualify(env,out):
    out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'ADAPTER_SOURCE_SNAPSHOT';snapshot.mkdir()
    source_hashes={}
    for source in sorted(Path(__file__).parent.glob('*.py')):
        data=source.read_bytes()
        (snapshot/source.name).write_bytes(data)
        source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    (out/'ADAPTER_SOURCE_HASHES_BEFORE.json').write_text(json.dumps(source_hashes,indent=2))
    protocol=load_protocol()
    friction=float(env.af_contact_friction)
    report={'scope':'original P4 native query engineering','root':200002,'friction':friction,
            'formal_collection_gate_passed':False,'protocol':protocol.binding_receipt,
            'task_success_evaluated':False,'sensor_depth_binding_still_diagnostic':True}
    bridge=None
    try:
        bridge=NativeP4QueryEnv(env,out)
        bridge.ready()
        report['sensor_depth_source']=bridge.depth_source
        report['sensor_depth_binding_still_diagnostic']=not bridge.depth_source.startswith('SAPIEN_RENDER')
        report['squeeze_controller_binding']=('ORIGINAL_OUTER_AND_ORIGINAL_INNER_POSITION_FEEDBACK'
                                             if bridge.inner else 'ORIGINAL_OUTER_ONLY_NATIVE_FORCE_CAP_DIAGNOSTIC')
        report['original_inner_receipt']=bridge.inner.receipt if bridge.inner else None
        sys.path.insert(0,'/home/exouser/FORTE')
        from activeforcing_probe_friction_contract import FrictionProbeBudget
        budget=FrictionProbeBudget(bridge.readbacks,protocol._quat_apply_np)
        rows,summary=protocol.run_probe_episode(bridge,seed_idx=200002,mu=friction,
            trial_id=out.name,dt=.05,termination_signal=budget)
        raw=[asdict(row) for row in rows]
        (out/'original_raw_rows.json').write_text(json.dumps(raw,indent=2))
        (out/'original_probe_summary.json').write_text(json.dumps(summary,indent=2))
        features=original_evidence(raw,bridge.readbacks)
        np.save(out/'original58_engineering.npy',features)
        report.update(completed=True,rows=len(rows),features_shape=list(np.asarray(features).shape),
            finite_features=bool(np.isfinite(features).all()),phases={phase:sum(r.probe_phase==phase for r in rows)
                for phase in sorted(set(r.probe_phase for r in rows))},
            probe_failure=summary['probe_failure'],stop_trigger=summary['stop_trigger'],
            true_elapsed_s=bridge.elapsed_s,physics_steps=bridge.physics_steps,
            nominal_period_s=.05,actual_periods_s=sorted(set(r['interval_s'] for r in bridge.readbacks)))
    except BaseException as exc:
        report.update(completed=False,error=repr(exc),traceback=traceback.format_exc(),
                      query_steps_completed=bridge.step_count if bridge is not None else 0)
        raise
    finally:
        if bridge is not None:bridge.save()
        (out/'qualification.json').write_text(json.dumps(report,indent=2))
        print('ORIGINAL_P4_NATIVE_QUALIFICATION '+json.dumps(report),flush=True)


if __name__=='__main__':
    os.environ['AF_QUALIFICATION_BEFORE_SUPPLIED_GRASP']='1'
    base.qualify=qualify
    base.main()
