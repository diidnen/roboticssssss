"""Read-only exact V1 telemetry audit; writes V2's audit artifact once."""
import glob, hashlib, json
from pathlib import Path
import numpy as np

V1=Path('/media/volume/data/exouser/activeforcing_table_push_20260910')
OUT=Path('/media/volume/data/exouser/activeforcing_table_push_v2_20260910/V1_FORENSIC_AUDIT.json')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def mean_or_none(a): return None if not a else float(np.mean(a))
def main():
    assert not OUT.exists(), f'immutable artifact already exists: {OUT}'
    branches=[]
    for receipt_path in sorted(V1.glob('rollouts/*/receipt.json')):
        rec=json.loads(receipt_path.read_text())
        if rec.get('error') is not None: continue
        tel=receipt_path.with_name('telemetry.jsonl')
        rows=[json.loads(x) for x in tel.read_text().splitlines() if x]
        d=np.asarray(rec['direction'],float)
        active=[x for x in rows if x['controller'].get('active')]
        contact=[x for x in rows if x['contact_count']>0]
        # Exactly the V1 controller's signed selected-force definition: sum world-on-plate dot frozen direction.
        projection=[float(np.dot(x['force_sum_world_on_plate'],d)) for x in active]
        ema=[x['controller'].get('ema_force_n') for x in active if x['controller'].get('ema_force_n') is not None]
        nominal_xy=np.asarray([x['nominal_action'][:2] for x in active],float)
        force_xy=np.asarray([x['force_sum_world_on_plate'][:2] for x in active],float)
        residual_norm=[float(np.linalg.norm(x['controller'].get('residual_xy',[0,0]))) for x in active]
        branches.append({
          'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),'telemetry_sha256':sha(tel),
          'target_push_n':rec['target_push_n'],'frozen_direction_world':d.tolist(),
          'all_contact_steps':len(contact),'active_contact_steps':len(active),
          'active_nominal_xy_mean':nominal_xy.mean(axis=0).tolist() if len(active) else None,
          'active_force_world_on_plate_xy_mean_n':force_xy.mean(axis=0).tolist() if len(active) else None,
          'active_signed_projection_mean_n':mean_or_none(projection),'active_signed_projection_median_n':None if not projection else float(np.median(projection)),
          'active_ema_mean_n':mean_or_none(ema),'active_ema_final_n':None if not ema else float(ema[-1]),
          'residual_norm_min_max':[min(residual_norm),max(residual_norm)] if residual_norm else None,
          'residual_limit_saturated_steps':sum(x >= .02-1e-12 for x in residual_norm),
          'active_action_dimension_invariant':all(np.allclose(x['residual_action'][2:],0) for x in active),
        })
    doc={'artifact':'V1 forensic audit, read-only source analysis','v1_path':str(V1),'v1_immutable':True,
      'calculation_definition':{'force':'sum of MuJoCo robot-on-plate contact forces transformed to world and signed by receipt frozen direction','phase':'V1 controller active rows only','nominal_xy_mean':'arithmetic mean of V1 nominal action[:2] over that phase'},
      'branches':branches,
      'conclusion':'V1 is retained as a valid negative engineering result. Its single-action direction is inconsistent with ensuing nominal motion/force axis; this supports a testable V2 causal-chunk redesign only, not a claim that force tracking or task recovery will pass.'}
    OUT.write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__': main()
