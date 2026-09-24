"""Read-only V2 raw-telemetry forensic audit; never writes V2."""
import hashlib, json
from pathlib import Path
import numpy as np

V2=Path('/media/volume/data/exouser/activeforcing_table_push_v2_20260910')
OUT=Path('/media/volume/data/exouser/activeforcing_table_push_v3_20260910/V2_ENGINEERING_FORENSIC_AUDIT.json')
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def main():
    assert not OUT.exists(), 'immutable audit already exists'
    cells=[]
    for p in sorted((V2/'calibration').glob('*/telemetry.jsonl')):
        ts=rows(p); receipt=json.loads(p.with_name('receipt.json').read_text()); active=[x for x in ts if x['active']]
        d=np.asarray(receipt['direction_record']['direction_world_xy'])
        postless=[x['force_sum_world_on_plate'] for x in active] # saved value was gathered before action
        raw=[np.sum([c['force_world_on_plate'] for c in x['raw_contacts']],axis=0) if x['raw_contacts'] else np.zeros(3) for x in active]
        proj=np.asarray(raw)[:,:2]@d
        final=np.asarray([x['executed_action'][:2] for x in active])
        nominal=np.asarray([x['nominal_action'][:2] for x in active])
        residual=np.asarray([x['residual_action'][:2] for x in active])
        cells.append({'residual':receipt['residual_normalized'],'run':receipt['run'],'steady_mean_n':float(proj[-10:].mean()),
          'raw_sum_reconstructs_saved':bool(np.allclose(raw,np.asarray(postless),atol=1e-9)),
          'force_row_alignment':'pre_action_force_written_with_executed_action',
          'active_n':len(active),'final_xy_action_clipped_n':int(np.sum(np.any(np.abs(final)>=1-1e-12,axis=1))),
          'logger_clipped_true_n':sum(bool(x['controller']['clipped']) for x in active),
          'last10_contact_counts':[x['contact_count'] for x in active[-10:]],
          'last10_actual_residual_mean':float(np.mean(np.linalg.norm(residual[-10:],axis=1))),
          'last10_downward_force_mean_n':float(np.mean(np.asarray(raw)[-10:,2])),
          'nominal_y_last10_mean':float(nominal[-10:,1].mean())})
    cells.sort(key=lambda x:x['residual']); plus12=next(x for x in cells if x['residual']==.12)
    doc={'scope':'read-only reproduction directly from V2 raw telemetry','v2_immutable':str(V2),
      'grid_normalized_residual':[x['residual'] for x in cells],'steady_force_means_n':[x['steady_mean_n'] for x in cells],
      'cells':cells,'findings':{'ordered_first_three':bool(np.all(np.diff([x['steady_mean_n'] for x in cells[:3]])>0)),
        'plus_006_plateau_vs_zero':float(cells[3]['steady_mean_n']-cells[2]['steady_mean_n']),
        'plus_012_final_y_limit_count':plus12['final_xy_action_clipped_n'],
        'plus_012_actual_last_window_residual_mean':plus12['last10_actual_residual_mean'],
        'plus_012_last_window_contact_counts':plus12['last10_contact_counts'],
        'plus_012_downward_force_n':plus12['last10_downward_force_mean_n'],
        'zero_downward_force_n':next(x for x in cells if x['residual']==0.0)['last10_downward_force_mean_n']},
      'logger_defect':'V2 clipped checked residual cap, not final nominal+residual action clipping.',
      'causal_defect':'V2 contact wrench was collected before env.step(executed) and stored in that action row.',
      'historical_decision':'V2 FALSIFIED_STOP is preserved unchanged.',
      'corrected_conclusion':'The physical conclusion is unresolved: V2 did not isolate a larger force command because saturation, state/contact drift, online replanning, and one-action telemetry offset confounded the comparison.'}
    OUT.write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__': main()
