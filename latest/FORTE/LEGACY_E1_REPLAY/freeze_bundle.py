#!/usr/bin/env python3
"""Freeze existing historical evidence. No training, controller or simulator calls."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ARCHIVE=Path('/media/volume/newdata/exouser/ACTIVEFORCING_DISK_ARCHIVE_20260902/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000')
OLD_PREFIX='/home/exouser/Tabero/analysis/results/gnp_style_visual_context_prospective_20260831_011000'
OLD_REPLAY=ROOT/'analysis/results/e1_pipeline_recovery_and_current_rerun_20260905'
VERIFIED=ROOT/'analysis/results/e1_verified_recovery_revision_20260905'
TRANSFER=Path('/home/exouser/Tabero/analysis/activeforcing_historical_transfer_20260904/posterior')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    manifest=HERE/'MANIFEST.json'
    if manifest.exists():
        m=json.loads(manifest.read_text())
        assert all(sha(HERE/p['path'])==p['sha256'] for p in m['files'])
        print('Existing locked bundle verified; not overwritten.')
        return
    records=[]
    def copy(src,rel):
        src=Path(src);dst=HERE/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==sha(src),f'Non-identical existing bundle file: {dst}'
        else:shutil.copy2(src,dst)
        records.append({'path':str(dst.relative_to(HERE)),'sha256':sha(dst),'source':str(src),'source_sha256':sha(src)})
        return dst
    def generated(path,sources):
        records.append({'path':str(path.relative_to(HERE)),'sha256':sha(path),'derived_from':sources})
    copy(ROOT/'e1_verified_inference.py','vendor/e1_verified_inference.py')
    copy(OLD_REPLAY/'recover_e1_offline.py','vendor/recover_e1_offline.py')
    for name in ['evidence_46d.py','NORMALIZATION_46D.json']:copy(TRANSFER/name,'vendor/'+name)
    for name in ['task0_visual_context_early.py','run_pooled_predictive_verifier.py','run_residual_utility.py','utility_and_causal_ablation_closure.py','gnp_style_continuous_collect.py']:
        copy(ROOT/name,'source_evidence/'+name)
    copy('/home/exouser/Tabero/analysis/trajectory_physical_imagination.py','source_evidence/trajectory_physical_imagination.py')
    copy('/home/exouser/Tabero_old720_exact_80ab/analysis/p5s0c_paired_boundary_probe_value.py','source_evidence/archived_physical_runner.py')
    copy(ARCHIVE.parent/'p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py','source_evidence/p4_collect_probe.py')
    for fold in range(3):
        for seed in range(3):
            name=f'DIRECT_POOLED_fold{fold}_seed{seed}.pt'
            copy(ROOT/'pooled_predictive_verifier_20260901_033804/checkpoints'/name,'checkpoints/direct/'+name)
            name=f'probe_fold{fold}_seed{seed}.pt'
            copy(ROOT/'activeforcing_probe_conditioned_wm_20260901_064627/probe'/name,'checkpoints/probe/'+name)
    refs={
        'OOF_DATA.csv':ROOT/'activeforcing_residual_utility_20260901_055605/POOLED_OOF_UTILITY_DATASET.csv',
        'OOF_PROBE.csv':ROOT/'activeforcing_probe_conditioned_wm_20260901_064627/POOLED_OOF_PROBE_PREDICTIONS.csv',
        'E1_SELECTED_EPISODES.csv':ROOT/'analysis/results/ACTIVEFORCING_E1_FINAL_UTILITY_MAIN_TABLE_20260902_052942/E1_SELECTED_EPISODES.csv',
        'POSTERIOR_REFERENCE.csv':ROOT/'UTILITY_CAUSAL_ABLATION_PER_EPISODE.csv',
        'VERIFIED_HISTORICAL_INPUTS.npz':VERIFIED/'VERIFIED_HISTORICAL_INPUTS.npz',
        'ORIGINAL_SEVEN_GOLDEN.csv':OLD_REPLAY/'E1_GOLDEN_INFERENCE_CASES.csv',
        'LEGACY_407_SELECTION.json':OLD_REPLAY/'CURRENT_TASK0_E1_STYLE_SELECTION.json',
        'LEGACY_407_CURVES.csv':OLD_REPLAY/'E1_STYLE_CURRENT_TASK0_SUCCESS_CURVES.csv',
    }
    for name,src in refs.items():copy(src,'reference/'+name)
    data=pd.read_csv(refs['OOF_DATA.csv']);probe=pd.read_csv(refs['OOF_PROBE.csv'])
    contexts=probe[probe.seed.astype(str)=='0'].sort_values('probe_index')
    for r in contexts.itertuples():
        copy(Path(str(r.raw_path).replace(OLD_PREFIX,str(ARCHIVE))),f'inputs/historical_probes/{r.context_id}.csv')
    commands=[];phases=[];command_sources=[]
    for r in data.itertuples():
        source=ARCHIVE/f'collection_train/task{int(r.task)}/P5S0C_BRANCH_TELEMETRY/{r.branch_id}_trajectory.csv'
        if not source.exists():
            # Historical reconstructed task1 IDs omit the redundant rounded
            # force suffix; retain the exact unique archived trajectory.
            matches=list(source.parent.glob(r.branch_id+'_F*_trajectory.csv'))
            assert len(matches)==1,(r.branch_id,matches)
            source=matches[0]
        d=pd.read_csv(source,nrows=8);assert len(d)==8
        commands.append(d[['cmd_x','cmd_y','cmd_z']].to_numpy(float));phases.append(d.phase.to_numpy(str))
        command_sources.append({'branch_id':r.branch_id,'path':str(source),'sha256':sha(source),'fields_used':['cmd_x','cmd_y','cmd_z','phase'],'rows_used':8})
    p=HERE/'inputs/COMMAND_PREFIXES.npz'
    np.savez_compressed(p,commands=np.stack(commands),phases=np.stack(phases),branch_ids=data.branch_id.to_numpy(str))
    generated(p,command_sources)
    current=[]
    for band,mu in [('HIGH','0.940189'),('MID','0.450580'),('LOW','0.293710')]:
        cid=f'p5s0c_train_t0_r00_s5100_{band.lower()}_mu{mu}'
        src=ROOT/f'analysis/results/activeforcing_e2e_task0_smoke_20260905/PROBE_TELEMETRY/{cid}.csv'
        if not src.exists():src=src.with_name(cid+'_repeat1.csv')
        copy(src,f'inputs/legacy_407/probes/{cid}.csv')
        command=sorted((ROOT/'analysis/results/current_runtime_setpoint_mapping_validation_20260905/TASK0_TRACES').glob(cid+'_*.csv'))[0]
        copy(command,'inputs/legacy_407/commands/TASK0_TRACES/'+command.name)
        current.append(dict(band=band,context_id=cid,probe_rows=len(pd.read_csv(src))))
    protected_paths=[ROOT/'activeforcing_feasibility_features.py',ROOT/'activeforcing_e2e_task0_smoke_20260905.py',ROOT/'activeforcing_belief_contract.py',ROOT/'UTILITY_FINAL_CONFIG.json',
                     Path('/home/exouser/Tabero/analysis/p5s0c_paired_boundary_probe_value.py'),Path('/home/exouser/Tabero/analysis/results/p4_contact_conditioned_probe_20260822_184213/scripts/p4_collect_probe.py')]
    for directory in (ROOT/'analysis/results/clean_fulltask_e1_recipe_ablation_20260905').glob('variant_*'):
        protected_paths.extend(directory.glob('FEAS_seed*.pt'))
    protected=[dict(path=str(p),sha256=sha(p)) for p in protected_paths]
    for name in ['replay.py','freeze_bundle.py','README.md']:generated(HERE/name,['new isolated replay entrypoint'])
    m={'mode':'LEGACY_E1_REPLAY','scope':'offline exact numerical/decision reproduction only','bundle_absolute_path':str(HERE),
       'training':False,'physical_execution':False,'physics_entrypoint':None,
       'lanes':{
           'E1_MAIN_POINT_MU':'72 historical 215-row/10-outward-step probes; mean of 3 OOF mu members; actual archived E1 strict-preprobe masked input; five saved candidates',
           'E1_POSTERIOR_ABLATION':'same historical inputs; mean probability over 3 OOF mu supports; separate from main table',
           'LEGACY_407_452_419':'saved 206-row task0 probes + exact original buggy prefix; intentionally preserved ONLY in this isolated historical regression'},
       'utility':'p*(1-F/Fmax)+(1-p)*(-1)','historical_fmax':{'0':5,'1':6,'5':5,'6':4},
       'legacy_407_grid':[3,5,.01],'candidate_normalization':'F/8','tie_break':'historical lane atol1e-12 lower F; legacy407 original tuple(U,-F)',
       'tolerances':{'probability':1e-6,'probe_mu':1e-6,'selected_setpoint':1e-10,'feature':0.0},
       'source_commit':'E1 runtime commit unrecorded; file hashes frozen; archived physical source80ab is separate provenance',
       'no_current_pipeline_imports':True,'current_fixture_contexts':current,'files':records,'protected_current_files':protected}
    manifest.write_text(json.dumps(m,indent=2)+'\n')
    print(f'Frozen {len(records)} bundle files; manifest SHA256={sha(manifest)}')

if __name__=='__main__':main()
