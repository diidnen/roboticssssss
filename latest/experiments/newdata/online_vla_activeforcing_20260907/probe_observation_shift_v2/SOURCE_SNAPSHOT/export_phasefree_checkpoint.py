"""Serialize exact frozen-column deletion; no fitting or model selection."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/home/exouser/FORTE')
from common import HERE,read,write,sha
from phase_free_feasibility import PhaseFreeFeasibility,KEEP_INPUT,KEEP_SEQUENCE

def main():
    torch.set_num_threads(2)
    out=HERE/'phase_free_exact_checkpoints';out.mkdir(exist_ok=False)
    model=PhaseFreeFeasibility();files=[]
    for original,network in zip(model.manifest['checkpoints'],model.models):
        old=torch.load(original['path'],map_location='cpu',weights_only=False)
        state={k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
        path=out/f'PHASE_FREE_EXACT_seed{old["seed"]}.pt'
        record=dict(state_dict=state,normalization_mean=model.mean,normalization_std=model.std,
            input_shape=[8,64],sequence_dim=10,condition_dim=54,
            original_input_indices=KEEP_INPUT,original_sequence_indices=KEEP_SEQUENCE,
            seed=old['seed'],original_checkpoint=original['path'],original_checkpoint_sha256=sha(Path(original['path'])),
            original_training_protocol_sha256=old['protocol_sha256'],original_selected_epoch=old['selected_epoch'],
            target=old['target'],derivation='DELETE_7_NORMALIZED_CONSTANT_ZERO_PHASE_INPUT_COLUMNS',
            new_training_steps=0,online_vla_transfer_qualified=False)
        torch.save(record,path)
        restored=torch.load(path,map_location='cpu',weights_only=False)
        assert all(torch.equal(v,restored['state_dict'][k]) for k,v in state.items())
        assert np.array_equal(restored['normalization_mean'],model.mean)
        assert np.array_equal(restored['normalization_std'],model.std)
        assert state['command_gru.weight_ih_l0'].shape==(192,10)
        files.append(dict(path=str(path),sha256=sha(path),seed=old['seed'],
            original_checkpoint_sha256=record['original_checkpoint_sha256'],parameter_count=sum(v.numel() for v in state.values()),
            reload_state_parity=True))
    manifest=dict(version='PHASE_FREE_EXACT_DERIVATIVE_CANDIDATE_V1',role='CANDIDATE_NOT_FINAL_RUNTIME_FREEZE',
        source_runtime_manifest=str(model.manifest_path),source_runtime_manifest_sha256=sha(model.manifest_path),
        derivation_source_sha256=sha(HERE/'phase_free_feasibility.py'),exporter_sha256=sha(Path(__file__)),
        checkpoints=files,input_shape=[8,64],phase_representation='NONE',new_training_steps=0,
        source_647_prediction_and_planner_parity=read(HERE/'offline_feature_audit_v3/PHASE_SENSITIVITY_SUMMARY.json')['results']['PHASE_REMOVED_EXACT']['vs_original'],
        online_chunk_runtime_parity=read(HERE/'PHASE_FREE_RUNTIME_PARITY.json'),
        online_transfer_status='PENDING_36_BRANCH_QUALIFICATION',
        qualification_runtime='dev_v4_phasefree performs this identical conversion in memory from original checkpoints; export changes storage only')
    write(out/'PHASE_FREE_EXACT_MANIFEST.json',manifest)
    print(json.dumps(files,indent=2))

if __name__=='__main__':main()
