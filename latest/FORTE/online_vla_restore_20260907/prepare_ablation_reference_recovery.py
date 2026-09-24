"""Preserve historical shared observation state when reusing dev AF controls."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from common import read, write, sha, HERE
from prepare_online_ablations import prepare


def recover(final, stopped, destination):
    failure = HERE / 'ABLATION_INITIAL_OBSERVATION_PARITY_FAILURE.json'
    if read(failure)['status'] != 'STOPPED_INITIAL_OBSERVATION_PARITY_FAILED':
        raise RuntimeError('Required stopped-evidence diagnosis missing')
    protocol = read(stopped / 'ONLINE_ABLATION_PROTOCOL.json')
    sources = {}
    for context, control_path in protocol['prospective_AF_control_reuse'].items():
        control = Path(control_path)
        gate = read(control / 'COMMON_OBSERVATION_RESTORE_GATE.json')
        reference = Path(gate['source'])
        receipt = read(reference / 'COMMON_POSTPROBE_OBSERVATION.json')
        identity = read(control / 'INITIAL_ONLINE_CHUNK_IDENTITY.json')
        if (not gate['passed'] or not gate['all_nonimage_inputs_recomputed_live_and_equal']
                or gate['vla_actions_loaded_from_disk']
                or receipt['payload_sha256'] != identity['payload_sha256']
                or sha(reference / 'COMMON_POSTPROBE_OBSERVATION.npz') != receipt['sha256']
                or sha(reference / 'DECISION_STATE.pt') != receipt['decision_state_sha256']
                or read(reference / 'WORKER_COMPLETION.json') != {'logical_success': True, 'method': 'REFERENCE'}):
            raise RuntimeError('Invalid historical reference: ' + context)
        if (read(reference / 'PREACTION_POSTERIOR.json') != read(control / 'PREACTION_POSTERIOR.json')
                or read(reference / 'CONTACT_PATCH_READBACK.json') != read(control / 'CONTACT_PATCH_READBACK.json')
                or (reference / 'RAW_PROBE.csv').read_bytes() != (control / 'RAW_PROBE.csv').read_bytes()):
            raise RuntimeError('Historical probe/posterior mismatch')
        with np.load(reference / 'COMMON_POSTPROBE_OBSERVATION.npz') as saved, np.load(control / 'RAW_OBSERVATION_0001.npz') as observed:
            for name in ('agentview_cam', 'eye_in_hand_cam'):
                if not np.array_equal(saved['camera__' + name], observed[name]):
                    raise RuntimeError('Historical camera/reference mismatch')
        sources[context] = dict(control=str(control), reference=str(reference), identity=identity)
    if len(sources) != 12:
        raise RuntimeError('Expected all 12 prospectively reused controls')
    prepare(final, destination)
    refs = destination / 'references'; refs.mkdir()
    identities = destination / 'initial_chunk_identity'; identities.mkdir()
    evidence_hashes = {str(failure): sha(failure), str(Path(__file__).resolve()): sha(__file__)}
    for context, source in sources.items():
        ref = Path(source['reference'])
        target = refs / context
        shutil.copytree(ref, target)
        for p in ref.rglob('*'):
            if p.is_file():
                copied = target / p.relative_to(ref)
                if sha(p) != sha(copied):
                    raise RuntimeError('Historical reference copy changed')
                evidence_hashes[str(copied)] = sha(copied)
        canonical = identities / (context + '.json')
        write(canonical, source['identity'])
        evidence_hashes[str(canonical)] = sha(canonical)
    receipt_path = destination / 'HISTORICAL_REFERENCE_RECOVERY.json'
    write(receipt_path, dict(stopped_evidence=str(stopped), references=sources,
        repair='Reuse complete historical reference and preseed exact initial identity before candidate execution',
        policy_actions_replayed=False, only_initial_camera_cache_restored=True,
        new_vla_inference_required=True, outcome_based_selection=False,
        prior_unmatched_coarse_branch_retained_and_excluded=True,
        method_parameters_changed=False, physics_started=False))
    manifest_path = destination / 'CANDIDATE_RUNTIME_MANIFEST.json'
    manifest = read(manifest_path)
    shutil.copy2(manifest_path, destination / 'PRE_RECOVERY_CANDIDATE_RUNTIME_MANIFEST.json')
    manifest['version'] = 'ONLINE_VLA_DEV_ABLATION_REFERENCE_RECOVERY_V2'
    manifest['source_hashes'].update(evidence_hashes)
    manifest['source_hashes'][str(receipt_path)] = sha(receipt_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared exact historical references and pre-action identities for 12 reused AF controls')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('final', 'stopped', 'destination'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    recover(a.final, a.stopped, a.destination)
