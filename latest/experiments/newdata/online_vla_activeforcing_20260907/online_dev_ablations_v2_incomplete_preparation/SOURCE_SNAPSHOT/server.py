"""Dedicated frozen original pi0; logs actual per-request model execution.

No training, action replay, or fallback. Explicit request-local noise makes
multiple worker schedules independent, while every request samples online.
"""
import argparse
import logging
import os
import time
import numpy as np
from common import CHECKPOINT, VTLA, HERE, sha, read, write, payload_sha, array_sha, checkpoint_inventory

def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', required=True); a.add_argument('--port', type=int, default=18885)
    args = a.parse_args()
    from pathlib import Path
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    inventory = checkpoint_inventory()
    expected = '0598a390733235fde0bf5633b1543d91176d31bb5012d4d26f9373a90642fe17'
    if inventory['sha256'] != expected: raise RuntimeError('Original pi0 checkpoint changed')
    write(out/'CHECKPOINT_INVENTORY.json', inventory)
    import jax
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    cfg = config.get_config('pi0_lora_tacfield_tabero')
    policy = policy_config.create_trained_policy(cfg, CHECKPOINT,
        norm_stats=normalize.load(CHECKPOINT/'assets/NathanWu7/tabero'), pytorch_device='cuda:0')
    if policy._is_pytorch_model: raise RuntimeError('Expected original frozen JAX pi0')
    paths = [HERE/'server.py', HERE/'common.py'] + list((VTLA/'src/openpi').rglob('*.py'))
    metadata = {'downstream_action_source':'ONLINE_VLA', 'checkpoint_loaded':True,
        'checkpoint':str(CHECKPOINT), 'checkpoint_sha256':inventory['sha256'],
        'policy_config':'pi0_lora_tacfield_tabero', 'backend':'JAX_PI0', 'pid':os.getpid(),
        'source_hashes':{str(p):sha(p) for p in paths},
        'noise_shape':[cfg.model.action_horizon,cfg.model.action_dim],
        'request_rng':'numpy.default_rng(explicit context/step seed); float32 Gaussian initial flow noise',
        'server_log_dir':str(out), 'model_retrained':False}
    # Capture the native sampler output immediately before unchanged unnormalization.
    original_output = policy._output_transform
    capture = {}
    def output_transform(x):
        capture['raw'] = np.array(x['actions'], copy=True)
        return original_output(x)
    policy._output_transform = output_transform
    original_input = policy._input_transform
    def input_transform(x):
        result = original_input(x)
        capture['transformed_hash'] = payload_sha({k: np.asarray(v) for k,v in
            __import__('flax').traverse_util.flatten_dict(result, sep='/').items()})
        return result
    policy._input_transform = input_transform
    write(out/'SERVER_READY.json', metadata)
    seen = set()
    class AuditedPolicy:
        def infer(self, obs):
            audit = obs.pop('_audit')
            request = audit['request_id']; worker = audit['worker_id']
            if request in seen: raise RuntimeError('Duplicate request/replay forbidden')
            digest = payload_sha(obs)
            if digest != audit['observation_sha256']: raise RuntimeError('Observation RPC mismatch')
            if obs['prompt'] != audit['instruction']: raise RuntimeError('Task prompt mismatch')
            seen.add(request)
            started = time.time_ns()
            noise = np.random.default_rng(int(audit['noise_seed'])).standard_normal(metadata['noise_shape']).astype(np.float32)
            result = policy.infer(obs, noise=noise)
            actions = np.asarray(result['actions'])
            raw = capture['raw']
            receipt = {**audit, 'server_pid':os.getpid(), 'server_start_ns':started,
                'server_end_ns':time.time_ns(), 'checkpoint_sha256':inventory['sha256'],
                'model_inference_called':True, 'action_sha256':array_sha(actions),
                'raw_model_action_sha256':array_sha(raw),
                'transformed_observation_sha256':capture['transformed_hash'],
                'noise_sha256':array_sha(noise), 'downstream_action_source':'ONLINE_VLA'}
            with (out/'INFERENCE.jsonl').open('a') as f:
                import json
                f.write(json.dumps(receipt,sort_keys=True)+'\n'); f.flush()
            result['raw_model_actions'] = raw
            result['provenance'] = receipt
            return result
    logging.basicConfig(level=logging.INFO, force=True)
    WebsocketPolicyServer(AuditedPolicy(),host='127.0.0.1',port=args.port,metadata=metadata).serve_forever()

if __name__ == '__main__': main()
