"""Audited stock-websocket server for the official frozen pi0_libero checkpoint."""
import hashlib, json, os, time
from pathlib import Path
import numpy as np

OUT = Path('/media/volume/data/exouser/pi0_table_friction_20260910')
CHECKPOINT = Path('/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero')
TREE_SHA256 = '92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103'

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()

def main():
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    cfg = config.get_config('pi0_libero')
    policy = policy_config.create_trained_policy(
        cfg, CHECKPOINT, norm_stats=normalize.load(CHECKPOINT/'assets/physical-intelligence/libero'))
    ready = {
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'pid': os.getpid(), 'host': '127.0.0.1', 'port': 8010,
        'policy_config': 'pi0_libero', 'checkpoint_path': str(CHECKPOINT),
        'checkpoint_tree_sha256_expected': TREE_SHA256,
        'normalization_asset': str(CHECKPOINT/'assets/physical-intelligence/libero'),
        'mode': 'frozen official checkpoint; online websocket only',
        'request_rng_behavior': 'stock policy sampler; no server seed API exposed',
        'model_retrained': False,
    }
    (OUT/'server'/'SERVER_READY.json').write_text(json.dumps(ready, indent=2)+'\n')
    class AuditedPolicy:
        def infer(self, observation):
            t0 = time.monotonic(); result = policy.infer(observation); elapsed = time.monotonic()-t0
            actions = np.asarray(result['actions'])
            item = {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),
                    'policy_config':'pi0_libero','checkpoint_path':str(CHECKPOINT),
                    'model_inference_called':True,'payload_keys':sorted(observation.keys()),
                    'action_shape':list(actions.shape),'action_sha256':digest(actions),'infer_seconds':elapsed}
            with (OUT/'server'/'INFERENCE.jsonl').open('a') as f: f.write(json.dumps(item)+'\n')
            return result
    WebsocketPolicyServer(AuditedPolicy(), host='127.0.0.1', port=8010, metadata=ready).serve_forever()

if __name__ == '__main__': main()
