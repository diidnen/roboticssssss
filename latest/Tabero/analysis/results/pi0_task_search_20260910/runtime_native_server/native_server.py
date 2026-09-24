"""Native Tabero pi0 server for exploratory end-to-end evaluation.

Unlike the prior ActiveForcing server, this accepts the stock OpenPI client
payload (no private `_audit` field) and records one receipt per inference.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np

CHECKPOINT = Path('/media/volume/newdata/exouser/tabero/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999')

def digest(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--port', type=int, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    cfg = config.get_config('pi0_lora_tacfield_tabero')
    # This exact checkpoint carries NathanWu7/tabero normalization assets. The
    # config's repo_id is separately recorded as a metadata discrepancy.
    policy = policy_config.create_trained_policy(cfg, CHECKPOINT, norm_stats=normalize.load(CHECKPOINT/'assets/NathanWu7/tabero'), pytorch_device='cuda:0')
    class NativePolicy:
        def infer(self, obs):
            start = time.time_ns(); result = policy.infer(obs); actions = np.asarray(result['actions'])
            with (a.out/'INFERENCE.jsonl').open('a') as f:
                f.write(json.dumps({'time_ns':start,'checkpoint':str(CHECKPOINT),'policy_config':'pi0_lora_tacfield_tabero','payload_keys':sorted(obs.keys()),'action_sha256':digest(actions),'action_shape':list(actions.shape),'model_inference_called':True})+'\n')
            return result
    (a.out/'SERVER_READY.json').write_text(json.dumps({'pid':os.getpid(),'checkpoint':str(CHECKPOINT),'policy_config':'pi0_lora_tacfield_tabero','mode':'native_stock_openpi_payload','normalization_asset':'assets/NathanWu7/tabero'},indent=2))
    WebsocketPolicyServer(NativePolicy(),host='127.0.0.1',port=a.port).serve_forever()
if __name__ == '__main__': main()
