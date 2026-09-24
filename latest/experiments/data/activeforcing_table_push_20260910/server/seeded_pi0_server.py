"""Frozen official pi0 server with episode-scoped private sampler seed."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
OUT=Path('/media/volume/data/exouser/activeforcing_table_push_20260910')
CKPT=Path('/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero')
def main():
 import jax
 from openpi.training import config
 from openpi.policies import policy_config
 from openpi.shared import normalize
 from openpi.serving.websocket_policy_server import WebsocketPolicyServer
 policy=policy_config.create_trained_policy(config.get_config('pi0_libero'),CKPT,norm_stats=normalize.load(CKPT/'assets/physical-intelligence/libero'))
 class Seeded:
  def infer(self,obs):
   private=obs.pop('_activeforcing_episode_start_seed',None) # never reaches transforms/model
   if private is not None:
    policy._rng=jax.random.key(int(private))
   ans=policy.infer(obs); a=np.asarray(ans['actions'])
   with (OUT/'server'/'INFERENCE.jsonl').open('a') as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'episode_seed_reset':None if private is None else int(private),'private_field_stripped':private is not None,'payload_keys':sorted(obs),'action_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'shape':list(a.shape)})+'\n')
   return ans
 ready={'pid':os.getpid(),'checkpoint':str(CKPT),'checkpoint_tree_sha256':'92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103','seed_semantics':'reset policy._rng=jax.random.key(seed) once iff private episode-start field present; field stripped before transforms','port':8011}
 (OUT/'server'/'SERVER_READY.json').write_text(json.dumps(ready,indent=2)+'\n')
 WebsocketPolicyServer(Seeded(),host='127.0.0.1',port=8011,metadata=ready).serve_forever()
if __name__=='__main__':main()
