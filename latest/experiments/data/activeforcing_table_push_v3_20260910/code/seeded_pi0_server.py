"""V3-owned frozen pi0 service; only request seed reset is permitted."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
OUT=Path('/media/volume/data/exouser/activeforcing_table_push_v3_20260910'); CKPT=Path('/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero')
def main():
 from openpi.training import config
 from openpi.policies import policy_config
 from openpi.shared import normalize
 from openpi.serving.websocket_policy_server import WebsocketPolicyServer
 import jax
 p=policy_config.create_trained_policy(config.get_config('pi0_libero'),CKPT,norm_stats=normalize.load(CKPT/'assets/physical-intelligence/libero'))
 class Policy:
  def infer(self,o):
   seed=o.pop('_activeforcing_episode_start_seed',None)
   if seed is not None:p._rng=jax.random.key(int(seed))
   a=p.infer(o); x=np.asarray(a['actions'])
   with (OUT/'server/INFERENCE.jsonl').open('a') as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'seed_reset':seed,'private_stripped':seed is not None,'sha256':hashlib.sha256(x.tobytes()).hexdigest(),'shape':list(x.shape)})+'\n')
   return a
 ready={'pid':os.getpid(),'port':8013,'checkpoint':str(CKPT),'checkpoint_tree_sha256':'92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103','v3_owned':True}
 (OUT/'server/SERVER_READY.json').write_text(json.dumps(ready,indent=2)+'\n'); WebsocketPolicyServer(Policy(),host='127.0.0.1',port=8013,metadata=ready).serve_forever()
if __name__=='__main__':main()
