import hashlib,json,os,time
from pathlib import Path
import numpy as np
O=Path('/media/volume/data/exouser/activeforcing_table_push_v4_20260910');C=Path('/media/volume/newdata/exouser/pi0_libero_activeforcing_20260910/pi0_libero')
def main():
 import jax
 from openpi.training import config
 from openpi.policies import policy_config
 from openpi.shared import normalize
 from openpi.serving.websocket_policy_server import WebsocketPolicyServer
 p=policy_config.create_trained_policy(config.get_config('pi0_libero'),C,norm_stats=normalize.load(C/'assets/physical-intelligence/libero'))
 class X:
  def infer(self,o):
   s=o.pop('_activeforcing_episode_start_seed',None)
   if s is not None:p._rng=jax.random.key(int(s))
   a=p.infer(o);x=np.asarray(a['actions']);(O/'server'/'INFERENCE.jsonl').open('a').write(json.dumps({'seed':s,'sha256':hashlib.sha256(x.tobytes()).hexdigest(),'shape':list(x.shape)})+'\n');return a
 r={'pid':os.getpid(),'port':8014,'checkpoint':str(C),'tree_sha256':'92b4ac0c5ed929c81677b750fd13b93aa1d30d1fed50fef06a3143dfda9df103','v4_owned':True};(O/'server'/'SERVER_READY.json').write_text(json.dumps(r,indent=2)+'\n');WebsocketPolicyServer(X(),host='127.0.0.1',port=8014,metadata=r).serve_forever()
if __name__=='__main__':main()
