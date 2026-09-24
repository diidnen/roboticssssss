"""Exact action replay of native successful init-0, with world-frame contact audit."""
import hashlib,json,os,pathlib,sys
import numpy as np, torch, mujoco
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
OUT=pathlib.Path('/media/volume/data/exouser/activeforcing_table_push_20260910')
SOURCE=pathlib.Path('/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/rollouts/task5_init0_native_pi0_libero')
def name(m,i): return m.geom_id2name(i)
def main():
 torch_load=torch.load; torch.load=lambda *a,**kw:torch_load(*a,weights_only=False,**kw)
 suite=benchmark.get_benchmark_dict()['libero_goal'](); task=suite.get_task(5); init=suite.get_task_init_states(5)[0]
 assert task.language=='push the plate to the front of the stove'
 env=OffScreenRenderEnv(bddl_file_name=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,camera_heights=256,camera_widths=256)
 env.seed(7); env.reset(); env.set_init_state(init); acts=np.load(SOURCE/'actions.npy'); m,d=env.sim.model,env.sim.data
 robot={'gripper0_hand_collision','gripper0_finger1_collision','gripper0_finger1_pad_collision','gripper0_finger2_collision','gripper0_finger2_pad_collision'}
 plate={i for i in range(m.ngeom) if str(name(m,i)).startswith('plate_1_')}; samples=[]; allc=[]; pushdir=None; done=False
 for step,a in enumerate(acts):
  # Contact begins at 55 in this exact replay.  Direction uses only its frozen
  # pi0 commands during that causal contact phase, never plate state.
  if step == 55:
   v=np.mean(acts[55:159,:2],axis=0); pushdir=np.array([v[0],v[1],0.]); pushdir/=np.linalg.norm(pushdir)
  _,_,done,_=env.step(a.tolist())
  for ci in range(d.ncon):
   c=d.contact[ci]; n1,n2=name(m,c.geom1),name(m,c.geom2)
   if ((n1 in robot and c.geom2 in plate) or (n2 in robot and c.geom1 in plate)):
    # robosuite exposes wrappers; official MuJoCo API needs their native objects.
    wrench=np.zeros(6);mujoco.mj_contactForce(m._model,d._data,ci,wrench)
    # MuJoCo wrench is force on geom2 in contact frame; frame columns are world axes.
    fw=c.frame.reshape(3,3).T@wrench[:3]
    force_on_plate=fw if c.geom2 in plate else -fw
    rec={'step':step,'contact_index':ci,'geom1':n1,'geom2':n2,'wrench_contact':wrench.tolist(),'frame_world_axes_rows':c.frame.reshape(3,3).tolist(),'force_world_on_plate':force_on_plate.tolist(),'projected_push_n':float(force_on_plate@pushdir)}
    allc.append(rec)
    if len(samples)<3: samples.append(rec)
 by_step={}
 for x in allc: by_step.setdefault(x['step'],0.); by_step[x['step']]+=x['projected_push_n']
 result={'source_actions':str(SOURCE/'actions.npy'),'source_actions_sha256':hashlib.sha256(acts.tobytes()).hexdigest(),'task_id':5,'language':task.language,'init_index':0,'steps_replayed':len(acts),'native_done':bool(done),'push_direction_world':pushdir.tolist(),'push_direction_construction':'normalized mean frozen pi0 action[55:159,:2], selected after stable contact start; no plate state','contact_count':len(allc),'contact_steps':len(by_step),'contacts':allc,'hand_checks':samples,'per_step_projected_force_n':by_step,'sign_rule':'mj_contactForce gives force on geom2; contact frame stores world axes in rows, so world=frame.T@force; negate only when plate is geom1','mean_summed_projected_n':float(np.mean(list(by_step.values()))) if by_step else None}
 (OUT/'FORCE_INTERFACE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n'); env.close()
if __name__=='__main__':main()
