import pathlib, sys, json, torch, mujoco
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
orig=torch.load; torch.load=lambda *a,**kw: orig(*a,weights_only=False,**kw)
suite=benchmark.get_benchmark_dict()['libero_goal']()
for tid in (0,5,7):
 task=suite.get_task(tid); bddl=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
 e=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256); e.reset(); e.set_init_state(suite.get_task_init_states(tid)[0]); m=e.sim.model
 print('\nTASK',tid)
 for i in range(m.ngeom):
  n=m.geom_id2name(i) or ''
  if any(x in n for x in ('wooden_cabinet_1_g18','wooden_cabinet_1_g22','plate_1_g','flat_stove_1_g19','gripper0_finger')):
   print('GEOM',i,n,'friction',m.geom_friction[i].tolist(),'body',m.body_id2name(m.geom_bodyid[i]))
 for i in range(m.njnt):
  n=m.joint_id2name(i) or ''
  if 'flat_stove_1_button' in n: print('JOINT',i,n,'dof',m.jnt_dofadr[i], 'damping',m.dof_damping[m.jnt_dofadr[i]])
 e.close()
