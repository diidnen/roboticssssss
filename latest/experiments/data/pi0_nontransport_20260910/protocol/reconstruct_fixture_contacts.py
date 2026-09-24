import pathlib, numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
orig=torch.load; torch.load=lambda *a,**kw: orig(*a,weights_only=False,**kw)
suite=benchmark.get_benchmark_dict()['libero_goal'](); tid=5; task=suite.get_task(tid)
e=OffScreenRenderEnv(bddl_file_name=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,camera_heights=256,camera_widths=256); e.reset(); e.set_init_state(suite.get_task_init_states(tid)[0]); m,d=e.sim.model,e.sim.data
counts={}
for a in np.load('/home/exouser/Tabero/analysis/results/pi0_nontransport_20260910/rollouts/task5_init0_native_pi0_libero/actions.npy'):
 e.step(a.tolist())
 for i in range(d.ncon):
  x,y=m.geom_id2name(d.contact[i].geom1) or '',m.geom_id2name(d.contact[i].geom2) or ''
  if ('plate_1' in x and ('flat_stove_1' in y or 'table' in y)) or ('plate_1' in y and ('flat_stove_1' in x or 'table' in x)):
   counts[(x,y)]=counts.get((x,y),0)+1
print(counts)
