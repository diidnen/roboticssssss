"""Original ForcePositionAction squeeze subsystem, not an actuator-cap proxy.

Extract the actual filtering, feed-forward and position-correction statements
unchanged. Native arm execution remains a separate embodiment binding. Target
force F/2 per finger is a REFERENCE, not a per-finger actuator effort limit.
"""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

SOURCE=Path('/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py')
CONFIG=Path('/home/exouser/Tabero/source/tac_manip/tac_manip/tasks/manipulation/libero/config/franka/franka_tactile_libero_env_cfg.py')


def load_subsystem():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ForcePositionAction')
    apply=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply_actions')
    start=next(i for i,n in enumerate(apply.body) if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='alpha')
    end=next(i for i,n in enumerate(apply.body) if isinstance(n,ast.If)
             and ast.unparse(n.test)=='self.cfg.target_contact_squeeze_enabled')
    statements=deepcopy(apply.body[start:end+1])
    if not any(isinstance(n,ast.If) and ast.unparse(n.test)=='self.cfg.squeeze_ff_k_load_z != 0.0' for n in statements):
        raise RuntimeError('Original force feed-forward source drift')
    inner=next(n for n in ast.walk(apply) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.cfg.squeeze_kp != 0.0')
    statements.extend(deepcopy(inner.body))
    # All executable selected nodes remain exact original AST; wrapper only
    # binds scalar measured/reference/aperture tensors and returns the result.
    fn=ast.parse('def original_step(self, f_sq_meas_raw, f_sq_target, d_pred):\n    pass').body[0]
    fn.body=statements+ast.parse('return d_cmd, f_sq_meas, f_sq_target_eff').body
    ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),ns)
    cfg_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ForcePositionActionCfg')
    wanted={'meas_force_filter_alpha','squeeze_ff_k_load_z','squeeze_ff_contact_threshold','target_contact_squeeze_enabled'}
    values={n.target.id:ast.literal_eval(n.value) for n in cfg_class.body if isinstance(n,ast.AnnAssign)
            and isinstance(n.target,ast.Name) and n.target.id in wanted}
    config_tree=ast.parse(CONFIG.read_text(encoding='utf-8'))
    config_cls=next(n for n in config_tree.body if isinstance(n,ast.ClassDef) and n.name=='ForcePositionTactileLiberoCameraEnvCfg')
    call=next(n for n in ast.walk(config_cls) if isinstance(n,ast.Call) and ast.unparse(n.func)=='mdp.ForcePositionActionCfg')
    for keyword in call.keywords:
        if keyword.arg in ('squeeze_kp','squeeze_deadzone'):values[keyword.arg]=ast.literal_eval(keyword.value)
    if values!={'meas_force_filter_alpha':.2,'squeeze_ff_k_load_z':.9,'squeeze_ff_contact_threshold':1.,
                'target_contact_squeeze_enabled':False,'squeeze_kp':.0008,'squeeze_deadzone':.25}:
        raise RuntimeError('Original qualified controller settings changed: '+str(values))
    receipt={'scope':'exact original squeeze subsystem, native arm binding separate',
             'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,CONFIG)},
             'selected_statement_AST_sha256':hashlib.sha256(ast.dump(ast.Module(body=statements,type_ignores=[])).encode()).hexdigest(),
             'config':values,'force_reference_is_not_an_actuator_cap':True}
    return ns['original_step'],values,receipt


class OriginalSqueezeInner:
    def __init__(self):
        self.original_step,values,self.receipt=load_subsystem()
        self.cfg=SimpleNamespace(**values)
        self._env=SimpleNamespace(cfg=SimpleNamespace(gripper_open_val=.04))
        self._robot=SimpleNamespace(set_joint_position_target=self.write_target)
        self._gripper_joint_ids=[0,1]
        self._f_sq_meas_ema=torch.zeros(1)
        self._f_sq_meas_ema_initialized=False
        self.last=None

    def write_target(self,value,joint_ids):
        if joint_ids!=[0,1] or value.shape!=(1,2):raise RuntimeError('Unexpected original finger command layout')
        self.target=value.detach().cpu().numpy()[0].copy()

    def step(self,measured_raw,force,outer_command):
        data=np.asarray([measured_raw,force,outer_command],float)
        if not np.isfinite(data).all() or min(measured_raw,force)<0 or not 0<=outer_command<=.04:
            raise ValueError('Invalid original squeeze subsystem input')
        tensors=[torch.tensor([float(x)],dtype=torch.float32) for x in data]
        with torch.no_grad():command,measured,effective=self.original_step(self,*tensors)
        self.last={'measured_raw_N':float(measured_raw),'measured_filtered_N':float(measured.item()),
                   'force_reference_N':float(force),'effective_reference_N':float(effective.item()),
                   'outer_aperture_m':float(outer_command),'inner_aperture_m':float(command.item())}
        return float(command.item())


if __name__=='__main__':
    inner=OriginalSqueezeInner();rng=np.random.default_rng(913)
    ema=None
    for _ in range(1000):
        raw=float(rng.uniform(0,20));force=float(rng.uniform(3,8));command=float(rng.uniform(0,.04))
        result=inner.step(raw,force,command)
        ema=np.float32(raw) if ema is None else np.float32(np.float32(ema)*np.float32(.8)+np.float32(raw)*np.float32(.2))
        effective=force*1.9 if raw>=1 else force
        delta=.0008*.5*(effective-float(ema))
        expected=np.clip(command-delta if abs(delta)>=.0008*.25 else command,0,.04)
        assert abs(result-expected)<2e-8
        assert abs(inner.last['measured_filtered_N']-float(ema))<4e-6
    print('ORIGINAL_SQUEEZE_INNER_1000_STEP_EQUATION_TEST_PASSED',inner.receipt)
