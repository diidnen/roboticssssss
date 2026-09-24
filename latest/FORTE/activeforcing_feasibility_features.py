"""Canonical offline/runtime feasibility features, matching the training tensor.

P4-B force vectors are already gripper-local. Its gripper_opening field is
the first signed policy joint value, not total aperture. Never manufacture
two finger readings from aggregate probe metrics.
"""
from pathlib import Path
import csv
import re
import numpy as np

PHASES = ('branch_hold', 'lift', 'transit', 'over_basket', 'place', 'release', 'settle')
TASKS = (0, 1, 5, 6)

def rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def nominal_input(commands, phases, task, force, mu, state, mask):
    """P(context,mu,F) tensor; actual state is fixed at the decision time."""
    cmd = np.asarray(commands, dtype=np.float64)
    state = np.asarray(state, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if cmd.shape != (8, 3) or state.shape != (13,) or mask.shape != (13,):
        raise ValueError('Expected eight xyz commands and 13-dimensional state/mask')
    if len(phases) != 8 or any(p not in PHASES for p in phases):
        raise ValueError('Unknown nominal phase; do not silently emit all-zero one-hot')
    if task not in TASKS or not np.isfinite([force, mu]).all():
        raise ValueError('Invalid task or continuous condition')
    phase = np.asarray([[float(p == name) for name in PHASES] for p in phases])
    taskvec = np.zeros((8, len(TASKS))); taskvec[:, TASKS.index(task)] = 1
    repeat = lambda a: np.repeat(np.asarray(a)[None], 8, axis=0)
    x = np.concatenate([cmd-cmd[0], np.vstack([np.zeros((1,3)),np.diff(cmd,axis=0)]),
                        phase,taskvec,repeat([force/8.,mu]),repeat(state),repeat(mask),repeat(state),repeat(mask)],axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite feasibility input')
    return x.astype(np.float32)

def from_saved_probe(context_id, probe_path, command_path, task, force, mu, state_path=None):
    """Use only saved probe/handoff state and the frozen command prefix.

    Missing velocity/right-finger readings are masked, not invented. A saved
    Franka snapshot supplies exact policy q=[joint[-2], -joint[-1]], following
    the actual IsaacLab23 gripper_pos implementation used by this runtime.
    """
    probe_path = Path(probe_path)
    probe = rows(probe_path)
    prefix = rows(command_path)[:8]
    if not probe or len(prefix) != 8:
        raise ValueError('Empty probe or short frozen command prefix')
    last = probe[-1]
    trial = str(last.get('trial_id'))
    repeat_match = re.fullmatch(re.escape(str(context_id))+r'_R([1-9][0-9]*)', trial)
    if trial != str(context_id) and repeat_match is None:
        raise ValueError('Probe context does not match requested context')
    left = np.array([float(last['left_f'+a]) for a in 'xyz'])
    right = np.array([float(last['right_f'+a]) for a in 'xyz'])
    state = np.zeros(13, np.float32); mask = np.zeros(13, np.float32)
    mask[:3] = 1  # relative displacement is zero at the initial reference
    state[6:10] = [abs(left[2]), abs(right[2]), np.linalg.norm(left[:2]), np.linalg.norm(right[:2])]
    mask[6:10] = 1
    opening = float(last['gripper_opening'])
    state[11] = opening; mask[11] = 1
    snapshot = Path(state_path) if state_path else probe_path.with_name(probe_path.stem+'_post_probe_state.pt')
    provenance = {'probe':str(probe_path),'commands':str(command_path),'snapshot':None,
                  'force_mapping':'abs(local_z) per finger; norm(local_xy) per finger',
                  'gripper_mapping':'first joint from probe; second signed joint from matched snapshot or masked',
                  'nominal_phase':'preserved from frozen command prefix','missing_fields':['velocity','right_joint']}
    if snapshot.exists():
        import torch
        saved = torch.load(snapshot, map_location='cpu', weights_only=False)
        if saved.get('context_id') != context_id:
            raise ValueError('Snapshot context mismatch')
        if repeat_match is not None and saved.get('repeat') != int(repeat_match.group(1)):
            raise ValueError('Probe/snapshot repeat mismatch')
        robot = saved['state']['articulation']['robot']
        joints = np.asarray(robot['joint_position']).reshape(-1)
        if len(joints) != 9 or not np.isclose(joints[-2],opening,rtol=0,atol=1e-6):
            raise ValueError('Franka joint layout or probe/snapshot alignment mismatch')
        state[11:13] = [joints[-2], -joints[-1]]; mask[11:13] = 1
        obj = saved['state']['rigid_object'][last['object_id']]
        velocity = np.asarray(obj['root_velocity']).reshape(-1)[:3]
        state[3:6] = velocity; state[10] = np.linalg.norm(velocity[:2])
        mask[3:6] = 1; mask[10] = 1
        provenance.update(snapshot=str(snapshot),missing_fields=[])
    commands = [[float(r['cmd_'+a]) for a in 'xyz'] for r in prefix]
    phases = [r['phase'] for r in prefix]
    return nominal_input(commands,phases,task,force,mu,state,mask),provenance
