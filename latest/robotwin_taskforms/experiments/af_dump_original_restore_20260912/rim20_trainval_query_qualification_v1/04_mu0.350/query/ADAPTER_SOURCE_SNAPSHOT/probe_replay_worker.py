#!/usr/bin/env python3
"""Read-only instrumentation around original RoboTwin pre-action evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def vector(value):
    return np.asarray(value).tolist()

def pose(value):
    return {'p': vector(value.p), 'q': vector(value.q)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--force', type=float, default=3)
    ap.add_argument('--policy-seed', type=int, default=140200002)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo))
    from envs.dump_bin_bigbin import dump_bin_bigbin
    original = dump_bin_bigbin.activate_activeforcing_candidate_force

    def capture(self):
        # No simulator step, controller write, or state restore in instrumentation.
        actors = self.scene.get_all_actors() if hasattr(self.scene, 'get_all_actors') else []
        articulations = self.scene.get_all_articulations() if hasattr(self.scene, 'get_all_articulations') else []
        state = {'object_pose': pose(self.deskbin.get_pose()),
                 'query_trace': self._af_query_context['trace'], 'actors': [], 'articulations': [],
                 'scene_snapshot_methods': [name for name in dir(self.scene) if 'pack' in name or 'state' in name],
                 'policy_seed': args.policy_seed, 'force_not_yet_activated': args.force}
        for actor in actors:
            rec = {'name': actor.get_name(), 'pose': pose(actor.get_pose())}
            for name in ('get_velocity', 'get_angular_velocity'):
                if hasattr(actor, name): rec[name] = vector(getattr(actor, name)())
            state['actors'].append(rec)
        for art in articulations:
            rec = {'name': art.get_name(), 'pose': pose(art.get_pose())}
            for name in ('get_qpos', 'get_qvel', 'get_qf', 'get_drive_target', 'get_drive_velocity_target'):
                if hasattr(art, name): rec[name] = vector(getattr(art, name)())
            state['articulations'].append(rec)
        # Inspect portable snapshot capability without treating a pose-only snapshot as complete.
        if callable(getattr(self.scene, 'pack', None)):
            packed = self.scene.pack()
            if isinstance(packed, np.ndarray):
                np.save(args.out / 'scene_pack.npy', packed, allow_pickle=False)
                state['scene_pack_sha256'] = hashlib.sha256(packed.tobytes()).hexdigest()
                state['scene_pack_shape'] = list(packed.shape)
        state['physical_state_sha256'] = sha({k: state[k] for k in ('object_pose', 'actors', 'articulations')})
        (args.out / 'handoff_readback.json').write_text(json.dumps(state, indent=2))
        original(self)

    dump_bin_bigbin.activate_activeforcing_candidate_force = capture
    checkpoint = args.repo.parent / 'checkpoints/pi0_robotwin_30000/30000'
    additional = ','.join([f'ckpt_name={checkpoint}', 'action_type=joint', 'activeforcing_enabled=true',
        'af_dynamic_evaluator=false', 'af_supplied_grasp=true', 'af_query_enabled=true', 'af_preaction_only=true',
        'af_query_force_n=4', 'af_query_displacement_m=0.002', 'af_contact_friction=0.55',
        f'af_force_limit_n={args.force:g}', 'start_seed=200002', f'af_policy_seed={args.policy_seed}', 'strict_seed=true'])
    sys.argv = [str(args.repo / 'scripts/eval_policy_xpolicylab.py'), '--task_name', 'dump_bin_bigbin',
        '--env_cfg_type', 'arx_x5', '--policy_name', 'Pi_0', '--host', 'localhost', '--port', '6001',
        '--protocol', 'ws', '--seed', '1', '--test_num', '1', '--expert_check', 'false', '--frequency', '30',
        '--additional_info', additional]
    runpy.run_path(sys.argv[0], run_name='__main__')

if __name__ == '__main__':
    main()
