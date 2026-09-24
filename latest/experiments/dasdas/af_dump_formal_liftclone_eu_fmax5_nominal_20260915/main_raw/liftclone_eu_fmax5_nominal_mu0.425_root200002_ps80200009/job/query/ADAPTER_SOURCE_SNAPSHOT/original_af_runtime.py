"""Original transport AF entry point. No replacement model or missing-input fallback.

This entry point intentionally rejects the old RoboTwin 15D/query schema. The
robot-specific acquisition adapter must provide the original observable data
and a qualified task binding before this can execute on dump_bin_bigbin.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

FORTE = Path('/home/exouser/FORTE')
REFERENCE = Path('/media/volume/newdata/exouser/online_vla_activeforcing_20260907/final_continuous_friction_generalization_v1')
BELIEF_MANIFEST = FORTE / 'analysis/results/current_matched_runtime_collection_v5_20260906/BELIEF_MANIFEST.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class OriginalActiveForcing:
    def __init__(self):
        manifest_path = REFERENCE / 'CONTINUOUS_FRICTION_RUNTIME_MANIFEST.json'
        self.manifest = json.loads(manifest_path.read_text())
        for path, expected in self.manifest['source_hashes'].items():
            if sha(path) != expected:
                raise ValueError(f'Original source/checkpoint changed: {path}')
        sys.path.insert(0, str(FORTE))
        sys.path.insert(0, str(REFERENCE / 'SOURCE_SNAPSHOT'))
        from phase_free_feasibility import PhaseFreeFeasibility
        source = FORTE / 'analysis/results/current_multitask58_loader_candidate_20260905/continuous_belief.py'
        spec = importlib.util.spec_from_file_location('original_af_verified_belief', source)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        self.belief = mod.ContinuousBelief(BELIEF_MANIFEST, manifest_sha256=sha(BELIEF_MANIFEST))
        self.feasibility = PhaseFreeFeasibility(device='cpu')

    @staticmethod
    def validate_inputs(raw, readback, feature):
        if not isinstance(raw, list) or not raw:
            raise ValueError('Original RAW_PROBE rows required; 15D RoboTwin query is incompatible')
        if not isinstance(readback, list) or len(readback) != len(raw):
            raise ValueError('Original per-finger CONTACT_PATCH_READBACK rows required')
        required = set(('probe_phase task_id step trial_id contact_state eef_x eef_y eef_z '
            't_s force_target measured_squeeze target_normal_force measured_fn measured_ft ft_over_fn '
            'left_fx left_fy left_fz right_fx right_fy right_fz force_imbalance force_imbalance_ratio '
            'gripper_opening contact_normal_x contact_normal_y contact_normal_z contact_tangent_x '
            'contact_tangent_y contact_tangent_z commanded_tangent_increment_mm accumulated_displacement_mm '
            'marker_motion marker_tangential marker_velocity marker_loading_unloading contact_left '
            'contact_right tactile_ok').split())
        for row in raw:
            missing = required - row.keys()
            if missing:
                raise ValueError(f'Missing original sensor fields: {sorted(missing)}')
        if not isinstance(feature, dict) or feature.get('source') != 'ONLINE_VLA_ACTION_CHUNK':
            raise ValueError('Verified online VLA feature provenance required')
        if feature.get('candidate_actions_executed') != 0:
            raise ValueError('Decision features must precede candidate execution')
        x = np.asarray(feature.get('sequence'), dtype=np.float32)
        if x.shape != (8, 64) or not np.isfinite(x).all():
            raise ValueError('Original phase-free (8,64) motion/state/mask/task feature required')
        return x

    def select(self, raw, readback, feature, *, runtime_manifest_sha256):
        x = self.validate_inputs(raw, readback, feature)
        posterior = self.belief.rows(raw, readback, runtime_manifest_sha256=runtime_manifest_sha256)
        posterior['candidate_actions_executed'] = feature['candidate_actions_executed']
        decision = self.feasibility.select(x, posterior)
        return decision, posterior
