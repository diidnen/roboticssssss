"""Pre-action feature contract; no simulator import and no branch label access.

An artifact is frozen while the live action clock is at DECISION_STATE. Branch
labels can be attached later, but may never replace this artifact's features.
Scene snapshots alone are NOT certificates of full PhysX/controller restoration.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from activeforcing_feasibility_features import nominal_input


class ContractError(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class DecisionState:
    context_id: str
    task: int
    step: int
    snapshot_path: str
    snapshot_sha256: str
    execution_contract_hash: str
    feature_builder_hash: str
    state: tuple
    mask: tuple
    commands: tuple
    phases: tuple
    posterior_support: tuple
    posterior_weights: tuple
    observation_source: str = 'POST_PROBE_PRE_CANDIDATE_RETURNED_OBSERVATION'

    def input(self, force, mu):
        if not 3.0 <= force <= 5.0:
            raise ContractError('Candidate outside frozen [3,5] setpoint domain')
        if sha256(self.snapshot_path) != self.snapshot_sha256:
            raise ContractError('Snapshot changed after decision')
        if sha256(Path(__file__).with_name('activeforcing_feasibility_features.py')) != self.feature_builder_hash:
            raise ContractError('Feature builder changed after decision')
        return nominal_input(self.commands, self.phases, self.task, force, mu,
                             self.state, self.mask)

    def candidate_parity(self, forces=(3., 4., 5.)):
        # Mu is fixed within each comparison; posterior marginalization is external.
        errors = []
        for mu in self.posterior_support:
            xs = [self.input(f, mu) for f in forces]
            for f, x in zip(forces, xs):
                expected = xs[0].copy()
                expected[:, 17] = np.float32(f / 8.)
                errors.append(float(np.max(np.abs(x - expected))))
        if not errors or max(errors) != 0.:
            raise ContractError('Candidate contaminated non-candidate features')
        return {'CANDIDATE_PREACTION_STATE_EQUALITY': True,
                'TEMPORAL_FEATURE_PARITY': True, 'MAX_FEATURE_DIFF': max(errors),
                'changed_feature_indices': [17], 'candidate_actions_before_extraction': 0}


class DecisionClock:
    """Use around the actual env.step, including probe and stabilization steps."""
    def __init__(self):
        self.step = 0
        self.phase = 'PROBE'
        self.decision_step = None
        self.candidate_actions = 0
        self.artifact_frozen = False

    def stepped(self):
        self.step += 1
        if self.phase == 'EXECUTION':
            self.candidate_actions += 1
        elif self.phase == 'DECISION_STATE':
            raise ContractError('Simulation stepped while hypothetical candidates were evaluated')

    def finish_probe(self):
        if self.phase != 'PROBE':
            raise ContractError('Probe termination out of order')
        self.phase = 'DECISION_STATE'
        self.decision_step = self.step

    def extract(self, *, context_id, task, snapshot_path, execution_contract_hash,
                state, mask, commands, phases, posterior_support, posterior_weights):
        if self.phase != 'DECISION_STATE' or self.candidate_actions or self.step != self.decision_step:
            raise ContractError('Feature extraction must precede ALL candidate actions')
        support = np.asarray(posterior_support, float)
        weights = np.asarray(posterior_weights, float)
        if support.ndim != 1 or support.size == 0 or support.shape != weights.shape or not np.isfinite(support).all() or np.any(support <= 0):
            raise ContractError('Invalid posterior support')
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.):
            raise ContractError('Invalid posterior weights')
        artifact = DecisionState(context_id, task, self.step, str(snapshot_path),
            sha256(snapshot_path), execution_contract_hash,
            sha256(Path(__file__).with_name('activeforcing_feasibility_features.py')),
            tuple(map(float, state)), tuple(map(float, mask)),
            tuple(tuple(map(float, c)) for c in commands), tuple(phases),
            tuple(support), tuple(weights))
        artifact.candidate_parity()
        self.artifact_frozen = True
        return artifact

    def begin_execution(self):
        if self.phase != 'DECISION_STATE' or not self.artifact_frozen:
            raise ContractError('Cannot execute before decision is frozen')
        self.phase = 'EXECUTION'


def collection_gate(*, probe_validated, posterior_validated, decision_parity,
                    execution_restore_validated, empirical_boundary_validated):
    checks = locals().copy()
    return {'ready': all(v is True for v in checks.values()),
            'blockers': [k for k, v in checks.items() if v is not True]}


def matched_training_input(decision, force, mu, label_record):
    """Join ONLY freshly executed labels to an immutable pre-action artifact.

    The outcome record is not an input to nominal_input. No branch state,
    force readback or outcome information may be used to construct X.
    """
    if label_record.get('label_source') != 'CURRENT_MATCHED_FULL_TASK_BRANCH':
        raise ContractError('Historical, reconstructed or ambiguous labels forbidden')
    if label_record.get('context_id') != decision.context_id:
        raise ContractError('Context mismatch')
    for key, value in [('decision_snapshot_sha256',decision.snapshot_sha256),
                       ('execution_contract_hash',decision.execution_contract_hash),
                       ('candidate_F',force)]:
        if label_record.get(key) != value:
            raise ContractError('Branch/decision join mismatch: '+key)
    if label_record.get('restore_execution_verified') is not True:
        raise ContractError('Unverified full execution-state restore')
    labels = [label_record.get(k) for k in ('lift_success','place_success','dropped')]
    if any(type(v) not in (int,bool) or v not in (0,1) for v in labels):
        raise ContractError('Missing or invalid full-task outcome components')
    lift, place, dropped = labels
    y = int(lift and place and not dropped)
    if label_record.get('full_task_success_y') != y:
        raise ContractError('Wrong full_task_success_y definition')
    return decision.input(force, mu), y
