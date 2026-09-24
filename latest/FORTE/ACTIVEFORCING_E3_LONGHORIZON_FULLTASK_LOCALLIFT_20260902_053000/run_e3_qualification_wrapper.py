#!/usr/bin/env python3
"""Run the historical B5/OpenPI evaluator with two audited E3 seams.

The historical source is executed in-memory.  E3_OBJECT_NAME supplies the
suite-specific object for telemetry, and E3_FORCE_N sets only the two normal
force slots after the frozen policy produces its unchanged motion chunk.
"""
from __future__ import annotations

import os
from pathlib import Path


SOURCE = Path(
    "/home/exouser/Tabero/analysis/results/b5_tabero_neutral_20260822_040652/"
    "scripts/b5_tabero_neutral_client.py"
)


def transform_source() -> str:
    object_name = os.environ.get("E3_OBJECT_NAME", "").strip()
    force_n = os.environ.get("E3_FORCE_N", "").strip()
    if not object_name:
        raise RuntimeError("E3_OBJECT_NAME is required")
    if not force_n or float(force_n) <= 0:
        raise RuntimeError("positive E3_FORCE_N is required")
    source = SOURCE.read_text(encoding="utf-8")
    old_object = "b5_obj_name = B5_TASK_OBJECTS.get(b5_task_id)"
    new_object = "b5_obj_name = os.environ.get('E3_OBJECT_NAME') or B5_TASK_OBJECTS.get(b5_task_id)"
    old_action = "                action = inference_actions\n"
    new_action = (
        "                action = inference_actions\n"
        "                if action.shape[-1] == 13:\n"
        "                    _e3_force = float(os.environ['E3_FORCE_N']) / 2.0\n"
        "                    action[:, 7:13] = 0.0\n"
        "                    action[:, 9] = _e3_force\n"
        "                    action[:, 12] = _e3_force\n"
    )
    old_goals = "        task_suite_config = json.load(f)\n"
    new_goals = (
        "        task_suite_config = json.load(f)\n"
        "    _e3_task_cfg = next(t for t in task_suite_config['tasks'] if int(t['task_id']) == int(args.task_id))\n"
        "    _e3_goals = list(_e3_task_cfg.get('goals', []))\n"
    )
    old_official = "                            official_now = 0\n"
    new_official = (
        "                            official_now = 0\n"
        "                        try:\n"
        "                            from tac_manip.tasks.manipulation.libero import mdp as _e3_mdp\n"
        "                            _e3_goal_status = [int(bool(_e3_mdp.libero_goals_reached(env, goals=[g])[0].item())) for g in _e3_goals]\n"
        "                        except Exception:\n"
        "                            _e3_goal_status = [0 for _ in _e3_goals]\n"
    )
    old_row = '                            "official_success_now": official_now,\n'
    new_row = (
        '                            "official_success_now": official_now,\n'
        '                            "e3_obj_xy_displacement_m": float(np.linalg.norm((obj_p - b5_obj0_w)[:2])),\n'
        '                            "e3_goal_status_json": json.dumps(_e3_goal_status),\n'
        '                            "e3_relationship_success": max([s for s, g in zip(_e3_goal_status, _e3_goals) if "relationship" in g] or [0]),\n'
        '                            "e3_close_success": max([s for s, g in zip(_e3_goal_status, _e3_goals) if g.get("operation") == "close"] or [0]),\n'
        '                            "e3_turn_success": max([s for s, g in zip(_e3_goal_status, _e3_goals) if g.get("operation") in ("turnon", "turnoff")] or [0]),\n'
    )
    old_helper = "B5_NOMINAL_MAT = None\n"
    new_helper = old_helper + '''\n\ndef _e3_stable_state_hash(value):
    """Hash an Isaac scene state without serializing or mutating it."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            digest.update(b"D")
            for key in sorted(item, key=lambda k: str(k)):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"L")
            for child in item:
                visit(child)
        elif isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()
'''
    old_root = "            b5_obj0_w = _b5_object_pos(env, b5_obj_name)\n"
    new_root = old_root + "            _e3_root_state_hash = _e3_stable_state_hash(env.scene.get_state(is_relative=True))\n"
    old_episode = '                    "initial_force_intent": "",\n'
    new_episode = old_episode + '                    "root_state_hash": _e3_root_state_hash,\n'
    seams = (old_object, old_action, old_goals, old_official, old_row, old_helper, old_root, old_episode)
    if any(source.count(seam) != 1 for seam in seams):
        raise RuntimeError("historical B5 source seam changed; refuse unaudited execution")
    source = source.replace(old_object, new_object)
    source = source.replace(old_action, new_action)
    source = source.replace(old_goals, new_goals)
    source = source.replace(old_official, new_official)
    source = source.replace(old_row, new_row)
    source = source.replace(old_helper, new_helper)
    source = source.replace(old_root, new_root)
    source = source.replace(old_episode, new_episode)
    compile(source, str(SOURCE), "exec")
    return source


def main() -> None:
    source = transform_source()
    # Execute in the real __main__ module namespace.  Tyro resolves postponed
    # dataclass annotations through sys.modules[cls.__module__]; a detached
    # globals dict therefore loses names imported by the historical source
    # (for example typing.Optional) even though ordinary name lookup works.
    module_globals = globals()
    module_globals["__file__"] = str(SOURCE)
    module_globals["__package__"] = None
    exec(compile(source, str(SOURCE), "exec"), module_globals, module_globals)


if __name__ == "__main__":
    main()
