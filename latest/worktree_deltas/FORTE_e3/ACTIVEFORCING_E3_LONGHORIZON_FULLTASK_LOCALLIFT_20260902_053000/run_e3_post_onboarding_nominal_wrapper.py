#!/usr/bin/env python3
"""Add a locked-policy handshake to the audited E3 qualification wrapper."""

from __future__ import annotations

import os

from run_e3_reserve_qualification_wrapper import transform_source as base_transform_source


def transform_source() -> str:
    fingerprint = os.environ.get("E3_POLICY_FINGERPRINT", "").strip()
    lock_sha = os.environ.get("E3_CANDIDATE_LOCK_SHA256", "").strip()
    gate_sha = os.environ.get("E3_COORDINATOR_DYNAMIC_GATE_SHA256", "").strip()
    if len(fingerprint) != 64 or len(lock_sha) != 64 or len(gate_sha) != 64:
        raise RuntimeError("locked candidate, lock, and second-silent gate hashes are required")
    source = base_transform_source()
    old_client = "    client = _websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)\n"
    new_client = old_client + (
        "    _e3_server_meta = client.get_server_metadata()\n"
        "    _e3_expected_meta = {\n"
        "        'e3_locked_candidate': True,\n"
        "        'e3_task': 'libero_10/task5',\n"
        "        'e3_policy_config': 'pi0_lora_tacfield_e3_task5_5demo_7dpf',\n"
        "        'e3_checkpoint_step': 999,\n"
        "        'e3_checkpoint_tree_sha256': os.environ['E3_POLICY_FINGERPRINT'],\n"
        "        'e3_candidate_lock_sha256': os.environ['E3_CANDIDATE_LOCK_SHA256'],\n"
        "        'e3_openpi_commit': '31049447d685cb36ddaeddda4f1d62fec0bc6392',\n"
        "    }\n"
        "    _e3_bad_meta = {k: (_e3_server_meta.get(k), v) for k, v in _e3_expected_meta.items() if _e3_server_meta.get(k) != v}\n"
        "    if _e3_bad_meta:\n"
        "        raise RuntimeError(f'E3 locked-policy metadata mismatch: {_e3_bad_meta}')\n"
    )
    old_episode = '                    "root_state_hash": _e3_root_state_hash,\n'
    new_episode = old_episode + (
        '                    "e3_policy_fingerprint": os.environ["E3_POLICY_FINGERPRINT"],\n'
        '                    "e3_candidate_lock_sha256": os.environ["E3_CANDIDATE_LOCK_SHA256"],\n'
        '                    "e3_coordinator_dynamic_gate_sha256": os.environ["E3_COORDINATOR_DYNAMIC_GATE_SHA256"],\n'
        '                    "e3_policy_config": "pi0_lora_tacfield_e3_task5_5demo_7dpf",\n'
        '                    "e3_checkpoint_step": 999,\n'
    )
    if source.count(old_client) != 1 or source.count(old_episode) != 1:
        raise RuntimeError("post-onboarding wrapper seam changed; refuse execution")
    source = source.replace(old_client, new_client).replace(old_episode, new_episode)
    compile(source, "post_onboarding_b5_client.py", "exec")
    return source


def main() -> None:
    source = transform_source()
    module_globals = globals()
    module_globals["__file__"] = "post_onboarding_b5_client.py"
    module_globals["__package__"] = None
    exec(compile(source, "post_onboarding_b5_client.py", "exec"), module_globals, module_globals)


if __name__ == "__main__":
    main()
