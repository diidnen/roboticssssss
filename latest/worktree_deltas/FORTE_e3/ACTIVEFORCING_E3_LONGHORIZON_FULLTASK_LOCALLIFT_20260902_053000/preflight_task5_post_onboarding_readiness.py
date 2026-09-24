#!/usr/bin/env python3
"""CPU-only static QA for the post-onboarding nominal DEV path."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ART = Path(__file__).resolve().parent
TABERO = Path("/media/volume/newdata/exouser/Tabero_e3lh")
OPENPI = Path("/home/exouser/FORTE/agent_lanes/Tabero_VTLA_e3_onboarding_20260902")
EXPECTED = {
    ART / "E3_POST_ONBOARDING_FULLTASK_LOCALLIFT_PROTOCOL.json": "af18dd12cb2a9ceb190c6a375669d923e7d8154fd5844f40335e2a5db823844a",
    ART / "run_e3_reserve_qualification_wrapper.py": "ad5faabc8f0ef9c40d04f1f1fb996ed43a0b0171f4d4c0b7ce07415d0c9e047d",
    TABERO / "analysis/results/b5_tabero_neutral_20260822_040652/scripts/b5_tabero_neutral_client.py": "3b38317efad4ee0b2d9e1bfc4379198bbe0d2eb146665e170070b3ad82ec0a18",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    for path, digest in EXPECTED.items():
        if sha256(path) != digest:
            raise RuntimeError(f"authoritative input hash mismatch: {path}")
    if git_head(TABERO) != "f20281944f9771aa32c65745313a193a10665ad1":
        raise RuntimeError("isolated Tabero commit mismatch")
    if git_head(OPENPI) != "31049447d685cb36ddaeddda4f1d62fec0bc6392":
        raise RuntimeError("isolated OpenPI commit mismatch")
    protocol = json.loads((ART / "E3_POST_ONBOARDING_FULLTASK_LOCALLIFT_PROTOCOL.json").read_text())
    gate = protocol["nominal_dev_gate"]
    if gate["roots"] != [7600, 7601, 7602, 7603, 7604] or gate["force_N"] != 8:
        raise RuntimeError("preregistered nominal DEV roots/force changed")
    if protocol["test_and_outcome_firewall"]["heldout_outcomes_used_for_training_or_model_choice"] is not False:
        raise RuntimeError("heldout firewall changed")

    os.environ.update({
        "E3_OBJECT_NAME": "black_book_1",
        "E3_FORCE_N": "8",
        "E3_POLICY_FINGERPRINT": "a" * 64,
        "E3_CANDIDATE_LOCK_SHA256": "b" * 64,
        "E3_COORDINATOR_DYNAMIC_GATE_SHA256": "c" * 64,
    })
    from run_e3_post_onboarding_nominal_wrapper import transform_source

    transformed = transform_source()
    required_fragments = (
        "_e3_server_meta = client.get_server_metadata()",
        '"e3_policy_fingerprint": os.environ["E3_POLICY_FINGERPRINT"]',
        '"e3_checkpoint_step": 999',
        '"e3_coordinator_dynamic_gate_sha256": os.environ["E3_COORDINATOR_DYNAMIC_GATE_SHA256"]',
        "action[:, 9] = _e3_force",
        "action[:, 12] = _e3_force",
    )
    if any(fragment not in transformed for fragment in required_fragments):
        raise RuntimeError("post-onboarding source transform is incomplete")
    compile(transformed, "post_onboarding_b5_client.py", "exec")

    template = json.loads((ART / "E3_SECOND_SILENT_GATE_TEMPLATE.json").read_text())
    if template.get("authorized") is not False or template.get("active_e5_pids") != [2059513, 2059639]:
        raise RuntimeError("second-silent template must remain non-authorizing and name both protected E5 jobs")
    v2 = json.loads((ART / "E3_COORDINATOR_FINAL_GATE_V2_TEMPLATE.json").read_text())
    if v2.get("authorized") is not False or v2.get("status") != "TEMPLATE_NOT_AUTHORIZED":
        raise RuntimeError("V2 coordinator template must remain non-authorizing")
    override = json.loads((ART / "E3_SECOND_SILENT_GATE_COORDINATOR_OVERRIDE_20260902.json").read_text())
    if override.get("preserve_v1_as_lineage") is not True or override["v2_launch_requirements"]["minimum_silence_seconds"] != 10:
        raise RuntimeError("V2 coordinator override lineage mismatch")
    print("E3_POST_ONBOARDING_READINESS_PREFLIGHT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
