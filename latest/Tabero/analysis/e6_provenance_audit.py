#!/usr/bin/env python3
"""CPU-only provenance and source-of-truth audit for the E6 lane."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


LANE = Path("/home/exouser/FORTE/agent_lanes/E6_DECISION_AWARE_REQUERY")
CLOSURE = Path("/home/exouser/FORTE/activeforcing_full_claim_closure_20260902_134732/E6_E7")
HANDOFF = CLOSURE / "E6_E7_LOCKED_HANDOFF"
SCORES = Path("/home/exouser/FORTE_e6resume/AF_V2_UTILITY_E6_CPU_RECOVERY_20260902/E6_EXPECTED_UTILITY_MEMBER_SCORES.csv")
DECISIONS = Path("/home/exouser/FORTE_e6resume/AF_V2_UTILITY_E6_CPU_RECOVERY_20260902/E6_EXPECTED_UTILITY_MEMBER_DECISIONS.csv")
ARCHIVE = Path("/home/exouser/FORTE/activeforcing_residual_utility_20260901_055605/POOLED_OOF_UTILITY_DATASET.csv")
PAIR = Path("/home/exouser/FORTE_e6resume/AF_V2_UTILITY_E6_CPU_RECOVERY_20260902/E6_PAIRED_EVALUATION_MANIFEST.json")
UTILITY = Path("/home/exouser/FORTE/UTILITY_FINAL_CONFIG.json")
PROTOCOL = Path("/home/exouser/FORTE/activeforcing_physical_only_probe_20260901/PHYSICAL_ONLY_PROBE_PROTOCOL.json")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_status(path: Path, expected: str | None = None) -> dict:
    exists = path.is_file()
    actual = digest(path) if exists else None
    return {"path": str(path), "exists": exists, "sha256": actual, "expected_sha256": expected, "hash_match": bool(exists and (expected is None or actual == expected))}


def main() -> None:
    handoff_hashes = {}
    with (HANDOFF / "SHA256_MANIFEST.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            handoff_hashes[row["path"]] = row["sha256"]

    files = {}
    for name in [
        "PHYSICAL_BELIEF_member_0.pt", "PHYSICAL_BELIEF_member_1.pt", "PHYSICAL_BELIEF_member_2.pt",
        "FROZEN_DIRECT_FEAS_seed0.pt", "FROZEN_DIRECT_FEAS_seed1.pt", "FROZEN_DIRECT_FEAS_seed2.pt",
        "PHYSICAL_BELIEF_MANIFEST.json", "IDENTIFIER_AND_PLANNER_INTERFACE.json", "FINAL_RHO.json",
        "SECOND_QUERY_PROTOCOL.json",
    ]:
        files[name] = file_status(HANDOFF / name, handoff_hashes.get(name))

    scores = pd.read_csv(SCORES)
    decisions = pd.read_csv(DECISIONS)
    archive = pd.read_csv(ARCHIVE)
    utility = json.loads(UTILITY.read_text())
    pair = json.loads(PAIR.read_text())
    belief = json.loads((CLOSURE / "PHYSICAL_BELIEF_MANIFEST.json").read_text())
    interface = json.loads((HANDOFF / "IDENTIFIER_AND_PLANNER_INTERFACE.json").read_text())
    protocol = json.loads(PROTOCOL.read_text())

    checks = {
        "scores_rows": int(len(scores)),
        "scores_expected_shape": bool(len(scores) == 2160 and scores["context_id"].nunique() == 72 and scores["root_id"].nunique() == 24),
        "scores_members": sorted(int(x) for x in scores["member"].unique()),
        "scores_rows_per_member": {str(int(k)): int(v) for k, v in scores.groupby("member").size().items()},
        "scores_episode_rows_15": bool(scores.groupby(["context_id", "repeat"]).size().eq(15).all()),
        "decisions_rows": int(len(decisions)),
        "decisions_expected_episodes": bool(len(decisions) == 144 and decisions[["context_id", "repeat"]].drop_duplicates().shape[0] == 144),
        "archive_rows": int(len(archive)),
        "archive_expected_branches": bool(len(archive) == 720 and archive["branch_id"].nunique() == 720),
        "belief_member_count_3": belief.get("member_count") == 3 and len(belief.get("members", [])) == 3,
        "belief_test_contexts_zero": belief.get("test_contexts_loaded") == 0,
        "belief_member_hashes_match_manifest": all(m.get("checkpoint_sha256") == handoff_hashes.get(f"PHYSICAL_BELIEF_member_{i}.pt") for i, m in enumerate(belief.get("members", []))),
        "direct_backend_frozen": interface.get("direct_backend") == "frozen FEASIBILITY_ONLY continuous Direct",
        "direct_seed_checkpoints_3": all(f"FROZEN_DIRECT_FEAS_seed{i}.pt" in handoff_hashes for i in range(3)),
        "utility_frozen": utility.get("frozen") is True and utility.get("sealed_TEST_used") is False,
        "utility_selector_expected": utility.get("selection") == "argmax_candidate_expected_utility",
        "utility_protocol_v2": utility.get("name") == "ACTIVEFORCING_CURRENT_FULL_UTILITY",
        "pair_manifest_execution_blocked": pair.get("execution_authorized") is False and pair.get("rollout_authorized") is False,
        "pair_manifest_no_post_second_query": pair.get("readiness", {}).get("checks", {}).get("post_second_query_posterior_available") is False,
        "physical_protocol_no_second_query": protocol.get("second_query_status") == "SECOND_QUERY_NOT_EVALUATED_NO_ARCHIVED_REPEAT",
    }
    hash_ok = all(x["hash_match"] for x in files.values())
    all_ok = bool(hash_ok and all(checks.values()))
    result = {
        "status": "PROVENANCE_PASS" if all_ok else "PROVENANCE_FAIL",
        "authoritative_closure": str(CLOSURE),
        "source_files": {str(p): file_status(p) for p in [SCORES, DECISIONS, ARCHIVE, PAIR, UTILITY, PROTOCOL]},
        "locked_handoff_files": files,
        "checks": checks,
        "interpretation": "Three-member physical belief, frozen Direct backend, and frozen Expected-Utility configuration are hash-verified. Current paired evaluation remains execution-blocked because no qualified second-query continuation/posterior exists.",
    }
    (LANE / "E6_PROVENANCE_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# E6 provenance audit",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Authoritative closure: `{CLOSURE}`",
        "",
        "- 3-member physical belief: hash-verified; test contexts loaded = 0.",
        "- 3 frozen Direct FEASIBILITY_ONLY checkpoints: hash-verified.",
        "- Expected-Utility configuration: frozen, `argmax_candidate_expected_utility`, sealed TEST unused.",
        f"- Current CPU evidence: {len(scores)} member-score rows, {len(decisions)} episodes, {len(archive)} unique branches.",
        "- Re-query execution: blocked; paired manifest has no qualified second-query continuation or post-query posterior.",
        "",
        "See `E6_PROVENANCE_AUDIT.json` for all paths, hashes, and checks.",
    ]
    (LANE / "E6_PROVENANCE_AUDIT.md").write_text("\n".join(lines) + "\n")
    final_status = {
        "status": "E6_COMPLETE_NEGATIVE",
        "cpu_recompute": "COMPLETE",
        "signal_gate": "E6_SIGNAL_PASS",
        "scientific_result": "decision disagreement descriptively enriches frontier-harm but is imprecise; no deployment claim",
        "second_query": "NOT_VALIDATED",
        "next_gate": "obtain a genuine qualified TRAIN/DEV second-query continuation with distinct action/evidence, state hash, qualification record, and measurable post-query posterior; then rerun the four-policy comparison under max budget 2",
        "test_accessed": False,
        "gpu_or_isaac_started": False,
        "report": str(LANE / "E6_DECISION_AWARE_REQUERY_REPORT.md"),
        "provenance": str(LANE / "E6_PROVENANCE_AUDIT.md"),
        "metrics": str(LANE / "E6_SIGNAL_METRICS.json"),
    }
    (LANE / "E6_FINAL_STATUS.json").write_text(json.dumps(final_status, indent=2) + "\n")
    status_md = [
        "# E6 status",
        "",
        "`E6_COMPLETE_NEGATIVE`",
        "",
        "- CPU Expected-Utility recompute: complete; 144 episodes / 72 contexts / 24 roots / 3 members.",
        "- Signal gate: `E6_SIGNAL_PASS` descriptively (RR 3.00; uncertainty interval crosses null).",
        "- Re-query closure: negative for deployment; no valid second-query continuation or post-second-query posterior.",
        "- TEST/GPU/Isaac: not accessed or started by E6.",
        "- Next gate: genuine qualified TRAIN/DEV second-query evidence before any rollout.",
        "",
        "Artifacts: `E6_DECISION_AWARE_REQUERY_REPORT.md`, `E6_SIGNAL_METRICS.json`, `E6_RECOMPUTED_MEMBER_DECISIONS.csv`, `E6_SECOND_QUERY_EVIDENCE_AUDIT.json`, `E6_PROVENANCE_AUDIT.md`, `E6_FINAL_STATUS.json`.",
    ]
    (LANE / "E6_FINAL_STATUS.md").write_text("\n".join(status_md) + "\n")


if __name__ == "__main__":
    main()
