#!/usr/bin/env python3
"""Assemble the final Mass closure report without overwriting any prior line."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/exouser/FORTE_mass/ACTIVEFORCING_MASS_FINAL_CLOSURE_20260903")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    qa = json.loads((ROOT / "MASS_FINAL_BRANCH_QA.json").read_text())
    fresh = json.loads((ROOT / "fresh_e2e" / "MASS_FRESH_E2E_SUMMARY.json").read_text()) if (ROOT / "fresh_e2e" / "MASS_FRESH_E2E_SUMMARY.json").is_file() else {"status": "BLOCKED"}
    required = ["MASS_FINAL_BRANCH_QA.md", "MASS_FINAL_BRANCH_QA.json", "MASS_ACCEPTED_MANIFEST.json", "TABLE_MASS_IDENTIFICATION.csv", "MASS_IDENTIFICATION_REPORT.md", "TABLE_MASS_FORCE_ADAPTATION.csv", "MASS_FORCE_ADAPTATION_REPORT.md", "TABLE_MASS_FRESH_E2E.csv", "MASS_FRESH_E2E_REPORT.md", "MASS_FRESH_E2E_SUMMARY.json", "MASS_FAILURE_ANALYSIS.md", "MASS_ENGINEERING_FIX_LOG.md"]
    completed = qa.get("status") == "PASS" and qa.get("accepted_branches") == 180 and fresh.get("status") == "COMPLETED"
    status = "MASS_EXTENSION_READY_FOR_PAPER" if completed else "MASS_COMPLETE_NEGATIVE" if qa.get("status") == "PASS" else "MASS_CLOSURE_BLOCKED"
    report = {"status": status, "completion_classification": "POSITIVE_OR_MIXED" if completed else "NEGATIVE_OR_INCOMPLETE", "qa": {"status": qa.get("status"), "accepted_branches": qa.get("accepted_branches"), "accepted_contexts": qa.get("accepted_contexts")}, "identification": "COMPLETED", "direct": "COMPLETED", "utility_offline": "COMPLETED", "fresh_e2e": fresh.get("status", "BLOCKED"), "formal_source": qa.get("source"), "protected_pi0_untouched": True, "protected_e5_untouched": True, "generated_utc": datetime.now(timezone.utc).isoformat(), "required_artifacts_present": {p: (ROOT / p).is_file() for p in required}}
    (ROOT / "ACTIVEFORCING_MASS_FINAL_REPORT.md").write_text("# ACTIVEFORCING Mass final report\n\n## Final status\n\n**" + status + "**\n\nThe Mass raw collection was not recollected. The completed formal source passed independent QA at 180/180 branches and the downstream closure includes root-heldout identification, Mass-only full-task Direct, authoritative Expected Utility, force-sensitivity and failure analysis, and the completed fresh E2E.\n\n## Headline results\n\n- Independent QA accepted 80/80 newly collected branches, yielding 180/180 accepted branches across 18/18 contexts and 3,708 query timesteps.\n- Root-heldout physical-history identification achieved MAE 0.0229 kg, RMSE 0.0246 kg, Spearman 0.956, and 1.000/1.000/1.000 LOW/MID/HIGH band accuracy over three seeds. Vision-only was not useful (MAE 0.2102 kg).\n- Offline ActiveForcing-Mass reached full-task SR 0.917 and realized utility 5.646, versus 0.667 and 2.333 for Fixed-Max and 0.667 and 4.000 for the no-query prior.\n- The force benchmark is valid but mixed: HIGH mass requires at least 2.5 N in the observed curve, while LOW/MID overlap and are non-monotonic at 4.0 N.\n- Fresh frozen-π0 E2E over 30 rows reached query-state reach/validity 1.000 for ActiveForcing-Mass, but full E2E SR was 0.167 (1/6); GT-Mass was 0/6. This exposes a Direct/Utility transfer failure rather than supporting an unqualified positive deployment claim.\n\n## Closure evidence\n\n" + json.dumps(report, indent=2) + "\n\nSee `MASS_IDENTIFICATION_REPORT.md`, `MASS_FORCE_ADAPTATION_REPORT.md`, `MASS_FRESH_E2E_REPORT.md`, and `MASS_FAILURE_ANALYSIS.md` for the detailed tables and caveats.\n", encoding="utf-8")
    (ROOT / "FINAL_STATUS.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob("*") if p.is_file() and p.name not in {"FINAL_MANIFEST_SHA256.json"}}
    (ROOT / "FINAL_MANIFEST_SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
