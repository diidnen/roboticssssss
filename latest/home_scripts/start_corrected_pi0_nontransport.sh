#!/usr/bin/env bash
set -euo pipefail
cd /home/exouser/Tabero
exec codex exec --dangerously-bypass-approvals-and-sandbox - < /home/exouser/corrected_pi0_nontransport_prompt.md
