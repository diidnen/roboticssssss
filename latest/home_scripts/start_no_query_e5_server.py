"""Start the archived frozen E5 policy wrapper as a detached local service."""
from pathlib import Path
import importlib.util
import json
import time

SOURCE = Path("/home/exouser/Tabero/analysis/p6g1r1_controller_grasp_vla_handoff.py")
OUT = Path("/home/exouser/no_query_e5_server_20260910_retry2")

spec = importlib.util.spec_from_file_location("p6r1_server", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.B5_RESULTS = Path("/home/exouser/Tabero_old720_80ab/analysis/results/b5_tabero_neutral_20260822_040652")
module.B5_WRAPPER = module.B5_RESULTS / "scripts/b5_serve_policy_with_explicit_norm_stats.py"
module.B5_STUBS = module.B5_RESULTS / "scripts/stubs"

OUT.mkdir(exist_ok=False)
process, owned = module.start_server(OUT, "127.0.0.1", 18881)
if not owned:
    raise RuntimeError("port 18881 is already occupied")
for _ in range(900):
    if module.p6g1.tcp_port_open("127.0.0.1", 18881, timeout_s=0.25):
        (OUT / "SERVER_STARTED.json").write_text(json.dumps({"pid": process.pid, "port": 18881}) + "\n")
        break
    if process.poll() is not None:
        raise RuntimeError(f"server exited with {process.returncode}")
    time.sleep(1)
else:
    process.terminate()
    raise RuntimeError("server did not become ready within 15 minutes")
