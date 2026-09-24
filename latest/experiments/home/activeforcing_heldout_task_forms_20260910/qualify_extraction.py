"""Frozen Fixed-5 screen plus source-grounded initial-state compatibility."""
from pathlib import Path
P=Path(__file__).resolve().parent
source=(P/'qualify_worker.py').read_text()
needle='    cfg.episode_length_s=60'
assert source.count(needle)==1
source=source.replace(needle,needle+"\n    from extraction_reset_compatibility import repair\n    repair(cfg,a.suite,a.task,job/'RESET_COMPATIBILITY.json')")
exec(compile(source,str(P/'qualify_worker.py')+'::extraction_reset_compatibility','exec'))
