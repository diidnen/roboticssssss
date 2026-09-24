"""Quantify full link-state mismatch from an unmodified failure log."""
import json
from pathlib import Path
import sys
import numpy as np
path=Path(sys.argv[1])
prefix='RuntimeError: Native snapshot state mismatch: '
line=next(line for line in path.read_text().splitlines() if line.startswith(prefix))
rows=json.loads(line[len(prefix):])
report=[]
for group in rows:
    if group['key']!='native_links':continue
    for a,b in zip(group['before'],group['restored']):
        if a==b:continue
        report.append({'link':a['name'],**{k:float(np.max(np.abs(np.array(a[k])-b[k])))
                      for k in ('pose','com_linear_velocity','angular_velocity')}})
result={'scope':'engineering restore mismatch','different_links':len(report),'differences':report}
path.with_suffix('.link_differences.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
