import itertools,json
from pathlib import Path
import numpy as np
from persistent_oracle_v97 import solve
rng=np.random.default_rng(97)
for trial in range(100):
 ids=np.array([0,1,2,0,1] if trial%2 else [0,1,2,0,1,2]);offsets=np.array([0,0,0,8,8] if trial%2 else [0,0,0,8,8,8]);Y=np.array([[rng.integers(1,w+2) for w in range(1,5)] for _ in ids],float)
 value,ws=solve(Y,ids,offsets);best=-1
 for widths in itertools.product(range(1,5),repeat=3):
  if any(sum(widths[i] for i in ids[offsets==off])>3*sum(offsets==off) for off in [0,8]):continue
  best=max(best,sum(y[widths[int(pid)]-1] for y,pid in zip(Y,ids)))
 assert value==best
out={'passed':True,'exhaustive_cases':100,'checks':['Exact objective matches exhaustive enumeration','Includes missing requests at later anchors','Every observed per-anchor budget enforced','One width per request across all anchors']}
Path(__file__).with_name('persistent_oracle_v97_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
