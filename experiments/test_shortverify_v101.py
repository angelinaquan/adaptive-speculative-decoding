import json,itertools
from pathlib import Path
import numpy as np
from shortverify_v101 import allocate_zero,prepare,predictions,replay
rng=np.random.default_rng(101)
for _ in range(200):
 q=rng.uniform(0,1,(3,3));curves=1+np.cumsum(np.cumprod(q,axis=1),axis=1);budget=int(rng.integers(0,10));ws=allocate_zero(curves,budget)['widths'];got=sum(1 if w==0 else curves[i,w-1] for i,w in enumerate(ws));best=max(sum(1 if w==0 else curves[i,w-1] for i,w in enumerate(widths)) for widths in itertools.product(range(4),repeat=3) if sum(widths)<=budget);np.testing.assert_allclose(got,best,atol=1e-12)
r=Path(__file__).resolve().parent;rows=prepare(json.loads((r/'score_v90_checks.json').read_text()));pred=predictions(rows,'raw',2);first=replay(rows,pred)[1]
for x in rows:x['accepted']=3-x['accepted'];x['native3']=0
assert replay(rows,pred)[1]==first
out={'passed':True,'exhaustive_zero_to_three_allocation_cases':200,'screen_path_reconstruction_states':len(rows),'checks':['Native proposal reconstruction','Exact optimal allocation for concave prefix reward including zero verification','Reward poisoning leaves selected widths unchanged'],'scope':'Replay audit only; no serving speed claim'};(r/'shortverify_v101_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
