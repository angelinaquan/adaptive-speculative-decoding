"""Post-hoc exhaustive subset sensitivity; not independent replications."""
import json,itertools
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent
assert json.loads((r/'profile_v77_summary.json').read_text())==json.load(open('/tmp/profile_v77_remote_summary.json'))
T=np.zeros((2,8,2,21))
for bi,b in enumerate([8,64]):
 for wi,w in enumerate(range(1,9)):
  for o in range(2):
   x=json.loads((r/f'profile_v77_b{b}_w{w}_r{o}.json').read_text())
   for z in x['records']:
    if z['repeat']>=0:T[bi,wi,o,z['start']//64]+=z['seconds']/2
profiles=[list(range(d*7,d*7+4)) for d in range(3)];tests=[list(range(d*7+5,d*7+7)) for d in range(3)]
rows=[]
# selection/test use disjoint prompt blocks; also swap timing rounds to test temporal stability.
for bi,b in enumerate([8,64]):
 for timing in ['mean','0_to_1','1_to_0']:
  p=T[bi].mean(axis=1) if timing=='mean' else T[bi,:,int(timing[0]),:]
  t=T[bi].mean(axis=1) if timing=='mean' else T[bi,:,int(timing[-1]),:]
  methods={'mixed192':list(itertools.product(*profiles))}
  for d,name in enumerate(['alpaca','gsm8k','mbpp']):methods[name+'192']=list(itertools.combinations(profiles[d],3))
  for method,subsets in methods.items():
   for ix in subsets:
    k=int(np.argmin(p[:,ix].sum(axis=1)))
    losses=[]
    for target in tests+[sum(tests,[])]:
     costs=t[:,target].sum(axis=1);losses.append(float(100*(1-costs.min()/costs[k])))
    rows.append(dict(batch=b,timing=timing,method=method,blocks=list(ix),width=k+1,loss_by_target=losses))
summary=[]
for b in [8,64]:
 for timing in ['mean','0_to_1','1_to_0']:
  for method in ['mixed192','alpaca192','gsm8k192','mbpp192']:
   rs=[z for z in rows if (z['batch'],z['timing'],z['method'])==(b,timing,method)];a=np.array([z['loss_by_target'] for z in rs]);summary.append(dict(batch=b,timing=timing,method=method,n_subsets=len(rs),width_counts={str(k):sum(z['width']==k for z in rs) for k in range(1,9)},mean_loss_by_target=a.mean(0).tolist(),max_loss_by_target=a.max(0).tolist(),within5_by_target=(100*(a<=5).mean(0)).tolist()))
out={'scope':'Post-hoc all64 mixed192 subsets versus all4 single-domain192 subsets; no population probability claim. Timing splits are only two sessions. Targets alpaca/gsm8k/mbpp/pooled.','summary':summary,'records':rows};(r/'profile_v77_robustness.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([x for x in summary if x['timing']=='mean'],indent=2))
