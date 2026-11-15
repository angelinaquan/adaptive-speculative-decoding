"""Diagnostic hindsight and privileged-feedback headroom; not deployable methods."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
from allocation_headroom_v73 import allocate
r=Path(__file__).resolve().parent;z=np.load(r/'labels_v83_features.npz');Y=np.minimum(z['labels'],z['remaining'][:,None]);ids=z['ids'];offsets=z['offsets'];out=[]
for split in ['train','validation']:
 ix=np.flatnonzero(z['splits']==split);unique=np.unique(ids[ix]);groups={int(pid):j//16 for j,pid in enumerate(unique)};prior=Y[z['splits']=='train'].mean(0);history={};totals={k:0. for k in ['fixed3','state_oracle','request_mean_hindsight','previous_full_feedback']};blocks=0
 request_mean={pid:Y[ix[ids[ix]==pid]].mean(0) for pid in unique}
 for offset in np.unique(offsets[ix]):
  for group in sorted(set(groups.values())):
   at=np.array([i for i in ix if offsets[i]==offset and groups[int(ids[i])]==group],int)
   if not len(at):continue
   blocks+=1;totals['fixed3']+=float(Y[at,2].sum())
   for name,pred in [('state_oracle',Y[at]),('request_mean_hindsight',np.array([request_mean[ids[i]] for i in at])),('previous_full_feedback',np.array([history.get(int(ids[i]),prior) for i in at]))]:
    ws=allocate(pred.tolist(),3*len(at))['widths'];assert sum(ws)<=3*len(at);totals[name]+=sum(float(Y[i,w-1]) for i,w in zip(at,ws))
   for i in at:history[int(ids[i])]=Y[i].copy()
 row={'split':split,'states':len(ix),'blocks':blocks,'reward':totals,'gain_vs_fixed3_percent':{k:100*(v/totals['fixed3']-1) for k,v in totals.items()}};out.append(row)
result={'scope':__doc__,'results':out,'limitations':['State oracle reads current counterfactual labels and is infeasible','Request mean reads future labels; diagnostic only, not an upper bound on all context models','Previous-full-feedback reads all widths at each prior anchor, unlike real bandit feedback','No throughput implication: width-dependent draft costs, packing, and policy overhead remain','Existing exposed teacher-state data, not independent evaluation']};(r/'headroom_v97_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
