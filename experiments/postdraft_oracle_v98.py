"""Fixed-width8 drafting with variable verification: hindsight feasibility diagnostic."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
from allocation_headroom_v73 import allocate
r=Path(__file__).resolve().parent;z=np.load(r/'labels_v83_features.npz');Y=np.minimum(z['labels'],z['remaining'][:,None]);ids=z['ids'];off=z['offsets'];out=[]
# Prefixes are truncated from the SAME width8 proposal, never substituted with a width-k draft.
R=np.minimum(Y[:,7,None],np.arange(2,10)[None,:]);assert np.all(np.diff(R,axis=1)>=0)
for split in ['train','validation']:
 ix=np.flatnonzero(z['splits']==split);unique=np.unique(ids[ix]);group={int(p):j//16 for j,p in enumerate(unique)};oracle=0
 for offset in np.unique(off[ix]):
  for g in sorted(set(group.values())):
   at=np.array([i for i in ix if off[i]==offset and group[int(ids[i])]==g],int)
   if len(at):oracle+=allocate(R[at].tolist(),3*len(at))['reward']
 native=float(Y[ix,2].sum());fixed=float(R[ix,2].sum());out.append({'split':split,'native_width3_reward':native,'draft8_verify3_reward':fixed,'draft8_adaptive_verification_oracle_reward':oracle,'oracle_gain_vs_native3_percent':100*(oracle/native-1),'fixed_truncation_gain_vs_native3_percent':100*(fixed/native-1)})
result={'scope':__doc__,'results':out,'limitations':['Hindsight oracle only; no method or speedup claim','All arms have same verification-token budget but drafting costs differ','DFlash2 noncausal draft widths differ; width8 truncation labels are not width-k labels','Existing exposed teacher-state data']};(r/'postdraft_oracle_v98_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
