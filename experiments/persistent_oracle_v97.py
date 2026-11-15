"""Exact hindsight upper bound for one fixed width per request in each fixed group."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint

def solve(Y,ids,offsets):
 pids=np.unique(ids);n=len(pids);W=Y.shape[1];c=np.concatenate([-Y[ids==pid].sum(0) for pid in pids]);A=[];lo=[];hi=[]
 for j in range(n):
  row=np.zeros(n*W);row[j*W:(j+1)*W]=1;A.append(row);lo.append(1);hi.append(1)
 for off in np.unique(offsets):
  row=np.zeros(n*W);available=set(ids[offsets==off])
  for j,pid in enumerate(pids):
   if pid in available:row[j*W:(j+1)*W]=np.arange(1,W+1)
  A.append(row);lo.append(0);hi.append(3*len(available))
 result=milp(c,integrality=np.ones(n*W),bounds=Bounds(np.zeros(n*W),np.ones(n*W)),constraints=LinearConstraint(np.array(A),lo,hi),options={'mip_rel_gap':0.})
 assert result.success and result.status==0
 x=np.rint(result.x);assert np.max(np.abs(x-result.x))<1e-6
 assert np.all(np.array(A)@x<=np.array(hi)+1e-6) and np.all(np.array(A)@x>=np.array(lo)-1e-6)
 widths={int(pid):int(np.argmax(x[j*W:(j+1)*W]))+1 for j,pid in enumerate(pids)}
 reward=sum(float(y[widths[int(pid)]-1]) for y,pid in zip(Y,ids));np.testing.assert_allclose(reward,-result.fun,atol=1e-6)
 return reward,widths

def run():
 r=Path(__file__).resolve().parent;z=np.load(r/'labels_v83_features.npz');Y=np.minimum(z['labels'],z['remaining'][:,None]);ids=z['ids'];offsets=z['offsets'];out=[]
 for split in ['train','validation']:
  ix=np.flatnonzero(z['splits']==split);pids=np.unique(ids[ix]);value=0;groups=[]
  for start in range(0,len(pids),16):
   group=pids[start:start+16];at=ix[np.isin(ids[ix],group)];v,widths=solve(Y[at],ids[at],offsets[at]);value+=v;groups.append({'reward':v,'widths':widths})
  base=float(Y[ix,2].sum());out.append({'split':split,'fixed3_reward':base,'exact_persistent_oracle_reward':value,'gain_percent':100*(value/base-1),'groups':groups})
 summary={'scope':__doc__,'results':out,'limitations':['Hindsight oracle, not a method','Bound applies only to fixed groups, observed anchors, widths1..8, and per-anchor budget3*n','A state-dependent policy is outside this bound','Reward is not throughput']};(r/'persistent_oracle_v97_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps([{k:v for k,v in row.items() if k!='groups'} for row in out],indent=2))
if __name__=='__main__':run()
