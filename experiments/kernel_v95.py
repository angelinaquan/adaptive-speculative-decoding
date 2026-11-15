"""Nonlinear-feature feasibility audit on existing v83 labels; not a novel method."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np
from correlated_v84 import normalize,replay

def score(pred,Y,ids,offsets):
 rows=replay(pred,ids,offsets,lambda i,w:Y[i,w],np.zeros((8,8)),np.ones(8),0,1)
 return sum(x['reward'] for x in rows)

def run():
 r=Path(__file__).resolve().parent;z=np.load(r/'labels_v83_features.npz');audit=json.loads((r/'labels_v83_summary.json').read_text());assert audit['passed'];assert hashlib.sha256((r/'labels_v83_features.npz').read_bytes()).hexdigest()==audit['feature_npz_sha256']
 ids=z['ids'];offsets=z['offsets'];Y=np.minimum(z['labels'],z['remaining'][:,None]).astype(float);X=np.concatenate([normalize(z['context']),normalize(z['anchor'])],axis=1)/np.sqrt(2)
 tr=np.flatnonzero(z['splits']=='train');va=np.flatnonzero(z['splits']=='validation');assert not set(ids[tr])&set(ids[va]);a=X[tr];b=X[va];yt=Y[tr];yv=Y[va]
 dot=a@a.T;dist=np.maximum(0,2-2*dot);bandwidth=float(np.median(dist[np.triu_indices(len(a),1)]));assert bandwidth>0
 results=[]
 for name,K,Kv in [('linear',dot,b@a.T),('rbf',np.exp(-dist/bandwidth),np.exp(-np.maximum(0,2-2*b@a.T)/bandwidth))]:
  candidates=[]
  for lam in [.01,.1,1.]:
   oof=np.zeros_like(yt)
   for f in range(3):
    it=np.flatnonzero(ids[tr]%3!=f);iv=np.flatnonzero(ids[tr]%3==f);mu=yt[it].mean(0);coef=np.linalg.solve(K[np.ix_(it,it)]+lam*np.eye(len(it)),yt[it]-mu);oof[iv]=np.clip(mu+K[np.ix_(iv,it)]@coef,1,np.arange(2,10))
   candidates.append({'lambda':lam,'oof_reward':score(oof,yt,ids[tr],offsets[tr]),'oof_mse':float(np.mean((oof-yt)**2))})
  choice=max(candidates,key=lambda row:row['oof_reward']);lam=choice['lambda'];mu=yt.mean(0);coef=np.linalg.solve(K+lam*np.eye(len(a)),yt-mu);pred=np.clip(mu+Kv@coef,1,np.arange(2,10));value=score(pred,yv,ids[va],offsets[va]);base=float(yv[:,2].sum())
  results.append({'model':name,'candidates':candidates,'selected_lambda':lam,'reward':value,'fixed3_reward':base,'gain_vs_fixed3_percent':100*(value/base-1)})
 out={'scope':__doc__,'training_states':len(tr),'screen_states':len(va),'training_only_median_squared_distance':bandwidth,'results':results,'limitations':['Validation outcomes already exposed in v84; exploratory only','Median bandwidth uses unlabeled training features across folds; not strictly inductive nested CV','Same penalties and decision-objective selection for linear and RBF','Teacher-anchor reward, not throughput; no serving or novelty claim']};(r/'kernel_v95_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
