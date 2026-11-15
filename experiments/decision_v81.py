import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np
from allocation_headroom_v73 import allocate
r=Path(__file__).resolve().parent;plan=json.loads((r/'decision_v81_plan.json').read_text());z=np.load(r/'context_v75_features.npz');ids=z['ids'];offsets=z['offsets'];Y=np.minimum(z['labels'],z['remaining'][:,None]);assert Y.shape==(277,8)
old=json.loads((r/'context_v75_summary.json').read_text());assert hashlib.sha256((r/'context_v75_features.npz').read_bytes()).hexdigest()==old['features_sha256']
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
features={'anchor':norm(z['anchor']),'context_anchor':np.concatenate([norm(z['context']),norm(z['anchor'])],axis=1)/np.sqrt(2)}
def predict(K,tr,te,y,lam):
 mu=y[tr].mean(0);coef=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),y[tr]-mu);return np.clip(mu+K[np.ix_(te,tr)]@coef,1,np.arange(2,10))
def reward(pred,rows,y,details=False):
 out=[]
 for offset in sorted(set(offsets[rows])):
  loc=np.flatnonzero(offsets[rows]==offset);ix=rows[loc];widths=np.array(allocate(pred[loc].tolist(),3*len(ix))['widths']);actual=float(y[ix,widths-1].sum());out.append({'offset':int(offset),'indices':ix.tolist(),'widths':widths.tolist(),'reward':actual})
 return out if details else sum(x['reward'] for x in out)
fits=[];records=[];models={};allpred={}
for feat,X in features.items():
 dot=X@X.T;distance=np.maximum(0,np.diag(dot)[:,None]+np.diag(dot)[None,:]-2*dot);kernels={str(g):dot if g=='linear' else np.exp(-g*distance) for g in plan['kernels']}
 for objective in ['mse','decision']:
  name=feat+'_'+objective;preds=np.zeros_like(Y,dtype=float)
  for fold in range(4):
   tr=np.flatnonzero(ids%4!=fold);te=np.flatnonzero(ids%4==fold);assert not set(ids[tr])&set(ids[te])
   def fit(y):
    candidates=[]
    for key,K in kernels.items():
     for lam in plan['ridge_penalties']:
      sq=0.;score=0.;n=0
      for inner in range(3):
       a=tr[ids[tr]%3!=inner];b=tr[ids[tr]%3==inner];assert not set(ids[a])&set(ids[b]);p=predict(K,a,b,y,lam);sq+=((p-y[b])**2).sum();n+=p.size;score+=reward(p,b,y)
      candidates.append({'kernel':key,'lambda':lam,'mse':float(sq/n),'reward':score})
    best=min(candidates,key=lambda c:c['mse'] if objective=='mse' else -c['reward']);return best,predict(kernels[best['kernel']],tr,te,y,best['lambda'])
   best,p=fit(Y);poison=Y.copy();poison[te]=np.random.default_rng(fold).uniform(1,8,(len(te),8));other,q=fit(poison);assert best==other and np.array_equal(p,q)
   preds[te]=p;rows=reward(p,te,Y,True)
   records.extend([dict(model=name,fold=fold,**row) for row in rows]);fits.append(dict(model=name,fold=fold,**best))
  total=sum(row['reward'] for row in records if row['model']==name);models[name]={'reward':total,'gain_vs_fixed3_percent':100*(total/Y[:,2].sum()-1),'mse':float(((preds-Y)**2).mean())};allpred[name]=preds
fixed=float(Y[:,2].sum());oracle=0
for fold in range(4):
 for offset in sorted(set(offsets)):
  ix=np.flatnonzero((ids%4==fold)&(offsets==offset))
  if len(ix):oracle+=allocate(Y[ix].tolist(),3*len(ix))['reward']
summary={'passed':True,'fixed3_reward':fixed,'oracle_reward':oracle,'models':models,'fits':fits,'scope':plan['candidate'],'limitations':plan['limitations'],'outer_label_poison_passed':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((r/'decision_v81_plan.json').read_bytes()).hexdigest()};(r/'decision_v81_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(r/'decision_v81_records.json').write_text(json.dumps(records,indent=2)+'\n');np.savez_compressed(r/'decision_v81_predictions.npz',**allpred);print(json.dumps({k:v for k,v in summary.items() if k!='fits'},indent=2))
