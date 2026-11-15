"""Causal request-specific residual correction of a frozen context predictor.
Teacher-anchor replay only: neither closed-loop decoding nor throughput.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json, hashlib
from pathlib import Path
import numpy as np
from allocation_headroom_v73 import allocate
r=Path(__file__).resolve().parent
z=np.load(r/'context_v75_features.npz'); ids=z['ids']; offsets=z['offsets']
Y=np.minimum(z['labels'],z['remaining'][:,None]).astype(float)
def norm(x):
 x=x.astype(float);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
X=np.concatenate([norm(z['context']),norm(z['anchor'])],axis=1)/np.sqrt(2);K=X@X.T
plan=json.loads((r/'residual_v82_plan.json').read_text())
def fit(train,test,y,lam):
 assert not set(ids[train])&set(ids[test])
 mu=y[train].mean(0)
 return np.clip(mu+K[np.ix_(test,train)]@np.linalg.solve(K[np.ix_(train,train)]+lam*np.eye(len(train)),y[train]-mu),1,np.arange(2,10))
def replay(rows,p,y,alpha,prior,decay):
 # Feedback is read only after every decision at this offset has been made.
 state={}; records=[]
 for offset in sorted(set(offsets[rows])):
  loc=np.flatnonzero(offsets[rows]==offset);ix=rows[loc]
  corrected=p[loc].copy()
  for j,i in enumerate(ix):
   total,count=state.get(int(ids[i]),(0.,0.))
   corrected[j]+=alpha*total/(prior+count)
  corrected=np.clip(corrected,1,np.arange(2,10))
  widths=np.array(allocate(corrected.tolist(),3*len(ix))['widths'])
  assert widths.sum()<=3*len(ix)
  for j,i in enumerate(ix):
   width=int(widths[j]); observed=float(y[i,width-1]);key=int(ids[i]);total,count=state.get(key,(0.,0.))
   records.append({'index':int(i),'prompt':key,'offset':int(offset),'width':width,'reward':observed,'predicted':corrected[j].tolist()})
   state[key]=(decay*total+observed-p[loc[j],width-1],decay*count+1.)
 return records
candidates=[(0.,1.,1.)]+[(a,s,d) for a in [.5,1.] for s in [1.,4.] for d in [.5,1.]]
records=[];fits=[]
for outer in range(4):
 tr=np.flatnonzero(ids%4!=outer);te=np.flatnonzero(ids%4==outer)
 def select(y):
  predictions={}
  for lam in [.01,.1,1.,10.]:
   p=np.zeros_like(y);mse=0.;n=0
   for inner in range(3):
    a=tr[ids[tr]%3!=inner];b=tr[ids[tr]%3==inner];p[b]=fit(a,b,y,lam);mse+=((p[b]-y[b])**2).sum();n+=p[b].size
   predictions[lam]=(mse/n,p)
  lam=min(predictions,key=lambda l:predictions[l][0]);p=predictions[lam][1]
  scores=[]
  for candidate in candidates:
   score=0.
   for inner in range(3):
    b=tr[ids[tr]%3==inner];score+=sum(v['reward'] for v in replay(b,p[b],y,*candidate))
   scores.append(score)
  selected=candidates[int(np.argmax(scores))]
  return lam,selected,scores,fit(tr,te,y,lam)
 lam,selected,scores,p=select(Y)
 poisoned=Y.copy();poisoned[te]=99.
 ll,ss,sc,pp=select(poisoned)
 assert (lam,selected,scores)==(ll,ss,sc) and np.array_equal(p,pp),'Outer labels affected model selection'
 fits.append({'fold':outer,'lambda':lam,'selected':selected,'inner_rewards':scores,'candidates':candidates})
 for name,params in [('frozen',(0.,1.,1.)),('residual',selected)]:
  rr=replay(te,p,Y,*params)
  # Every unobserved counterfactual outcome can change without changing predictions or actions.
  counter=Y.copy();counter[te]=99.
  for row in rr:counter[row['index'],row['width']-1]=row['reward']
  assert replay(te,p,counter,*params)==rr,'Unchosen outcomes affected behavior'
  # Current and future outcomes cannot affect current or earlier predictions/actions.
  for cut in sorted(set(offsets[te])):
   future=Y.copy();future[te[offsets[te]>=cut]]=99.
   other=replay(te,p,future,*params)
   a=[(x['index'],x['width'],x['predicted']) for x in rr if x['offset']<=cut]
   b=[(x['index'],x['width'],x['predicted']) for x in other if x['offset']<=cut]
   assert a==b,'Future feedback affected an earlier decision'
  records.extend([dict(model=name,fold=outer,**row) for row in rr])
fixed=float(Y[:,2].sum());totals={name:sum(x['reward'] for x in records if x['model']==name) for name in ['frozen','residual']}
summary={'passed':True,'fixed3_reward':fixed,'models':{k:{'reward':v,'gain_vs_fixed3_percent':100*(v/fixed-1)} for k,v in totals.items()},'residual_vs_frozen_percent':100*(totals['residual']/totals['frozen']-1),'fits':fits,'checks':['Prompt-disjoint nested folds','Outer-label poisoning','All unchosen labels poisoned','Current/future feedback poisoned at every offset','Draft budget verified'],'scope':__doc__,'limitations':plan['limitations'],'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),r/'residual_v82_plan.json',r/'context_v75_features.npz']}}
(r/'residual_v82_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(r/'residual_v82_records.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(summary,indent=2))
