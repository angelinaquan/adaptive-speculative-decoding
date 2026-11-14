"""Prompt-grouped nested cross-validation of frozen pre-draft feature predictors.
Prediction/reward diagnostic on teacher anchors; not serving speed or domain transfer.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,hashlib,itertools
from pathlib import Path
import numpy as np
from allocation_headroom_v73 import allocate
r=Path(__file__).resolve().parent
plan=json.loads((r/'context_v75_plan.json').read_text())
raw=json.loads((r/'labels_v73_checks.json').read_text());assert raw['passed']
assert hashlib.sha256((r/'labels_v73_features.pt').read_bytes()).hexdigest()==raw['feature_sha256']
tasks=json.loads((r/'oracle_v57_manifest.json').read_text());assert len({x['text'].strip() for x in tasks})==48
pack=r/'context_v75_features.npz'
if not pack.exists():
 import torch
 feats=torch.load(r/'labels_v73_features.pt',map_location='cpu',weights_only=True)
 contexts=[];embeddings=[];ids=[];offsets=[];labels=[];remaining=[]
 byid={x['task_index']:x for x in raw['records']}
 for f in feats:
  record=byid[f['task_index']];n=len(record['anchors'])
  c=f['context'].reshape(n,-1).numpy();e=f['anchor_embedding'].reshape(n,-1).numpy()
  assert f['anchors'].tolist()==record['anchors'] and c.shape==e.shape==(n,2560)
  contexts.extend(c);embeddings.extend(e)
  for j,a in enumerate(record['anchors']):
   offset=a-len(record['input_ids']);ids.append(f['task_index']);offsets.append(offset);remaining.append(len(record['output_ids'])-offset-1)
   labels.append([record['widths'][str(w)]['accepted'][j]+1 for w in range(1,9)])
 np.savez_compressed(pack,context=contexts,anchor=embeddings,ids=ids,offsets=offsets,labels=labels,remaining=remaining)
z=np.load(pack);ids=z['ids'];offsets=z['offsets'];Y=z['labels'].astype(float);remaining=z['remaining'];assert Y.shape==(277,8)
# Reconstruct labels independently from raw records to audit exported arrays.
for k,(i,offset) in enumerate(zip(ids,offsets)):
 record=raw['records'][int(i)];j=record['anchors'].index(len(record['input_ids'])+int(offset))
 assert Y[k].tolist()==[record['widths'][str(w)]['accepted'][j]+1 for w in range(1,9)]
 assert remaining[k]==len(record['output_ids'])-offset-1

def norm(x):
 x=x.astype(np.float64);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
features={'anchor':norm(z['anchor']),'context_anchor':np.concatenate([norm(z['context']),norm(z['anchor'])],axis=1)/np.sqrt(2)}

def predict(K,train,test,y,lam):
 mean=y[train].mean(axis=0)
 coef=np.linalg.solve(K[np.ix_(train,train)]+lam*np.eye(len(train)),y[train]-mean)
 return np.clip(mean+K[np.ix_(test,train)]@coef,1,np.arange(2,10))

def fit_outer(K,fold,y):
 train=np.flatnonzero(ids%4!=fold);test=np.flatnonzero(ids%4==fold)
 assert not set(ids[train])&set(ids[test])
 losses=[]
 for lam in plan['ridge_penalties']:
  loss=0.;count=0
  for inner in range(3):
   tr=train[ids[train]%3!=inner];va=train[ids[train]%3==inner]
   assert len(tr) and len(va) and not set(ids[tr])&set(ids[va])
   p=predict(K,tr,va,y,lam);loss+=float(((p-y[va])**2).sum());count+=p.size
  losses.append(loss/count)
 selected=int(np.argmin(losses));lam=plan['ridge_penalties'][selected]
 return train,test,predict(K,train,test,y,lam),{'penalty':lam,'inner_mse':losses}

predictions={m:np.zeros_like(Y) for m in ['mean','anchor','context_anchor']};fits=[]
for model,X in features.items():
 K=X@X.T
 for fold in range(4):
  train,test,p,fit=fit_outer(K,fold,Y)
  poisoned=Y.copy();poisoned[test]=np.random.default_rng(18657+fold).uniform(1,np.arange(2,10),size=(len(test),8))
  _,_,q,other=fit_outer(K,fold,poisoned)
  assert fit==other and np.array_equal(p,q),'Held-out labels influenced fitting or predictions'
  predictions[model][test]=p
  predictions['mean'][test]=Y[train].mean(axis=0)
  fits.append({'model':model,'fold':fold,'train_prompt_ids':sorted(map(int,set(ids[train]))),'test_prompt_ids':sorted(map(int,set(ids[test]))),**fit})

records=[]
# Score each held-out fold separately at each offset: no training states enter allocations.
for fold in range(4):
 for offset in sorted(set(offsets)):
  selected=np.flatnonzero((ids%4==fold)&(offsets==offset));n=len(selected)
  if not n:continue
  actual=np.minimum(Y[selected],remaining[selected,None])
  fixed=float(actual[:,2].sum());oracle=allocate(actual.tolist(),3*n)
  row={'fold':fold,'offset':int(offset),'states':n,'prompt_ids':ids[selected].tolist(),'fixed3':fixed,'oracle':oracle['reward'],'methods':{}}
  for model,preds in predictions.items():
   choice=allocate(preds[selected].tolist(),3*n);widths=np.array(choice['widths'])
   reward=float(actual[np.arange(n),widths-1].sum())
   row['methods'][model]={'reward':reward,'widths':widths.tolist(),'budget_used':int(widths.sum())}
  records.append(row)
fixed=sum(x['fixed3'] for x in records)
summary={'scope':__doc__,'passed':True,'n_states':len(ids),'prompts_with_states':len(set(ids)),'fixed3_reward':fixed,'oracle_reward':sum(x['oracle'] for x in records),'models':{},'fits':fits,'gates':['All folds disjoint byprompt; no duplicateprompttext','Each outertest-label poisoning leavespenalty andpredictions unchanged','Allrawlabels reconstructedfromchecked v73records','Budgeted allocation withinheldoutfold only'],'input_sha256':hashlib.sha256((r/'labels_v73_checks.json').read_bytes()).hexdigest(),'features_sha256':hashlib.sha256(pack.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((r/'context_v75_plan.json').read_bytes()).hexdigest(),'limitations':plan['limitations']}
for m,p in predictions.items():
 reward=sum(x['methods'][m]['reward'] for x in records)
 summary['models'][m]={'reward':reward,'gain_vs_fixed3_percent':100*(reward/fixed-1),'mse':float(((p-Y)**2).mean())}
np.savez_compressed(r/'context_v75_predictions.npz',ids=ids,offsets=offsets,labels=Y,remaining=remaining,**predictions)
(r/'context_v75_records.json').write_text(json.dumps(records,indent=2)+'\n');(r/'context_v75_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='fits'},indent=2))
