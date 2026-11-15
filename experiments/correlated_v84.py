"""Width-dependent Bayesian residual correction; validation is never used to fit or select."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np
from allocation_headroom_v73 import allocate

def normalize(x):
 x=np.asarray(x,float);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def covariance(residual,ids):
 # Cross-time products estimate a persistent request effect, excluding same-state noise.
 matrices=[]
 for pid in np.unique(ids):
  e=residual[ids==pid];n=len(e)
  if n>1:matrices.append((np.outer(e.sum(0),e.sum(0))-e.T@e)/(n*(n-1)))
 raw=np.mean(matrices,axis=0) if matrices else np.zeros((8,8))
 vals,vec=np.linalg.eigh((raw+raw.T)/2);B=(vec*np.maximum(vals,0))@vec.T
 noise=np.maximum(np.mean(residual**2,axis=0)-np.diag(B),.05)
 return B,noise

def replay(pred,ids,offsets,feedback,B,noise,scale=0.,decay=1.):
 unique=np.unique(ids);groups={int(pid):j//16 for j,pid in enumerate(unique)};state={};records=[]
 prior=scale*B
 for offset in np.unique(offsets):
  for group in sorted(set(groups.values())):
   ix=np.array([i for i in np.flatnonzero(offsets==offset) if groups[int(ids[i])]==group],dtype=int)
   if not len(ix):continue
   corrected=[];states=[]
   for i in ix:
    m,C=state.get(int(ids[i]),(np.zeros(8),prior.copy()))
    m=decay*m;C=decay**2*C+(1-decay**2)*prior
    states.append((m,C));corrected.append(np.clip(pred[i]+m,1,np.arange(2,10)))
   widths=allocate(np.asarray(corrected).tolist(),3*len(ix))['widths'];assert sum(widths)<=3*len(ix)
   for i,width,(m,C),p in zip(ix,widths,states,corrected):
    w=width-1;observed=feedback(int(i),w)
    records.append({'index':int(i),'prompt':int(ids[i]),'offset':int(offset),'group':group,'width':width,'reward':float(observed),'prediction':p.tolist()})
    gain=C[:,w]/(C[w,w]+noise[w]);m=m+gain*(observed-pred[i,w]-m[w]);C=C-np.outer(gain,C[w]);C=(C+C.T)/2
    state[int(ids[i])]=(m,C)
 return records

def fit_model(X,Y,ids,lambdas):
 K=X@X.T;fold=ids%3;models=[]
 for lam in lambdas:
  oof=np.zeros_like(Y)
  for f in range(3):
   tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);assert len(tr) and len(te)
   mu=Y[tr].mean(0);beta=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),Y[tr]-mu)
   oof[te]=np.clip(mu+K[np.ix_(te,tr)]@beta,1,np.arange(2,10))
  models.append((float(np.mean((oof-Y)**2)),lam,oof))
 _,lam,oof=min(models,key=lambda x:x[0]);mu=Y.mean(0);coef=np.linalg.solve(K+lam*np.eye(len(Y)),Y-mu)
 B,noise=covariance(Y-oof,ids)
 return {'lambda':lam,'mu':mu,'coef':coef,'oof':oof,'B':B,'noise':noise}

def main():
 r=Path(__file__).resolve().parent;plan=json.loads((r/'correlated_v84_plan.json').read_text());z=np.load(r/'labels_v83_features.npz');audit=json.loads((r/'labels_v83_summary.json').read_text());assert audit['passed'] and hashlib.sha256((r/'labels_v83_features.npz').read_bytes()).hexdigest()==audit['feature_npz_sha256']
 manifest=json.loads((r/'labels_v83_manifest.json').read_text());ids=z['ids'];offsets=z['offsets'];splits=z['splits'];Y=np.minimum(z['labels'],z['remaining'][:,None]).astype(float);X=np.concatenate([normalize(z['context']),normalize(z['anchor'])],axis=1)/np.sqrt(2)
 va=np.flatnonzero(splits=='validation');allrecords=[];summaries=[]
 for size in [32,64,128]:
  selected=[]
  for domain in ['alpaca','gsm8k','mbpp']:
   selected.extend([i for i,t in enumerate(manifest) if t['domain']==domain and t['split']=='train'][:size])
  tr=np.flatnonzero(np.isin(ids,selected));assert not set(ids[tr])&set(ids[va])
  model=fit_model(X[tr],Y[tr],ids[tr],plan['ridge_penalties'])
  candidates=[(0.,1.)]+[(s,d) for s in [.25,1.,4.] for d in [.5,1.]];scores=[]
  # This is training-objective optimization, not an unbiased CV estimate:
  # B is fitted to these same cross-fitted residuals. Independent validation is the gate.
  for scale,decay in candidates:
   rr=replay(model['oof'],ids[tr],offsets[tr],lambda i,w:Y[tr[i],w],model['B'],model['noise'],scale,decay);scores.append(sum(x['reward'] for x in rr))
  choice=candidates[int(np.argmax(scores))]
  pred=np.clip(model['mu']+X[va]@X[tr].T@model['coef'],1,np.arange(2,10));mean=np.repeat(Y[tr].mean(0)[None,:],len(va),axis=0)
  # Fitting API sees only training arrays. Validation feedback is accessible only through replay callback.
  totals={};records={}
  for name,p,params in [('training_mean',mean,(0.,1.)),('frozen',pred,(0.,1.)),('correlated',pred,choice)]:
   rr=replay(p,ids[va],offsets[va],lambda i,w:Y[va[i],w],model['B'],model['noise'],*params)
   observed={(x['index'],x['width']-1):x['reward'] for x in rr}
   def restricted(i,w):
    assert (i,w) in observed,'Read counterfactual feedback'
    return observed[(i,w)]
   assert replay(p,ids[va],offsets[va],restricted,model['B'],model['noise'],*params)==rr
   for cut in np.unique(offsets[va]):
    other=replay(p,ids[va],offsets[va],lambda i,w:99. if offsets[va[i]]>=cut else Y[va[i],w],model['B'],model['noise'],*params)
    signature=lambda rows:[(x['index'],x['width'],x['prediction']) for x in rows if x['offset']<=cut]
    assert signature(rr)==signature(other),'Future feedback changed decisions'
   totals[name]=sum(x['reward'] for x in rr);records[name]=rr;allrecords.extend([dict(training_per_domain=size,model=name,**x) for x in rr])
  fixed=float(Y[va,2].sum());row={'training_per_domain':size,'training_states':len(tr),'validation_states':len(va),'lambda':model['lambda'],'residual_parameters':choice,'training_objectives':scores,'fixed3_reward':fixed,'models':{k:{'reward':v,'gain_vs_fixed3_percent':100*(v/fixed-1)} for k,v in totals.items()},'correlated_vs_frozen_percent':100*(totals['correlated']/totals['frozen']-1),'covariance':model['B'].tolist(),'noise':model['noise'].tolist()}
  row['gate_passed']=totals['correlated']>=1.1*fixed and totals['correlated']>totals['frozen'];row['domains']={}
  for domain in ['alpaca','gsm8k','mbpp']:
   di=np.array([i for i in va if manifest[int(ids[i])]['domain']==domain]);base=float(Y[di,2].sum());row['domains'][domain]={'fixed3_reward':base,**{name:sum(x['reward'] for x in rr if manifest[x['prompt']]['domain']==domain) for name,rr in records.items()}}
  summaries.append(row)
 out={'passed':True,'primary_gate_passed':summaries[-1]['gate_passed'],'learning_curve':summaries,'limitations':plan['limitations'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((r/'correlated_v84_plan.json').read_bytes()).hexdigest()}
 (r/'correlated_v84_summary.json').write_text(json.dumps(out,indent=2)+'\n');(r/'correlated_v84_records.json').write_text(json.dumps(allrecords,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
