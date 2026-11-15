"""Small residual score calibration, with CE and expected-prefix training controls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

def features(grids):
 width=len(grids);result=[]
 for j,g in enumerate(grids):
  g=np.asarray(g,float);k=g.shape[1];center=np.clip((g-g.mean(-1,keepdims=True))/5,-10,10)
  shared=np.broadcast_to(np.clip((g.mean(0)-g.mean())/5,-10,10),g.shape)
  rank=np.broadcast_to(-np.arange(k)/max(k-1,1),g.shape);first=np.broadcast_to((np.arange(k)==0).astype(float),g.shape)
  lp=g-logsumexp(g,axis=-1,keepdims=True);ent=-(np.exp(lp)*lp).sum(-1,keepdims=True)/np.log(max(k,2))
  base=np.stack([center,shared,rank,first,center*ent],-1)
  f=np.zeros((*g.shape,5*(width+1)));f[...,:5]=base;f[...,5*(j+1):5*(j+2)]=base;result.append(f)
 return result

def prepare(raw,width):
 examples=[]
 for record in raw['records']:
  if not record['anchors']:continue
  entry=record['widths'][str(width)];sequence=record['input_ids']+record['output_ids']
  for j,a in enumerate(record['anchors']):
   grids=[np.asarray(g,float) for g in entry['score_grids'][j]];candidate_ids=entry['candidate_ids'][j];truth=sequence[a+1:a+width+1];gold=[]
   for ids,t in zip(candidate_ids,truth):
    if t not in ids:break
    gold.append(ids.index(t))
   examples.append({'prompt':record['task_index'],'domain':record['domain'],'grids':grids,'features':features(grids),'gold':gold,'candidates':candidate_ids,'truth':truth})
 return examples

def loss_grad(theta,examples,objective,penalty):
 assert objective in ['ce','prefix'];width=len(examples[0]['grids']);loss=0.;gradient=np.zeros_like(theta)
 for x in examples:
  logqs=[];derivatives=[];prev=0
  for j,y in enumerate(x['gold']):
   f=x['features'][j][prev];scores=x['grids'][j][prev]+f@theta;lp=scores-logsumexp(scores);p=np.exp(lp)
   logqs.append(lp[y]);derivatives.append(f[y]-p@f);prev=y
  if not logqs:continue
  if objective=='ce':loss-=sum(logqs);gradient-=np.sum(derivatives,axis=0)
  else:
   survival=np.exp(np.cumsum(logqs));loss-=survival.sum();weights=np.cumsum(survival[::-1])[::-1]
   gradient-=weights@np.asarray(derivatives)
 n=len(examples)*width;return loss/n+.5*penalty*(theta@theta),gradient/n+penalty*theta

def fit(examples,objective,penalty):
 d=examples[0]['features'][0].shape[-1];theta=np.zeros(d)
 ans=minimize(lambda t:loss_grad(t,examples,objective,penalty),theta,jac=True,method='L-BFGS-B',bounds=[(-5,5)]*d,options={'maxiter':150,'ftol':1e-10,'gtol':1e-6})
 assert np.isfinite(ans.fun) and np.isfinite(ans.x).all()
 return ans.x,{'success':bool(ans.success),'message':str(ans.message),'iterations':int(ans.nit),'loss':float(ans.fun)}

def evaluate(theta,examples):
 records=[]
 for x in examples:
  prev=0;tokens=[]
  for g,f,ids in zip(x['grids'],x['features'],x['candidates']):
   prev=int(np.argmax(g[prev]+f[prev]@theta));tokens.append(ids[prev])
  accepted=0
  for a,b in zip(tokens,x['truth']):
   if a!=b:break
   accepted+=1
  records.append({'prompt':x['prompt'],'domain':x['domain'],'reward':accepted+1,'tokens':tokens})
 return records

if __name__=='__main__':
 from scipy.optimize._numdiff import approx_derivative
 rng=np.random.default_rng(1865790);grids=[rng.normal(size=(1,3)),rng.normal(size=(3,3)),rng.normal(size=(3,3))];f=features(grids)
 examples=[{'grids':grids,'features':f,'gold':[1,2,0]},{'grids':grids,'features':f,'gold':[2]},{'grids':grids,'features':f,'gold':[]}];theta=rng.normal(scale=.1,size=20)
 for objective in ['ce','prefix']:
  numerical=approx_derivative(lambda t:np.array([loss_grad(t,examples,objective,.01)[0]]),theta).reshape(-1)
  np.testing.assert_allclose(loss_grad(theta,examples,objective,.01)[1],numerical,rtol=1e-5,atol=1e-7)
 # Prefix loss equals direct expectation over all paths under the conditional score model.
 import itertools
 x=examples[0];total=0.
 for path in itertools.product(range(3),repeat=3):
  probability=1.;prev=0;accepted=0;alive=True
  for j,c in enumerate(path):
   scores=grids[j][prev]+f[j][prev]@theta;probability*=np.exp(scores[c]-logsumexp(scores));prev=c
   alive=alive and c==x['gold'][j]
   if alive:accepted+=1
  total+=probability*accepted
 loss,_=loss_grad(theta,[x],'prefix',0);np.testing.assert_allclose(-3*loss,total,atol=1e-12)
 print('PASS: CE/prefix analytic gradients vs finite differences; censored gold prefixes; expected-prefix loss vs exhaustive path expectation')
