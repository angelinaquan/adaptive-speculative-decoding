"""Censored-shortlist probability calibration for equal-budget trees.
Exploratory screen only: all input prompts have been examined previously.
"""
import json,heapq
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from scipy.optimize import minimize
from rerank_v90 import prepare
from tree_v91 import adaptive,chain,static_fit,static_tree,reward

def cov_features(row,j,width):
 lp=row-logsumexp(row);p=np.exp(lp);ordered=np.sort(row)
 return np.array([1.,j/max(width-1,1),-np.sum(p*lp)/np.log(len(row)),np.clip(ordered[-1]-ordered[-2],0,10)/5])

def coverage_rows(xs):
 features=[];labels=[];present=[]
 for x in xs:
  prev=0;width=len(x['grids'])
  for j in range(min(len(x['gold'])+1,width)):
   row=x['grids'][j][prev];features.append(cov_features(row,j,width));labels.append(float(j<len(x['gold'])))
   if j==len(x['gold']):break
   y=x['gold'][j];present.append((row,y));prev=y
 return np.array(features),np.array(labels),present

def logistic_loss(theta,f,y,penalty=.01):
 z=f@theta;mask=np.array([0.,1.,1.,1.]);loss=np.mean(np.logaddexp(0,z)-y*z)+penalty*.5*np.sum(mask*theta**2)
 return loss, f.T@(expit(z)-y)/len(y)+penalty*mask*theta

def fit_coverage(xs):
 f,y,present=coverage_rows(xs);ans=minimize(lambda t:logistic_loss(t,f,y),np.array([2.,0,0,0]),jac=True,method='L-BFGS-B',bounds=[(-10,10)]*4)
 assert ans.success and np.isfinite(ans.x).all()
 # Position-only coverage is a simpler statistical baseline, with fixed Beta(1,1) smoothing.
 width=len(xs[0]['grids']);counts=np.zeros((width,2))
 for x in xs:
  for j in range(min(len(x['gold'])+1,width)):counts[j,int(j<len(x['gold']))]+=1
 coverage=(counts[:,1]+1)/(counts.sum(1)+2)
 return ans.x,coverage

def tree(x,budget,temp,theta=None,coverage=None):
 heap=[];out=[];width=len(x['grids'])
 def expand(path,logp):
  j=len(path)
  if j==width:return
  row=x['grids'][j][path[-1] if path else 0]
  if theta is not None:c=float(expit(cov_features(row,j,width)@theta))
  elif coverage is not None:c=float(coverage[j])
  else:c=1.
  z=row/temp;lp=z-logsumexp(z)+np.log(max(c,1e-15))
  for k,p in enumerate(lp):heapq.heappush(heap,(-logp-float(p),path+(k,)))
 expand((),0.)
 while heap and len(out)<budget:
  neg,path=heapq.heappop(heap);out.append(path);expand(path,-neg)
 return out

def run():
 r=Path(__file__).resolve().parent;raw=json.loads((r/'path_v89_checks.json').read_text());results=[]
 for w in [3,8]:
  xs=prepare(raw,w);fit=[x for x in xs if x['prompt']%2==0];screen=[x for x in xs if x['prompt']%2==1]
  theta,cov=fit_coverage(fit);temps=[.1,.25,.5,1,2];methods={};params={}
  for name,kwargs in [('raw',{}),('position_coverage',{'coverage':cov}),('feature_coverage',{'theta':theta})]:
   vals=[sum(reward(x,tree(x,w,t,**kwargs)) for x in fit) for t in temps];t=temps[int(np.argmax(vals))];params[name]={'temperature':t,'fit_rewards':vals}
   methods[name]=[reward(x,tree(x,w,t,**kwargs)) for x in screen]
  fixed=static_fit(fit,w);methods['static']=[reward(x,static_tree(x,fixed)) for x in screen];methods['chain']=[reward(x,chain(x,w)) for x in screen]
  totals={k:sum(v) for k,v in methods.items()};strong=max(totals[k] for k in ['raw','position_coverage','static','chain'])
  results.append({'width':w,'fit_states':len(fit),'screen_states':len(screen),'params':params,'coverage_coefficients':theta.tolist(),'position_coverage':cov.tolist(),'rewards':totals,'feature_vs_raw_percent':100*(totals['feature_coverage']/totals['raw']-1),'feature_vs_strongest_control_percent':100*(totals['feature_coverage']/strong-1)})
 out={'scope':__doc__,'results':results,'limitations':['Previously exposed v89 fitting pool; not independent validation','No attention-mask/KV correctness or serving speed measurement','Temperature tuned on fitting reward with same grid for all probability methods','Coverage learned only along true prefixes, including first missing candidate; never label later unreachable positions']}
 (r/'coverage_tree_v92_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
