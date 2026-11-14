"""Does previous verification feedback predict errors in the current head?
Old temperature-1 traces only. Predictions use no feedback from their own block.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
r=Path(__file__).resolve().parent

def blocks(row,alpha):
 history=0.
 for block in row['blocks']:
  y=np.asarray(block['labels']);raw=np.cumprod(expit(block['logits']))
  assert len(y)==len(raw)>0
  yield logit(np.clip(raw,1e-6,1-1e-6)),history,y
  history=(1-alpha)*history+alpha*float(np.mean(y-raw))

def predict(t,h,g):return expit(np.exp(t[0])*h+t[1]+t[2]*g)
def objective(t,h,g,y,w):
 p=predict(t,h,g);v=2*w*(p-y)*p*(1-p)
 return float(np.sum(w*(p-y)**2)+1e-4*np.dot(t,t)),np.array([np.sum(v*np.exp(t[0])*h),v.sum(),np.sum(v*g)])+2e-4*t

def main():
 rows=[];hashes={}
 for domain in ['alpaca','gsm8k']:
  path=r/f'shift_v8_{domain}_fit.jsonl';hashes[domain]=hashlib.sha256(path.read_bytes()).hexdigest();rows.extend(json.loads(z) for z in path.read_text().splitlines())
 source=[x for x in rows if x['domain']=='alpaca' and x['split']=='calibration'];validation=[x for x in rows if x['split']=='validation']
 assert len(source)==256 and len(validation)==128
 assert not {x['prompt_sha256'] for x in source}&{x['prompt_sha256'] for x in validation}
 results={};fits={};details=[]
 for alpha in [0.,.25,.5,1.]:
  hs=[];gs=[];ys=[];ws=[]
  for row in source:
   count=sum(len(b['labels']) for b in row['blocks'])
   for h,g,y in blocks(row,alpha):
    hs.extend(h);gs.extend([g]*len(y));ys.extend(y);ws.extend([1/(256*count)]*len(y))
  h,g,y,w=map(np.asarray,[hs,gs,ys,ws]);assert abs(w.sum()-1)<1e-12
  t=np.array([.1,-.2,.3]);_,grad=objective(t,h[:50],g[:50],y[:50],w[:50]);eps=1e-6
  for i in range(3):
   d=np.eye(3)[i]*eps;num=(objective(t+d,h[:50],g[:50],y[:50],w[:50])[0]-objective(t-d,h[:50],g[:50],y[:50],w[:50])[0])/(2*eps);assert abs(grad[i]-num)<1e-8
  opts=[minimize(lambda t:objective(t,h,g,y,w),np.array(v,float),jac=True,method='L-BFGS-B',bounds=[(-2,2),(-6,6),(-10,10)],options={'maxiter':400,'ftol':1e-13,'gtol':1e-8}) for v in [(0,0,0),(-.5,-.5,1),(.5,.5,-1)]]
  assert all(o.success for o in opts);theta=min(opts,key=lambda o:o.fun).x;fits[str(alpha)]=theta.tolist()
  by_domain={d:[] for d in ['alpaca','gsm8k']}
  for row in validation:
   total=0.;n=0
   for h,g,y in blocks(row,alpha):
    p=predict(theta,h,g);assert np.all(np.diff(p)<=1e-12)
    total+=np.sum((p-y)**2);n+=len(y)
   score=float(total/n);by_domain[row['domain']].append(score);details.append(dict(alpha=alpha,prompt_sha256=row['prompt_sha256'],domain=row['domain'],brier=score))
  results[str(alpha)]={d:float(np.mean(v)) for d,v in by_domain.items()};results[str(alpha)]['combined']=float(np.mean([v for vs in by_domain.values() for v in vs]))
 # Causal regression test: changing current feedback cannot change that forecast's features.
 import copy
 a=validation[0];b=copy.deepcopy(a);b['blocks'][0]['labels']=[1-x for x in b['blocks'][0]['labels']]
 before=next(blocks(a,.5));after=next(blocks(b,.5));assert np.array_equal(before[0],after[0]) and before[1]==after[1]
 report=dict(scope='Old temperature1 source256 chat and reused validation64 chat64 math. Exploratory forecast test, NOT wall-clock gain.',hypothesis='Within-request persistence in head forecast errors can be predicted from prior verification feedback; same shared monotone mapping with history-dependent intercept.',source_hashes=hashes,fits=fits,results=results,details=details,checks={'gradient':True,'monotone_prefix':True,'current_feedback_not_in_current_forecast':True,'source_validation_disjoint':True},limitations=['History resets per request; no domain identity feature.','All EWMA rates reported; selection needs fresh validation and then fresh test.','Teacher-forced trace replay does not establish closed-loop serving speed or sampling correctness.','Temperature1 is not the matched greedy configuration. No novelty claim.'])
 (r/'history_v31_exploratory.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'fits':fits,'results':results},indent=2))
if __name__=='__main__':main()
