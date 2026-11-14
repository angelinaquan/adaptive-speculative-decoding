"""Exploratory prefix calibration on old temperature-1 validation traces only."""
import json,hashlib,copy
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from ops_baselines import OPS,isotonic_decreasing
r=Path(__file__).resolve().parent

def feature(z):
 return logit(np.clip(np.cumprod(expit(np.asarray(z))),1e-6,1-1e-6))
def prediction(t,h):return expit(np.exp(t[0])*h+t[1])
def objective(t,h,y,w):
 p=prediction(t,h);g=2*(p-y)*p*(1-p)
 return float(np.sum(w*(p-y)**2)+1e-4*np.dot(t,t)),np.array([np.sum(w*g*np.exp(t[0])*h),np.sum(w*g)])+2e-4*t

def main():
 rows=[];hashes={}
 for domain in ['alpaca','gsm8k']:
  path=r/f'shift_v8_{domain}_fit.jsonl';hashes[domain]=hashlib.sha256(path.read_bytes()).hexdigest();rows.extend(json.loads(line) for line in path.read_text().splitlines())
 source=[x for x in rows if x['domain']=='alpaca' and x['split']=='calibration'];validation=[x for x in rows if x['split']=='validation']
 assert len(source)==256 and len(validation)==128
 assert not {x['prompt_sha256'] for x in source}&{x['prompt_sha256'] for x in validation}
 hs=[];ys=[];ws=[]
 for row in source:
  count=sum(len(b['labels']) for b in row['blocks'])
  for b in row['blocks']:
   n=len(b['labels']);hs.extend(feature(b['logits']));ys.extend(b['labels']);ws.extend([1/(256*count)]*n)
 h,y,w=map(np.asarray,[hs,ys,ws]);assert abs(w.sum()-1)<1e-12
 t=np.array([.2,-.3]);value,g=objective(t,h[:50],y[:50],w[:50]);eps=1e-6
 for i in range(2):
  delta=np.eye(2)[i]*eps;num=(objective(t+delta,h[:50],y[:50],w[:50])[0]-objective(t-delta,h[:50],y[:50],w[:50])[0])/(2*eps);assert abs(num-g[i])<1e-8
 z=np.array([.4,-.3,1.,-.8,2.,.3,.1]);p=prediction(t,feature(z));assert np.all(np.diff(p)<=0)
 changed=z.copy();changed[4:]=9;assert np.array_equal(prediction(t,feature(changed))[:4],p[:4])
 fits=[minimize(lambda x:objective(x,h,y,w),np.array(initial,dtype=float),jac=True,method='L-BFGS-B',bounds=[(-2,2),(-6,6)],options={'maxiter':400,'ftol':1e-13,'gtol':1e-8}) for initial in [(0,0),(-.5,-.5),(.5,.5)]]
 assert all(f.success for f in fits);theta=min(fits,key=lambda f:f.fun).x
 prior=json.loads((r/'shift_v8_frozen_choices.json').read_text());assert prior['fit_hashes']==hashes
 global_t=np.array(prior['fits']['256:alpaca:global']);strong=prior['strongest_validation_selected_frozen'];strong_t=np.array(strong['theta'])
 initial_ops=OPS()
 for row in source:
  for b in row['blocks']:
   n=len(b['labels']);base=np.cumprod(expit(b['raw_logits']));_,_,ctx=initial_ops.predict(base);initial_ops.update(tuple(x[:n] for x in ctx),np.asarray(b['labels']))
 methods=['raw','conditional_global_frozen','conditional_best_frozen','prefix_frozen','prefix_online_0.001','prefix_online_0.01','prefix_online_0.05','prefix_online_0.1','ops_frozen','ops_online','tops_frozen','tops_online']
 results={};details=[]
 for scenario,domains in [('chat',['alpaca']),('math',['gsm8k']),('chat_to_math',['alpaca','gsm8k'])]:
  ordered=[]
  for domain in domains:ordered.extend(sorted([x for x in validation if x['domain']==domain],key=lambda x:hashlib.sha256(('v8-order:'+x['prompt_sha256']).encode()).hexdigest()))
  states=[theta.copy() for _ in range(4)];ops=copy.deepcopy(initial_ops);frozen_ops=copy.deepcopy(initial_ops);scores=[]
  for row in ordered:
   total=np.zeros(len(methods));count=0
   for b in row['blocks']:
    n=len(b['labels']);labels=np.array(b['labels']);z=np.array(b['logits']);h=feature(z);raw=np.cumprod(expit(z))
    base=np.cumprod(expit(b['raw_logits']));op,tp,ctx=ops.predict(base);fo,ft,_=frozen_ops.predict(base)
    def conditional(t):return np.cumprod(expit(np.exp(t[0])*z+(t[1] if len(t)==2 else t[1:1+n])))
    predictions=[raw,conditional(global_t),conditional(strong_t),prediction(theta,h)]+[prediction(t,h) for t in states]+[isotonic_decreasing(x)[:n] for x in [fo,op,ft,tp]]
    total+=np.array([np.sum((p-labels)**2) for p in predictions]);count+=n
    # All forecasts and scores are recorded before feedback updates.
    for i,lr in enumerate([.001,.01,.05,.1]):
     _,grad=objective(states[i],h,labels,np.ones(n)/n);states[i]=np.clip(states[i]-lr*grad,[-2,-6],[2,6])
    ops.update(tuple(x[:n] for x in ctx),labels)
   scores.append(total/count);details.append(dict(scenario=scenario,prompt_sha256=row['prompt_sha256'],domain=row['domain'],brier=dict(zip(methods,(total/count).tolist()))))
  values=np.mean(scores,axis=0);results[scenario]={m:float(v) for m,v in zip(methods,values)}
 result=dict(scope='Exploratory OLD temperature1 traces, no held-out or speed claim; frozen GPU v26 run untouched',fit_hashes=hashes,source_prompts=256,validation_prompts=128,theta=theta.tolist(),checks=dict(gradient=True,monotonicity=True,suffix_independence=True,disjoint_source_validation=True),results=results,details=details,limitations=['Temperature1 data cannot be transferred silently to greedy serving experiments.','OPS/TOPS use the existing full-block isotonic projection for offline scoring; no sampling-exact runtime claim.','Source and validation traces were used in earlier exploration; all learning rates are reported.','Direct shared prefix mapping is a variant to investigate, not a novelty claim.'])
 (r/'prefix_v27_exploratory.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
if __name__=='__main__':main()
