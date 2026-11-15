"""Exact CPU reference for a single path maximizing modeled prefix survival."""
import numpy as np

def probabilities(grids,temperature):
 assert temperature>0
 out=[]
 for a in grids:
  a=np.asarray(a,float)/temperature;a=a-a.max(-1,keepdims=True);e=np.exp(a);out.append(e/e.sum(-1,keepdims=True))
 return out

def decode(grids,temperature=1.,objective='prefix'):
 grids=[np.asarray(a,float) for a in grids];assert grids and grids[0].shape[0]==1
 assert all(np.isfinite(a).all() for a in grids)
 for a,b in zip(grids,grids[1:]):assert a.shape[1]==b.shape[0]
 if objective=='greedy':
  path=[];prev=0
  for a in grids:prev=int(a[prev].argmax());path.append(prev)
  return path
 probs=probabilities(grids,temperature);future=np.zeros(grids[-1].shape[1]);policies=[]
 for j in range(len(grids)-1,-1,-1):
  if objective=='prefix':values=probs[j]*(1+future[None,:])
  elif objective=='map':values=np.log(np.maximum(probs[j],1e-300))+future[None,:]
  elif objective=='raw_sum':values=grids[j]+future[None,:]
  elif objective=='geometric':values=.75**j*grids[j]+future[None,:]
  else:raise ValueError(objective)
  policy=values.argmax(-1);future=values[np.arange(len(policy)),policy];policies.append(policy)
 path=[];prev=0
 for policy in reversed(policies):prev=int(policy[prev]);path.append(prev)
 return path

def value(grids,path,temperature=1.,objective='prefix'):
 p=probabilities(grids,temperature);total=0.;survival=1.;prev=0
 for j,c in enumerate(path):
  if objective=='prefix':survival*=p[j][prev,c];total+=survival
  elif objective=='map':total+=np.log(max(p[j][prev,c],1e-300))
  elif objective=='raw_sum':total+=grids[j][prev,c]
  elif objective=='geometric':total+=.75**j*grids[j][prev,c]
  prev=c
 return total

if __name__=='__main__':
 import itertools
 rng=np.random.default_rng(1865789);count=0
 for length in [1,2,3,4]:
  for _ in range(25):
   grids=[rng.normal(size=(1 if j==0 else 3,3)) for j in range(length)]
   for objective in ['prefix','map','raw_sum','geometric']:
    path=decode(grids,.7,objective);best=max(value(grids,p,.7,objective) for p in itertools.product(range(3),repeat=length))
    assert np.isclose(value(grids,path,.7,objective),best,atol=1e-12);count+=1
   assert value(grids,decode(grids,.7),.7)>=value(grids,decode(grids,objective='greedy'),.7)-1e-12
 grids=[np.log([[.55,.45]]),np.log([[.5,.5],[.99,.01]])]
 assert decode(grids)==[1,0] and decode(grids,objective='greedy')==[0,0]
 print('PASS:',count,'DP cases vs exhaustive enumeration; prefix objective dominates greedy under its own probability model')
