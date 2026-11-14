"""Exploratory lag-aware allocation, holding the native total budget fixed.
Only already-generated token progress and current prefix confidence are used.
"""
import torch
import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive
from cost_only_v26 import CostOnlyWorker
_instances=[]
_original_init=adaptive.AdaptiveVerificationManager.__init__
_native_assign=adaptive._assign_draft_token_budget

def init(self,*args,**kwargs):
 _original_init(self,*args,**kwargs)
 self._lag_strength=torch.zeros((),device=self.req_states.device,dtype=torch.float32)
 _instances.append(self)
adaptive.AdaptiveVerificationManager.__init__=init

def assign_lag(confidence_probs,idx_mapping,capacities,draft_budget,num_steps,computed,prompt_len,strength):
 survival=confidence_probs[idx_mapping].cumprod(dim=1)
 progress=((computed[idx_mapping].float()-prompt_len[idx_mapping].float())/128.).clamp(0.,1.)
 weights=1.+strength*(progress.max()-progress)
 survival=survival*weights[:,None]
 steps=torch.arange(num_steps,device=survival.device)
 survival=survival.masked_fill(steps[None,:]>=capacities[:,None],-float('inf'))
 flat=survival.flatten();winners=flat.topk(draft_budget).indices
 admitted=torch.zeros_like(flat,dtype=torch.bool).index_fill_(0,winners,True)
 torch.sum(admitted.view_as(survival),dim=1,dtype=capacities.dtype,out=capacities)
_assign_compiled=torch.compile(assign_lag,dynamic=True)

def assign(confidence_probs,idx_mapping,capacities,draft_budget,num_steps):
 assert len(_instances)==1
 m=_instances[0]
 return _assign_compiled(confidence_probs,idx_mapping,capacities,draft_budget,num_steps,m.req_states.num_computed_tokens.gpu,m.req_states.prompt_len.gpu,m._lag_strength)
adaptive._assign_draft_token_budget_compiled=assign

class LagWorker(CostOnlyWorker):
 def lag_control_v36(self,strength):
  assert strength in [0.,1.,4.] and len(_instances)==1
  m=_instances[0];m._lag_strength.fill_(strength)
  return {'strength':strength,'scope':'Weights=1+strength*(maximum current output progress - request output progress)/128; no future feedback, no total-budget change. Restricted to128-token cap.'}
 def lag_checks_v36(self):
  import numpy as np
  rng=np.random.default_rng(18657);checks=0
  for n in [1,3,8]:
   p=rng.uniform(.1,.99,(n+2,7)).astype('float32');idx=np.arange(n,dtype='int64')[::-1].copy();cap=rng.integers(1,8,n,dtype='int64');prompt=np.full(n+2,20,dtype='int64');computed=prompt+rng.integers(0,129,n+2)
   for k in sorted({0,1,int(cap.sum())//2,int(cap.sum())}):
    for strength in [0.,1.,4.]:
     tensors=[torch.tensor(z,device='cuda') for z in [p,idx,cap,computed,prompt]];pt,it,ct,co,pr=tensors;st=torch.tensor(strength,device='cuda')
     _assign_compiled(pt,it,ct,k,7,co,pr,st);got=ct.cpu().numpy()
     progress=np.clip((computed[idx]-prompt[idx])/128.,0,1);scores=np.cumprod(p[idx],axis=1)*(1+strength*(progress.max()-progress))[:,None];scores[np.arange(7)[None,:]>=cap[:,None]]=-np.inf
     winners=np.argsort(-scores.ravel(),kind='stable')[:k];expected=np.bincount(winners//7,minlength=n)
     assert np.array_equal(got,expected),(n,k,strength,got,expected)
     assert got.sum()==k and np.all(got>=0) and np.all(got<=cap)
     if strength==0:
      native=torch.tensor(cap,device='cuda');_native_assign(pt,it,native,k,7);assert torch.equal(ct,native)
     checks+=1
  return {'passed':True,'cases':checks,'checks':['GPU compiled allocation matches independent numpy reference','Zero strength matches native allocation','Exact budget and per-request capacity bounds']}
