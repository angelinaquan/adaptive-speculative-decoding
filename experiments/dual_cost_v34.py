"""Preserve raw median draft costs as well as target costs; no per-step hooks."""
import numpy as np
from cost_only_v26 import CostOnlyWorker
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
_draft_raw=[]
_original=AdaptiveVerificationManager.set_initial_cost_curves

def capture_draft(self,samples):
 _draft_raw.extend((int(s.num_reqs),float(s.drafter_ms)) for s in samples if s.full_cudagraph)
 return _original(self,samples)
AdaptiveVerificationManager.set_initial_cost_curves=capture_draft

class DualCostWorker(CostOnlyWorker):
 def dual_cost_control_v34(self,mode):
  assert mode in ['original','target_only','both']
  self.cost_control_v26('original' if mode=='original' else 'measured')
  if not hasattr(self,'_v34_both'):
   sizes=np.array(sorted({n for n,_ in _draft_raw}))
   assert len(sizes)>1
   values=np.array([np.median([v for n,v in _draft_raw if n==size]) for size in sizes])
   assert np.isfinite(values).all() and (values>0).all()
   positions=np.arange(len(self._v26_original[0]))
   assert positions[-1]<=sizes[-1],'Do not extrapolate beyond observed request counts'
   draft=values[np.minimum(np.searchsorted(sizes,positions),len(sizes)-1)]
   self._v34_both=(draft,self._v26_measured[1].copy())
  if mode=='both':self.model_runner.adaptive_verification.cost_tables=tuple(x.copy() for x in self._v34_both)
  return dict(mode=mode,cost_tables=[x.tolist() for x in self.model_runner.adaptive_verification.cost_tables],raw_draft_samples=_draft_raw)
