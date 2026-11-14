"""Validation intervention: preserve measured target costs below graph limit."""
import numpy as np
import diagnostic_budget_v22 as budget
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
_raw=[]
_original=AdaptiveVerificationManager.set_initial_cost_curves

def capture(self,samples):
 _raw.extend((int(s.num_target_tokens),float(s.forward_ms)) for s in samples)
 return _original(self,samples)
AdaptiveVerificationManager.set_initial_cost_curves=capture

class CostPolicyWorker(budget.BudgetWorker):
 def cost_control_v24(self,mode):
  assert mode in ['original','measured']
  manager=self.model_runner.adaptive_verification
  if not hasattr(self,'_v24_original'):
   self._v24_original=tuple(x.copy() for x in manager.cost_tables)
   grouped={n:[value for size,value in _raw if size==n] for n,_ in _raw}
   sizes=np.array(sorted(grouped));values=np.array([np.median(grouped[n]) for n in sizes])
   assert np.isfinite(values).all() and (values>0).all()
   table=self._v24_original[1].copy()
   positions=np.arange(len(table));mask=positions<=manager._cudagraph_limit
   table[mask]=values[np.minimum(np.searchsorted(sizes,positions[mask]),len(sizes)-1)]
   self._v24_measured=(self._v24_original[0].copy(),table)
  selected=self._v24_original if mode=='original' else self._v24_measured
  manager.cost_tables=tuple(x.copy() for x in selected)
  return dict(mode=mode,cost_tables=[x.tolist() for x in selected],raw_target_samples=_raw)
