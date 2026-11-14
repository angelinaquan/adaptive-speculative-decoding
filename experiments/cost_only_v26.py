"""Cost-table intervention only: no confidence transforms or per-step logging."""
import numpy as np
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
_raw=[]
_original=AdaptiveVerificationManager.set_initial_cost_curves

def capture(self,samples):
 _raw.extend((int(s.num_target_tokens),float(s.forward_ms)) for s in samples)
 return _original(self,samples)
AdaptiveVerificationManager.set_initial_cost_curves=capture

class CostOnlyWorker:
 def cost_control_v26(self,mode):
  assert mode in ['original','measured']
  manager=self.model_runner.adaptive_verification
  assert manager is not None
  if not hasattr(self,'_v26_original'):
   self._v26_original=tuple(x.copy() for x in manager.cost_tables)
   sizes=np.array(sorted({n for n,_ in _raw}))
   values=np.array([np.median([v for n,v in _raw if n==size]) for size in sizes])
   assert len(sizes)>1 and np.isfinite(values).all() and (values>0).all()
   target=self._v26_original[1].copy();positions=np.arange(len(target));mask=positions<=manager._cudagraph_limit
   target[mask]=values[np.minimum(np.searchsorted(sizes,positions[mask]),len(sizes)-1)]
   self._v26_measured=(self._v26_original[0].copy(),target)
  selected=self._v26_original if mode=='original' else self._v26_measured
  manager.cost_tables=tuple(x.copy() for x in selected)
  return dict(mode=mode,cost_tables=[x.tolist() for x in selected],raw_target_samples=_raw,scope='Only target-cost table changes at/below graph limit; native confidence head, scheduler and sampler preserved')
