"""Diagnostic instrumentation only; its wall times must not be speed evidence."""
import numpy as np
from cost_only_v26 import CostOnlyWorker
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
_original=AdaptiveVerificationManager.get_num_tokens
_trace=[]
_enabled=False

def observe_budget(self,num_tokens_per_req,draft_tokens):
 result=_original(self,num_tokens_per_req,draft_tokens)
 if _enabled:
  drafts,anchors,budget=self._batch_budget
  ids=list(num_tokens_per_req)
  slots=np.array([self.req_states.req_id_to_index[x] for x in ids],dtype=np.int32)
  probs=self._stale_confidences[self._stale_idx].np[slots]
  survival=np.cumprod(probs.astype(np.float64),axis=1)
  capacities=np.array([len(draft_tokens.get(x,())) for x in ids])
  valid=np.arange(self.num_speculative_steps)[None,:]<capacities[:,None]
  scores=np.sort(survival[valid])[::-1]
  _trace.append(dict(requests=len(ids),available=int(capacities.sum()),budget=int(budget),non_draft=int(sum(anchors.values())),estimated_accepted_drafts=float(scores[:budget].sum()),profile_forward_ms=float(self.cost_tables[1][result]),profile_draft_ms=float(self.cost_tables[0][len(ids)])))
 return result
AdaptiveVerificationManager.get_num_tokens=observe_budget

# Upstream closes timing events only inside _dummy_run. Add actual-serving
# boundaries explicitly; otherwise collect() silently returns zero samples.
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
_execute_original=GPUModelRunner.execute_model
_propose_original=DSparkSpeculator.propose
_collector=None

def execute_with_end(self,*args,**kwargs):
 result=_execute_original(self,*args,**kwargs)
 if _enabled:self.step_timing.forward_end()
 return result

def propose_with_events(self,*args,**kwargs):
 if _enabled:_collector.drafter_start()
 try:return _propose_original(self,*args,**kwargs)
 finally:
  if _enabled:_collector.drafter_end()
GPUModelRunner.execute_model=execute_with_end
DSparkSpeculator.propose=propose_with_events

class RuntimeCostWorker(CostOnlyWorker):
 def runtime_cost_control_v33(self,start=False):
  global _enabled,_collector
  if start:
   assert not _enabled
   _trace.clear()
   _collector=self.model_runner.step_timing
   self._runtime_context=_collector.collect()
   self._runtime_samples=self._runtime_context.__enter__()
   _enabled=True
   return {'started':True}
  assert _enabled
  _enabled=False
  self._runtime_context.__exit__(None,None,None)
  samples=[dict(requests=int(s.num_reqs),tokens=int(s.num_target_tokens),forward_ms=float(s.forward_ms),draft_ms=float(s.drafter_ms),full_graph=bool(s.full_cudagraph)) for s in self._runtime_samples]
  return dict(budgets=list(_trace),samples=samples,scope='Instrumented real-step GPU timings and CPU budget records. Not uninstrumented throughput; traces are not assumed to align one-to-one.')
