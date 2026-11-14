"""Inspect startup and warmed execution costs without changing the policy."""
import random
import diagnostic_budget_v22 as budget
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
_initial=[]
_original=AdaptiveVerificationManager.set_initial_cost_curves

def serialize(samples):
 return [dict(num_reqs=int(s.num_reqs),tokens=int(s.num_target_tokens),forward_ms=float(s.forward_ms),draft_ms=float(s.drafter_ms),full_graph=bool(s.full_cudagraph)) for s in samples]
def capture(self,samples):
 _initial.extend(serialize(samples));return _original(self,samples)
AdaptiveVerificationManager.set_initial_cost_curves=capture

class CostWorker(budget.BudgetWorker):
 def cost_profile_v23(self):
  runner=self.model_runner
  sizes=[n for n in runner.cudagraph_manager.captured_token_counts() if 8<=n<=256 and n%8==0]
  rows={};rng=random.Random(18657)
  for uniform in [False,True]:
   for _ in range(3):
    for n in sizes:runner._dummy_run(num_tokens=n,context_len=512,uniform_decode=uniform)
   with runner.step_timing.collect() as samples:
    for _ in range(7):
     order=sizes.copy();rng.shuffle(order)
     for n in order:runner._dummy_run(num_tokens=n,context_len=512,uniform_decode=uniform)
   rows[str(uniform)]=serialize(samples)
  return dict(initial=_initial,warmed=rows,sizes=sizes,scope='Diagnostic only; no policy modification or speed claim')
