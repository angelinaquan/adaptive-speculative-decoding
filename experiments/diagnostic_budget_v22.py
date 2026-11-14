"""Diagnostic-only instrumentation; do not use its timings as speed evidence."""
import head_calibration_v17 as head
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
_rows=[]
_original=AdaptiveVerificationManager.get_num_tokens

def instrument(self,*args,**kwargs):
    result=_original(self,*args,**kwargs)
    drafts,non_drafts,budget=self._batch_budget
    _rows.append(dict(requests=len(drafts),available=sum(drafts.values()),budget=budget,non_draft=sum(non_drafts.values())))
    return result
AdaptiveVerificationManager.get_num_tokens=instrument

class BudgetWorker(head.HeadCalibrationWorker):
    def budget_state_v22(self,reset=False):
        manager=self.model_runner.adaptive_verification
        assert manager is not None
        result=dict(rows=list(_rows),cost_tables=[x.tolist() for x in manager.cost_tables])
        if reset:_rows.clear()
        return result
