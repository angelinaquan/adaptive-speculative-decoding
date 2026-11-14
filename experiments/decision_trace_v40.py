"""Instrument full verification to measure decision headroom, never throughput."""
import numpy as np
from cost_only_v26 import CostOnlyWorker
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
_original_budget=AdaptiveVerificationManager.get_num_tokens
_original_observe=DSparkSpeculator.observe_verification
_manager=None
_trace=[]
_enabled=False
_pending=None


def full_budget(self,tokens,drafts):
    global _pending
    result=_original_budget(self,tokens,drafts)
    if not _enabled:return result
    capacities,anchors,budget=self._batch_budget
    total=sum(capacities.values())
    assert total<=self._max_total_logits-len(tokens)*self.num_bonus_tokens
    self._batch_budget=(capacities,anchors,total)
    slots=[self.req_states.req_id_to_index[k] for k in tokens]
    with gpu_sync_allowed():p=self._confidence_probs.cpu().numpy()[slots].copy()
    assert _pending is None, 'Unmatched budget record'
    _pending=dict(slots=slots,capacities=list(capacities.values()),anchors=list(anchors.values()),
        confidence=p.tolist(),native_budget=int(budget),
        sampling_requests=int(np.count_nonzero(self.req_states.num_computed_tokens_np[slots]+np.array(list(anchors.values()))>=self.req_states.prefill_len.np[slots])))
    return sum(anchors.values())+total


def observe(self,slots,num_sampled,num_rejected):
    global _pending
    result=_original_observe(self,slots,num_sampled,num_rejected)
    if not _enabled:return result
    with gpu_sync_allowed():
        indices=slots.cpu().numpy().tolist();sampled=num_sampled.cpu().numpy();rejected=num_rejected.cpu().numpy()
    # Initial prefill may precede the first adaptive budget call.
    if _pending is None:
        assert np.all(np.maximum(sampled-1,0)+rejected==0)
        return result
    assert set(indices)==set(_pending['slots'])
    accepted=np.maximum(sampled-1,0);admitted=accepted+rejected
    mapping={slot:i for i,slot in enumerate(indices)}
    order=[mapping[slot] for slot in _pending['slots']]
    accepted=accepted[order];admitted=admitted[order]
    assert np.array_equal(admitted,np.asarray(_pending['capacities'])), (admitted,_pending['capacities'])
    assert np.all((accepted>=0)&(accepted<=admitted))
    _pending.update(accepted=accepted.tolist(),admitted=admitted.tolist())
    _trace.append(_pending);_pending=None
    return result

AdaptiveVerificationManager.get_num_tokens=full_budget
DSparkSpeculator.observe_verification=observe

class DecisionWorker(CostOnlyWorker):
    def trace_control_v40(self,enabled):
        global _enabled,_manager,_pending
        assert _pending is None
        previous=list(_trace);_trace.clear()
        _manager=self.model_runner.adaptive_verification
        assert _manager is not None
        _enabled=enabled
        return dict(trace=previous,scope='Full-verification diagnostic with synchronizing observations. Not runtime evidence or a deployable oracle policy.')
