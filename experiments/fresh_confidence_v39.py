"""Diagnostic and causal fresh-confidence ablation; no future labels used."""
import numpy as np
from cost_only_v26 import CostOnlyWorker
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
_original = AdaptiveVerificationManager.get_num_tokens


def get_num_tokens(self, num_tokens_per_req, draft_tokens):
    mode = getattr(self, '_v39_mode', 'native')
    if mode == 'native':
        return _original(self, num_tokens_per_req, draft_tokens)
    slots = np.array([self.req_states.req_id_to_index[k] for k in num_tokens_per_req], dtype=np.int32)
    target = self._stale_confidences[self._stale_idx].np
    saved = target[slots].copy()
    # Latest asynchronous copy is in the other buffer. Request-slot resets may
    # postdate that copy; those rows have current confidence 1 by definition.
    with gpu_sync_allowed():
        self._copy_events[self._stale_idx ^ 1].synchronize()
    fresh = self._stale_confidences[self._stale_idx ^ 1].np[slots].copy()
    fresh[np.isin(slots, self._pending_resets)] = 1.
    diagnostic = mode == 'diagnostic'
    if diagnostic:
        with gpu_sync_allowed():
            actual = self._confidence_probs.cpu().numpy()[slots]
        np.testing.assert_array_equal(fresh, actual)
        old_result = _original(self, num_tokens_per_req, draft_tokens)
        old_budget = self._batch_budget
    try:
        target[slots] = fresh
        result = _original(self, num_tokens_per_req, draft_tokens)
        if diagnostic:
            new_budget = self._batch_budget
            self._v39_trace.append(dict(requests=len(slots), capacities=list(new_budget[0].values()),
                anchors=list(new_budget[1].values()), old_budget=int(old_budget[2]),
                fresh_budget=int(new_budget[2]), old_tokens=int(old_result), fresh_tokens=int(result),
                stale_confidences=saved.tolist(), fresh_confidences=fresh.tolist(),
                max_logits=self._max_total_logits, bonus_tokens=self.num_bonus_tokens,
                sampling_requests=int(np.count_nonzero(self.req_states.num_computed_tokens_np[slots]
                    + np.array(list(new_budget[1].values())) >= self.req_states.prefill_len.np[slots]))))
            self._batch_budget = old_budget
            return old_result
        return result
    finally:
        target[slots] = saved

AdaptiveVerificationManager.get_num_tokens = get_num_tokens


class FreshWorker(CostOnlyWorker):
    def fresh_control_v39(self, mode):
        assert mode in ['native', 'fresh', 'diagnostic']
        manager = self.model_runner.adaptive_verification
        assert manager is not None
        previous = getattr(manager, '_v39_trace', [])
        manager._v39_mode = mode
        manager._v39_trace = []
        return dict(mode=mode, trace=previous)
