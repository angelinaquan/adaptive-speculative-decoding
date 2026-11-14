"""Generation-time target-candidate recency baseline; NOT a new algorithm claim.

Update after the unchanged target verifier. Only logits on the verified prefix,
including the correction/bonus row, contribute. No host tensor reads in updates.
"""
import inspect,hashlib,json
import torch
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from active_vocab_v55 import ActiveVocabularyWorker,compact_logits,compact_bias
import vllm.v1.worker.gpu.spec_decode.rejection_sampler as sampling
_original=sampling.rejection_sample
_enabled=False
_model=None
_state=None
_step=1
_updates=0
_target=None
_signature=None
_pointers=None

def candidate_ids(logits,cu,counts,k=16):
    row=torch.arange(logits.shape[0],device=logits.device)
    req=torch.bucketize(row,cu[1:],right=True)
    valid=row-cu[req]<counts[req]
    ids=torch.topk(logits,k=k,dim=-1).indices
    return ids,valid

def update_recency(state,ids,valid,step,budget):
    stamps=torch.where(valid[:,None],step,0).expand_as(ids).to(torch.int64)
    state.scatter_reduce_(0,ids.flatten(),stamps.flatten(),reduce='amax',include_self=True)
    # Unique integer keys give deterministic lowest-token-id tie breaking.
    vocab=state.numel()
    key=state*vocab+(vocab-1-torch.arange(vocab,device=state.device,dtype=torch.int64))
    return torch.sort(torch.topk(key,k=budget).indices).values

def observe(target_logits,draft_logits,draft_sampled,cu_num_logits,*args,**kwargs):
    global _step,_updates
    result=_original(target_logits,draft_logits,draft_sampled,cu_num_logits,*args,**kwargs)
    if not _enabled:return result
    assert draft_logits is None, 'Greedy draft only'
    with torch.inference_mode():
        _step+=1
        ids,valid=candidate_ids(target_logits,cu_num_logits,result[1])
        selected=update_recency(_state,ids,valid,_step,_model._active_lm_weight.shape[0])
        _model._active_lm_weight.copy_(_model.lm_head.weight.index_select(0,selected))
        _model._active_markov_weight.copy_(_model.model.markov_head.markov_w2.weight.index_select(0,selected))
        _model.draft_id_to_target_id.copy_(selected-torch.arange(len(selected),device=selected.device))
    _updates+=1
    return result
sampling.rejection_sample=observe

class DynamicVocabularyWorker(ActiveVocabularyWorker):
    def reset_dynamic(self,token_ids,enabled):
        global _enabled,_model,_state,_step,_updates,_target,_signature,_pointers
        _enabled=False
        info=self.set_active_vocabulary(token_ids,check=False)
        _model=self.model_runner.speculator.model
        _target=self.model_runner.model
        _signature=[(n,p.data_ptr(),p._version) for n,p in _target.named_parameters()]
        _pointers=[p.data_ptr() for p in [_model._active_lm_weight,_model._active_markov_weight,_model.draft_id_to_target_id]]
        with torch.inference_mode():
            _state=torch.zeros(_model.target_vocab_size,dtype=torch.int64,device=_model._active_lm_weight.device)
            ids=torch.tensor(token_ids,device=_state.device,dtype=torch.long)
            _state[ids]=1
        _step=1;_updates=0;_enabled=enabled
        return info

    def dynamic_audit(self,check=False):
        assert _signature==[(n,p.data_ptr(),p._version) for n,p in _target.named_parameters()]
        assert _pointers==[p.data_ptr() for p in [_model._active_lm_weight,_model._active_markov_weight,_model.draft_id_to_target_id]]
        with torch.inference_mode(),gpu_sync_allowed():
            ids=_model.draft_id_to_target_id+torch.arange(_model.draft_id_to_target_id.numel(),device=_state.device)
            selected=ids.cpu().tolist()
            assert selected==sorted(set(selected))
            if check:
                g=torch.Generator(device=ids.device).manual_seed(18657)
                h=torch.randn((8,_model._active_lm_weight.shape[1]),generator=g,device=ids.device)
                e=torch.randn((8,_model._active_markov_weight.shape[1]),generator=g,device=ids.device)
                torch.testing.assert_close(compact_logits(_model,h),_model.logits_processor(_model.lm_head,h).index_select(1,ids),rtol=1e-4,atol=1e-4)
                torch.testing.assert_close(compact_bias(_model,e),_model.logits_processor(_model.model.markov_head.markov_w2,e).index_select(1,ids),rtol=1e-4,atol=1e-4)
        return dict(updates=_updates,enabled=_enabled,final_ids=selected,projection_checked=check,target_unchanged=True,buffers_stable=True,scope=__doc__)
