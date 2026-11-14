"""Batch-context active vocabulary; inherited projection gates, new updates untested.

Only the greedy FP32, TP=1 Qwen3 DSpark path is supported. Target parameters
are never mutated. Both expensive draft projections use compact weights.
"""
import hashlib,json,os,types
from pathlib import Path
import torch
from vllm.model_executor.layers.vocab_parallel_embedding import linear_batch_invariant
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
original_load=DSparkSpeculator.load_draft_model

def compact_logits(model,hidden):
    return linear_batch_invariant(hidden,model._active_lm_weight,None)*model.logits_processor.scale

def compact_bias(model,embedding):
    return linear_batch_invariant(embedding,model._active_markov_weight,None)*model.logits_processor.scale

def load_active(self,target_model,target_attn_layer_names):
    target_signature=[(n,p.data_ptr(),p._version) for n,p in target_model.named_parameters()]
    model=original_load(self,target_model,target_attn_layer_names)
    assert type(model).__name__=='Qwen3DSparkForCausalLM'
    assert self.draft_logits is None, 'Greedy draft proposals only'
    assert self._draft_topk is None, 'Do not combine untested top-k and vocabulary restriction'
    assert self.vllm_config.parallel_config.tensor_parallel_size==1
    assert model.draft_id_to_target_id is None, 'Only originally full-vocabulary checkpoint'
    assert os.environ.get('VLLM_BATCH_INVARIANT')=='1'
    assert type(model.lm_head.quant_method).__name__=='UnquantizedEmbeddingMethod'
    assert type(model.model.markov_head.markov_w2.quant_method).__name__=='UnquantizedEmbeddingMethod'
    assert model.logits_processor.soft_cap is None
    assert model.logits_processor.head_dtype in (None,torch.float32)
    assert model.lm_head.weight.dtype==torch.float32
    assert model.model.markov_head.markov_w2.weight.dtype==torch.float32
    path=Path(os.environ['DRAFT_ACTIVE_VOCAB'])
    data=json.loads(path.read_text());selected=data['token_ids']
    assert len(selected)==len(set(selected)) and selected==sorted(selected)
    assert selected and min(selected)>=0 and max(selected)<model.target_vocab_size
    ids=torch.tensor(selected,dtype=torch.long,device=model.lm_head.weight.device)
    # Allocate new weights. Replacing or modifying the shared target weight
    # in place would invalidate the experiment.
    model._active_lm_weight=torch.nn.Parameter(model.lm_head.weight.index_select(0,ids).contiguous(),requires_grad=False)
    model._active_markov_weight=torch.nn.Parameter(model.model.markov_head.markov_w2.weight.index_select(0,ids).contiguous(),requires_grad=False)
    # Validate the compact projection against selected full-vocabulary logits.
    # This is initialization-only and excluded from timing.
    generator=torch.Generator(device=ids.device).manual_seed(18657)
    hidden=torch.randn((8,model._active_lm_weight.shape[1]),generator=generator,device=ids.device)
    embedding=torch.randn((8,model._active_markov_weight.shape[1]),generator=generator,device=ids.device)
    with torch.inference_mode(),gpu_sync_allowed():
        full_logits=model.compute_draft_logits(hidden).index_select(1,ids)
        small_logits=compact_logits(model,hidden)
        full_bias=model.markov_bias(embedding).index_select(1,ids)
        small_bias=compact_bias(model,embedding)
        torch.testing.assert_close(small_logits,full_logits,rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(small_bias,full_bias,rtol=1e-4,atol=1e-4)
        projection_check=dict(lm_max_abs=float((small_logits-full_logits).abs().max()),markov_max_abs=float((small_bias-full_bias).abs().max()),rtol=1e-4,atol=1e-4)
    assert target_signature==[(n,p.data_ptr(),p._version) for n,p in target_model.named_parameters()]
    model.draft_id_to_target_id=torch.nn.Parameter(ids-torch.arange(len(ids),device=ids.device),requires_grad=False)
    model.compute_draft_logits=types.MethodType(compact_logits,model)
    model.markov_bias=types.MethodType(compact_bias,model)
    model._active_vocab_metadata=dict(size=len(selected),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source=data.get('source'),projection_check=projection_check,target_parameter_signature_unchanged=True,scope='Static source-fitted vocabulary; not an online policy.')
    return model

DSparkSpeculator.load_draft_model=load_active
class ActiveVocabularyWorker:
    def active_vocab_metadata(self):
        return self.model_runner.speculator.model._active_vocab_metadata

    def set_active_vocabulary(self, token_ids, check=False):
        model=self.model_runner.speculator.model
        target=self.model_runner.model
        signature=[(n,p.data_ptr(),p._version) for n,p in target.named_parameters()]
        buffers=[model._active_lm_weight,model._active_markov_weight,model.draft_id_to_target_id]
        pointers=[p.data_ptr() for p in buffers]
        assert token_ids==sorted(set(token_ids)) and len(token_ids)==buffers[0].shape[0]
        assert min(token_ids)>=0 and max(token_ids)<model.target_vocab_size
        with torch.inference_mode(),gpu_sync_allowed():
            ids=torch.tensor(token_ids,dtype=torch.long,device=buffers[0].device)
            buffers[0].copy_(model.lm_head.weight.index_select(0,ids))
            buffers[1].copy_(model.model.markov_head.markov_w2.weight.index_select(0,ids))
            buffers[2].copy_(ids-torch.arange(len(ids),device=ids.device))
            if check:
                g=torch.Generator(device=ids.device).manual_seed(18657)
                h=torch.randn((8,buffers[0].shape[1]),generator=g,device=ids.device)
                e=torch.randn((8,buffers[1].shape[1]),generator=g,device=ids.device)
                torch.testing.assert_close(compact_logits(model,h),model.logits_processor(model.lm_head,h).index_select(1,ids),rtol=1e-4,atol=1e-4)
                torch.testing.assert_close(compact_bias(model,e),model.logits_processor(model.model.markov_head.markov_w2,e).index_select(1,ids),rtol=1e-4,atol=1e-4)
                torch.testing.assert_close(model.map_draft_to_target(torch.arange(len(ids),device=ids.device)),ids,rtol=0,atol=0)
            torch.cuda.synchronize()
        assert pointers==[p.data_ptr() for p in buffers], 'CUDA graph pointers changed'
        assert signature==[(n,p.data_ptr(),p._version) for n,p in target.named_parameters()]
        return dict(size=len(token_ids),ids_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),target_unchanged=True,buffers_stable=True,projection_checked=check)
