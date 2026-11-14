"""Read-only sampler snapshots for a single seed; never timing evidence."""
import inspect
import torch
import numpy as np
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
import vllm.v1.worker.gpu.spec_decode.rejection_sampler as sampler_module
_original=sampler_module.rejection_sample
_signature=inspect.signature(_original)
_enabled=False
_trace=[]
_seed=421000065


def traced(*args,**kwargs):
    result=_original(*args,**kwargs)
    if not _enabled:return result
    a=_signature.bind(*args,**kwargs).arguments
    with gpu_sync_allowed():
        slots=a['idx_mapping'].cpu().tolist();seeds=a['seed'].cpu().tolist()
        rows=[i for i,slot in enumerate(slots) if seeds[slot]==_seed]
        for i in rows:
            slot=slots[i];cu=a['cu_num_logits'].cpu().tolist();lo,hi=cu[i:i+2]
            logits=a['target_logits'][lo:hi].cpu().numpy().astype(np.float64)
            logp=logits-np.logaddexp.reduce(logits,axis=-1,keepdims=True)
            topids=np.argsort(-logp,axis=-1)[:,:16];topvalues=np.take_along_axis(logp,topids,axis=-1)
            samples,counts=result;n=int(counts[i].item())
            entry=dict(slot=slot,seed=seeds[slot],positions=a['pos'][lo:hi].cpu().tolist(),
                draft_tokens=a['draft_sampled'][lo:hi].cpu().tolist(),sampled=samples[i,:n].cpu().tolist(),
                num_sampled=n,top_ids=topids.tolist(),top_logprobs=topvalues.tolist(),
                watched_logprobs=logp[:,[17738,56116]].tolist(),temperature=float(a['temperature'][slot].item()),
                num_speculative_steps=a['num_speculative_steps'])
            q=a.get('draft_logits')
            if q is not None:
                ql=q[slot].cpu().numpy().astype(np.float64);qp=ql-np.logaddexp.reduce(ql,axis=-1,keepdims=True)
                qi=np.argsort(-qp,axis=-1)[:,:8];qv=np.take_along_axis(qp,qi,axis=-1)
                entry.update(draft_top_ids=qi.tolist(),draft_top_logprobs=qv.tolist())
            _trace.append(entry)
    return result

sampler_module.rejection_sample=traced

class SamplingTraceWorker:
    def sampling_trace_control_v44(self,enabled):
        global _enabled
        previous=list(_trace);_trace.clear();_enabled=enabled
        return dict(trace=previous,scope=__doc__)
