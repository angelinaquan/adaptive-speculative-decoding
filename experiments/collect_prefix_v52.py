"""Post-verification source-data collection; synchronization is never timed evidence."""
import inspect
import torch
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
import vllm.v1.worker.gpu.spec_decode.rejection_sampler as module
original=module.rejection_sample
signature=inspect.signature(original)
enabled=False
records=[]
def observe(*args,**kwargs):
    result=original(*args,**kwargs)
    if not enabled:return result
    a=signature.bind(*args,**kwargs).arguments
    with gpu_sync_allowed():
        cu=a['cu_num_logits'].cpu().tolist()
        ids=a['draft_sampled'].cpu().tolist()
        sampled,counts=result
        counts=counts.cpu().tolist()
        slots=a['idx_mapping'].cpu().tolist()
        for i,(lo,hi) in enumerate(zip(cu,cu[1:])):
            # Only full-width states. Prefix labels refer to sampler acceptance,
            # before final EOS/output-cap truncation, not committed output length.
            if hi-lo!=8:continue
            accepted=max(int(counts[i])-1,0)
            assert 0<=accepted<=7
            records.append(dict(slot=int(slots[i]),accepted_prefix=ids[lo:lo+accepted],accepted=accepted))
    return result
module.rejection_sample=observe
class PrefixCollectionWorker:
    def prefix_collection_control(self,start):
        global enabled
        previous=list(records);records.clear();enabled=start
        return dict(records=previous,scope=__doc__)
