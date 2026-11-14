"""Instrumented acceptance/update-cost diagnostic; NOT throughput evidence."""
import torch
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
import dynamic_vocab_v58 as dynamic
import vllm.v1.worker.gpu.spec_decode.rejection_sampler as sampling
_enabled=False
_rows=[]
_mid=None
original_verifier=dynamic._original
original_observe=sampling.rejection_sample

def verifier(*args,**kwargs):
    global _mid
    result=original_verifier(*args,**kwargs)
    if _enabled:
        _mid=torch.cuda.Event(enable_timing=True);_mid.record()
    return result

def observe(*args,**kwargs):
    if not _enabled:return original_observe(*args,**kwargs)
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();result=original_observe(*args,**kwargs);end.record()
    # Copies happen after timing events. Synchronization is deferred until audit.
    counts=result[1].clone();cu=args[3].clone()
    _rows.append((start,_mid,end,counts,cu))
    return result

dynamic._original=verifier
sampling.rejection_sample=observe
class DiagnosticWorker(dynamic.DynamicVocabularyWorker):
    def diagnostic_control(self,start):
        global _enabled
        if start:
            assert not _enabled
            _rows.clear();_enabled=True
            return dict(started=True)
        assert _enabled
        _enabled=False
        with gpu_sync_allowed():
            torch.cuda.synchronize()
            rows=[]
            for begin,middle,end,counts,cu in _rows:
                n=counts.cpu().tolist();offsets=cu.cpu().tolist()
                lengths=[b-a for a,b in zip(offsets,offsets[1:])]
                assert len(lengths)==len(n)
                full=[v-1 for v,l in zip(n,lengths) if l==8]
                assert all(0<=v<=7 for v in full)
                rows.append(dict(verifier_ms=begin.elapsed_time(middle),update_span_ms=middle.elapsed_time(end),full_width_accepted=full,requests=len(n)))
        _rows.clear()
        return dict(rows=rows,scope=__doc__)
