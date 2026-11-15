"""Observe greedy sampler agreement with torch.argmax; never change outputs."""
import hashlib
import inspect
from pathlib import Path
import numpy as np
import torch
from vllm.v1.worker.gpu.sample.sampler import Sampler

original = Sampler.sample


def audited_sample(self, logits, expanded_idx_mapping, idx_mapping, idx_mapping_np,
                   pos, input_ids, expanded_local_pos, return_logprobs=False):
    sampled, processed = original(self, logits, expanded_idx_mapping, idx_mapping,
                                  idx_mapping_np, pos, input_ids, expanded_local_pos,
                                  return_logprobs=return_logprobs)
    if not getattr(self, '_tie_audit_enabled', False):
        return sampled, processed
    assert np.all(self.sampling_states.temperature.np[idx_mapping_np] == 0)
    slots=expanded_idx_mapping.long()
    valid=slots>=0
    temperatures=self.sampling_states.temperature.gpu[slots[valid]].cpu()
    assert torch.all(temperatures==0), 'GPU greedy temperature differs from zero'
    reference=processed.argmax(dim=-1)
    bad=torch.nonzero((sampled!=reference)&valid).flatten()
    if not hasattr(self,'_tie_audit'):
        self._tie_audit=dict(calls=0,rows=0,mismatches=[])
    self._tie_audit['calls']+=1
    self._tie_audit['rows']+=int(valid.sum().item())
    for row in bad.tolist():
        a=int(sampled[row].item());b=int(reference[row].item())
        av=float(processed[row,a].item());bv=float(processed[row,b].item())
        self._tie_audit['mismatches'].append(dict(row=row,position=int(pos[row].item()),
            slot=int(slots[row].item()),sampled=a,canonical=b,sampled_logit=av,
            canonical_logit=bv,exact_tie=av==bv,return_logprobs=return_logprobs))
    return sampled, processed


Sampler.sample = audited_sample


class TieAuditWorker:
    def tie_begin(self):
        sampler=self.model_runner.sampler
        sampler._tie_audit_enabled=True
        sampler._tie_audit=dict(calls=0,rows=0,mismatches=[])
        return {'enabled':True}

    def tie_audit(self):
        sampler=self.model_runner.sampler
        source=Path(inspect.getsourcefile(original))
        result=getattr(sampler,'_tie_audit',dict(calls=0,rows=0,mismatches=[]))
        assert result['calls']>0
        return dict(**result,source_file=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    scope='Synchronizing observer; outputs never changed. Historical reproduction checked separately.')
