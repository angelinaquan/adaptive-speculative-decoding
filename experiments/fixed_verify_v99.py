"""Force width3 or8 drafting and exactly min(3, legal scheduled) verification.
No confidence head or learned score chooses the allocation. Audit syncs only in
an explicitly untimed pass; normal timing keeps GPU verification selection.
"""
import hashlib
from pathlib import Path
import numpy as np
from forced_policy_v72 import ForcedPolicyWorker
from forespark.fore_d.runtime.verification import JointVerificationManager

def caps_for(scheduled,legal,reset):
 caps=np.minimum(np.minimum(np.asarray(scheduled),np.asarray(legal)),3).astype(np.int64)
 caps[np.asarray(reset,dtype=bool)]=0
 assert np.all(caps>=0)
 return caps

def get_num_tokens(self,num_tokens_per_req,draft_tokens):
 req_ids=list(num_tokens_per_req);scheduled=np.array([len(draft_tokens.get(r,())) for r in req_ids],dtype=np.int64)
 non_draft={r:num_tokens_per_req[r]-int(s) for r,s in zip(req_ids,scheduled,strict=True)};assert all(v>=0 for v in non_draft.values())
 slots=np.array([self.req_states.req_id_to_index[r] for r in req_ids],dtype=np.int64);spec=self.speculator
 legal=spec.pending[1][slots] if spec.pending is not None else np.zeros_like(scheduled)
 caps=caps_for(scheduled,legal,np.isin(slots,list(spec.reset_slots)))
 quota=int(caps.sum());room=max(0,self._max_total_logits-len(req_ids)*self.num_bonus_tokens);assert quota<=room,'Insufficient logit capacity for fixed verification gate'
 self._batch_budget=(dict(zip(req_ids,caps.tolist(),strict=True)),non_draft,quota)
 spec._v99_quota_calls=getattr(spec,'_v99_quota_calls',0)+1
 return sum(non_draft.values())+quota

original_reallocate=JointVerificationManager.reallocate_drafts

def reallocate(self,req_ids,idx_mapping):
 budgets=self._batch_budget;out=original_reallocate(self,req_ids,idx_mapping)
 if getattr(self.speculator,'_v99_audit',False):
  drafts,non_draft,quota=budgets;n=len(req_ids)
  actual=np.diff(self.query_start_loc[:n+1].cpu().numpy())-np.array([non_draft[r] for r in req_ids])
  expected=np.array([drafts[r] for r in req_ids]);assert np.array_equal(actual,expected),('Actual GPU verification lengths differ',actual.tolist(),expected.tolist())
  assert int(actual.sum())==quota
  self.speculator._v99_audit_batches+=1
 return out

JointVerificationManager.get_num_tokens=get_num_tokens
JointVerificationManager.reallocate_drafts=reallocate

class FixedVerifyWorker(ForcedPolicyWorker):
 def force_begin(self,width):
  result=super().force_begin(width);spec=self.model_runner.speculator;spec._v99_audit=False;spec._v99_audit_batches=0;spec._v99_quota_calls=0;return result
 def verify_audit_start(self):
  self.model_runner.speculator._v99_audit=True
  return {'untimed_audit':True}
 def force_audit(self):
  result=super().force_audit();spec=self.model_runner.speculator;assert spec._v99_quota_calls>0
  result.update(verification_cap=3,quota_calls=spec._v99_quota_calls,exact_gpu_audit_batches=spec._v99_audit_batches,allocation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  if spec._v99_audit:assert spec._v99_audit_batches>0
  return result
