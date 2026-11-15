"""CUDA-event attribution around existing graph replays; no algorithm changes."""
import torch
from forced_policy_v72 import ForcedPolicyWorker
class StageProfileWorker(ForcedPolicyWorker):
 def stage_start(self):
  assert not getattr(self,'_stage_originals',[])
  self._stage_events=[];self._stage_originals=[];s=self.model_runner.speculator
  def attach(obj,name,label):
   assert obj is not None
   original=getattr(obj,name);self._stage_originals.append((obj,name,original))
   def wrapped(*args,**kwargs):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();result=original(*args,**kwargs);end.record()
    self._stage_events.append((label,start,end));return result
   setattr(obj,name,wrapped)
  for obj,name,label in [(s.pre_graphs,'run','pre'),(s.context_graphs,'run','context'),(s.backbone_graphs,'run_fullgraph','backbone'),(s.tail_graphs,'run','tail')]:attach(obj,name,label)
  return {'profiling':True,'sync_per_step':False}
 def stage_stop(self):
  torch.cuda.synchronize()
  for obj,name,original in self._stage_originals:setattr(obj,name,original)
  self._stage_originals=[];records=[{'stage':label,'ms':start.elapsed_time(end)} for label,start,end in self._stage_events]
  assert records and all(x['ms']>=0 for x in records)
  self._stage_events=[]
  return {'records':records,'scope':'Observed graph replay GPU elapsed time. Tail includes projection and selection. Eager fallback/target/host work not attributed; no head-only or speedup claim.'}
