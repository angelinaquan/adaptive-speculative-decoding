"""Force all legal draft tokens and verify all, preserving graph execution. No learned policy."""
import hashlib
from pathlib import Path
import numpy as np
from dataclasses import asdict, replace
from forespark.fore_d.runtime.speculator import ForeDSpeculator
from forespark.fore_d.scheduler.planner import Plan

def force_plan(self,input_batch,legal,dummy_run):
    total=int(legal.sum())
    if not dummy_run:
        assert self.config.log is None and self.steplog is None
        assert self.config.fixed_width in range(1,9)
        assert np.all((legal>=0)&(legal<=self.config.fixed_width))
        self._force_count=getattr(self,'_force_count',0)+1
    return Plan(total,total,0.0)
ForeDSpeculator._plan=force_plan

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x

class ForcedPolicyWorker:
    def force_begin(self,width):
        spec=self.model_runner.speculator
        assert isinstance(spec,ForeDSpeculator) and width in range(1,9)
        assert spec.config.log is None and spec.steplog is None
        spec.config=replace(spec.config,fixed_width=width)
        spec._force_count=0
        self._force_signature=[(n,p.data_ptr(),p._version) for n,p in self.model_runner.model.named_parameters()]
        return {'width':width}
    def force_audit(self):
        spec=self.model_runner.speculator
        assert self._force_signature==[(n,p.data_ptr(),p._version) for n,p in self.model_runner.model.named_parameters()]
        assert spec._force_count>0 and spec.config.log is None and spec.steplog is None
        return {'width':spec.config.fixed_width,'forced_steps':spec._force_count,'target_unchanged':True,'logging_disabled':True,'policy_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'startup_costs':serial(asdict(spec.planner.cost)) if spec.planner else None,'backbone_graphs':len(spec.backbone_graphs.graphs),'pre_graphs':len(spec.pre_graphs.graphs) if spec.pre_graphs else 0,'tail_graphs':len(spec.tail_graphs.graphs) if spec.tail_graphs else 0}
