"""Reconstruct the installed sampler's uniform at the observed divergence."""
import os,json,hashlib,inspect
from pathlib import Path
import numpy as np
import torch,wandb
from vllm.triton_utils import triton,tl
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as sampler

@triton.jit
def uniforms(seed_ptr,pos_ptr,out_ptr,BLOCK:tl.constexpr):
 i=tl.arange(0,BLOCK);seed=tl.load(seed_ptr);pos=tl.load(pos_ptr+i,mask=i<9,other=0)
 u=tl_rand32(seed,pos,includes_zero=False);tl.store(out_ptr+i,u,mask=i<9)

r=Path(__file__).resolve().parent;out=r/'acceptance_boundary_v45.json';assert not out.exists()
x=json.loads((r/'sampling_v44_results.json').read_text());assert x['status']=='completed'
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='sampling-v43-diagnostic',name='acceptance-boundary-reconstruction',dir=os.environ['SERVING_SCRATCH'])
result=dict(passed=False,scope=__doc__,run_url=run.url,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),trace_sha256=hashlib.sha256((r/'sampling_v44_results.json').read_bytes()).hexdigest(),installed_sampler_source_sha256=hashlib.sha256(inspect.getsource(sampler).encode()).hexdigest())
try:
 positions=torch.arange(127,136,dtype=torch.int64,device='cuda');seed=torch.tensor([421000065],dtype=torch.int64,device='cuda');values=torch.empty(9,device='cuda');uniforms[(1,)](seed,positions,values,BLOCK=16);lookup=dict(zip(positions.cpu().tolist(),values.cpu().tolist()));result['uniforms']=lookup
 cases=[]
 for p in x['passes'][4:]:
  z=next(z for z in p['trace'] if z['positions']==[131,132,133,134]);token=z['draft_tokens'][1]
  p_index=z['top_ids'][0].index(token);q_index=z['draft_top_ids'][0].index(token)
  logp=z['top_logprobs'][0][p_index];logq=z['draft_top_logprobs'][0][q_index];threshold=min(1.,float(np.exp(logp-logq)));u=lookup[131]
  predicted=u<threshold;observed=z['sampled'][0]==token
  assert predicted==observed
  cases.append(dict(repeat=p['repeat'],slot=z['slot'],seed=z['seed'],position=131,draft_token=token,target_logp=logp,draft_logq=logq,acceptance_threshold=threshold,uniform=u,predicted_accept=predicted,observed_accept=observed,emitted_tokens=z['sampled']))
 assert len({z['target_logp'] for z in cases})==1
 assert len({z['predicted_accept'] for z in cases})==2
 result.update(passed=True,cases=cases,interpretation='Saved target log probability is identical; changed draft q moves p/q across the same uniform, explaining the acceptance/rejection flip. Origin of draft variation remains to audit. Does not establish full target-logit-vector equality or full-model distribution equivalence.')
except BaseException as e:result['error']=repr(e);raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['passed']=result['passed'];art=wandb.Artifact('acceptance-boundary-v45','correctness-diagnostic');art.add_file(str(out));art.add_file(__file__);run.log_artifact(art);run.finish()
