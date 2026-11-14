"""Installed GPU allocator + sampler finite-state joint-distribution audit.
Not a full-model KV-cache test or proof for every scheduling rule.
"""
import os,json,itertools,hashlib,inspect
from pathlib import Path
import numpy as np
import torch
import wandb
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import _assign_draft_token_budget_compiled

r=Path(__file__).resolve().parent;out=r/'joint_sampler_v41.json';assert not out.exists()
rng=np.random.default_rng(414141)
p=np.array([[.2,.5,.3],[.6,.1,.3],[.25,.35,.4]])
q=np.array([[.7,.2,.1],[.1,.6,.3],[.4,.5,.1]])
starts=[0,1];sequences=list(itertools.product(range(3),repeat=2));lookup={seq:i for i,seq in enumerate(sequences)}
expected=np.outer([p[0,a]*p[a,b] for a,b in sequences],[p[1,a]*p[a,b] for a,b in sequences]).ravel()
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='stochastic-v41-correctness',name='joint-adaptive-allocator-sampler',dir=os.environ['SERVING_SCRATCH'])
result=dict(passed=False,cases=[],run_url=run.url,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=__doc__,draws_per_case=40000)

def tensor(x,dtype):return torch.tensor(np.asarray(x),dtype=dtype,device='cuda')
try:
 # Every possible pair of first proposals. Position-2 confidence depends on
 # that previous token; position-1 confidence is fixed before candidate1.
 # Native compiled allocator determines admission, not a replacement algorithm.
 cache={};checks=0
 for first in itertools.product(range(3),repeat=2):
  conf=np.array([[.73,[.2,.95,.5][first[0]]],[.61,[.9,.3,.7][first[1]]]],dtype=np.float32)
  for budget in [1,2,3]:
   cap=tensor([2,2],torch.int32)
   _assign_draft_token_budget_compiled(tensor(conf,torch.float32),tensor([0,1],torch.int32),cap,budget,2)
   actual=cap.cpu().numpy();score=np.cumprod(conf,axis=1);order=np.argsort(-score.ravel(),kind='stable')[:budget]
   wanted=np.bincount(order//2,minlength=2);assert np.array_equal(actual,wanted)
   for i in range(2):assert set((order[order//2==i]%2).tolist())==set(range(actual[i]))
   cache[first,budget]=actual.tolist();checks+=1
 result['compiled_allocator_cases_passed']=checks
 for mode in ['fixed2','budget1','budget2','budget3','invalid_candidate_peek']:
  counts=np.zeros(81,dtype=int);lengths=np.zeros(3,dtype=int)
  for chunk in range(40):
   targets=[];drafts=[];positions=[];mapping=[];cu=[0];draft_logits=[]
   for group in range(1000):
    proposal=[];probabilities=[]
    for initial in starts:
     prev=initial;tokens=[];probs=[]
     for j in range(2):
      probs.append(q[prev]);token=int(rng.choice(3,p=q[prev]));tokens.append(token);prev=token
     proposal.append(tokens);probabilities.append(probs)
    if mode=='fixed2':caps=[2,2]
    elif mode.startswith('budget'):caps=cache[tuple(z[0] for z in proposal),int(mode[-1])]
    else:caps=[0 if z[0]==0 else 2 for z in proposal] # intentionally illegal negative control
    for member in range(2):
     i=group*2+member;k=caps[member];lengths[k]+=1;tokens=[starts[member]]+proposal[member][:k]
     targets.extend([np.log(p[t]) for t in tokens]);drafts.extend(tokens);positions.extend(range(k+1));mapping.extend([i]*(k+1));cu.append(cu[-1]+k+1)
     draft_logits.append(np.log(probabilities[member]))
   sampled,num_sampled=rejection_sample(tensor(targets,torch.float32),tensor(draft_logits,torch.float32),tensor(drafts,torch.int64),tensor(cu,torch.int32),tensor(positions,torch.int64),torch.arange(2000,device='cuda',dtype=torch.int32),tensor(mapping,torch.int32),tensor(positions,torch.int32),torch.ones(2000,device='cuda'),tensor(rng.integers(1,2**60,size=2000),torch.int64),2)
   values=sampled.cpu().tolist();ns=num_sampled.cpu().tolist();pairs=[]
   for i,(ids,n) in enumerate(zip(values,ns)):
    assert 1<=n<=3;seq=ids[:n];assert all(0<=v<3 for v in seq)
    while len(seq)<2:seq.append(int(rng.choice(3,p=p[seq[-1]])))
    pairs.append(lookup[tuple(seq[:2])])
   for i in range(0,len(pairs),2):counts[pairs[i]*9+pairs[i+1]]+=1
  observed=counts/counts.sum();tv=float(np.abs(observed-expected).sum()/2);z=float(np.max(np.abs(observed-expected)/np.sqrt(expected*(1-expected)/counts.sum())))
  case=dict(mode=mode,draws=int(counts.sum()),joint_tv=tv,max_standardized_error=z,counts=counts.tolist(),expected=expected.tolist(),proposal_length_counts=lengths.tolist());result['cases'].append(case)
  if mode=='invalid_candidate_peek':assert tv>.04,'Negative control failed to detect selection bias'
  else:assert tv<.06 and z<6.5,case
  run.log({'mode':mode,'joint_tv':tv,'max_standardized_error':z});print(mode,tv,z,flush=True)
 result['passed']=True
except BaseException as e:result['error']=repr(e);raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n');run.summary['passed']=result['passed'];art=wandb.Artifact('joint-sampler-v41','correctness')
 for file in [out,Path(__file__)]:art.add_file(str(file))
 run.log_artifact(art);run.finish()
