"""Post-hoc equal-budget rule comparison; test outcomes never select widths."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,itertools,hashlib
import numpy as np
r=Path(__file__).resolve().parent
# Read measured timings only; exact-output provenance is independently checked.
def read(prefix,count,blocks):
 T=np.zeros((2,8,2,blocks));N=np.zeros_like(T)
 ref=json.loads((r/f'{prefix}_b64_w0_r-1.json').read_text())
 other=json.loads((r/f'{prefix}_b8_w0_r-1.json').read_text());assert ref['status']==other['status']=='completed' and ref['outputs']==other['outputs']
 for bi,b in enumerate([8,64]):
  for wi,w in enumerate(range(1,9)):
   for o in range(2):
    x=json.loads((r/f'{prefix}_b{b}_w{w}_r{o}.json').read_text());assert x['status']=='completed' and len(x['records'])==4*count//b
    for z in x['records']:
     ids=ref['outputs'][z['start']:z['start']+b];assert z['parity'] and z['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest() and z['tokens']==sum(map(len,ids))
     if z['repeat']>=0:T[bi,wi,o,z['start']//64]+=z['seconds']/2;N[bi,wi,o,z['start']//64]+=z['tokens']/2
 assert np.all(N==N[:,0:1]) and np.array_equal(N[0],N[1]);return T,N
T,N=read('profile_v77',1344,21);U,V=read('transfer_v78',384,6)
profiles=[list(range(d*7,d*7+4)) for d in range(3)];testix=[[d*7+5,d*7+6] for d in range(3)];records=[]
def choose(a,rule):
 if rule=='robust':return int(np.argmin((a/a.min(axis=0)).max(axis=1)))
 return int(np.argmin(a.sum(axis=1)))
# Deterministic tie-breaking and a synthetic case where rules should differ.
a=np.array([[1.,10.],[2.,8.]])
assert choose(a,'pooled')==1 and choose(a,'robust')==0
for bi,b in enumerate([8,64]):
 for timing in ['mean','0_to_1','1_to_0']:
  p=T[bi].mean(1) if timing=='mean' else T[bi,:,int(timing[0]),:]
  v77=T[bi].mean(1) if timing=='mean' else T[bi,:,int(timing[-1]),:]
  v78=U[bi].mean(1) if timing=='mean' else U[bi,:,int(timing[-1]),:]
  choices=[]
  for panel,ix in enumerate(itertools.product(*profiles)):
   a=p[:,ix]
   for rule in ['pooled','robust']:choices.append((rule,panel,list(ix),choose(a,rule)))
  for d,name in enumerate(['alpaca','gsm8k','mbpp']):
   for panel,ix in enumerate(itertools.combinations(profiles[d],3)):
    choices.append(('single_'+name,panel,list(ix),choose(p[:,ix],'pooled')))
  # Poison every non-profiling outcome and verify selection invariance.
  poison=p.copy();poison[:,[i for i in range(21) if i not in sum(profiles,[])]]=1e9
  for rule,panel,ix,k in choices:assert choose(poison[:,ix], 'robust' if rule=='robust' else 'pooled')==k
  for target,tm,nm,inds in [('v77',v77,N[bi].mean(1),testix),('v78',v78,V[bi].mean(1),[[0,1],[2,3],[4,5]])]:
   costs=np.stack([tm[:,ix].sum(axis=1) for ix in inds],axis=1);tokens=sum(nm[0,ix].sum() for ix in inds)
   for rule,panel,ix,k in choices:
    total=costs.sum(axis=1);slow=100*(costs[k]/costs.min(axis=0)-1)
    records.append({'batch':b,'timing':timing,'test':target,'rule':rule,'panel':panel,'profile_blocks':ix,'width':k+1,'throughput':float(tokens/total[k]),'pooled_throughput_loss_percent':float(100*(1-total.min()/total[k])),'worst_domain_slowdown_percent':float(slow.max()),'domain_slowdown_percent':slow.tolist(),'gain_vs_width8_percent':float(100*(total[7]/total[k]-1))})
summary=[];paired=[]
for b,timing,target in itertools.product([8,64],['mean','0_to_1','1_to_0'],['v77','v78']):
 group=[z for z in records if (z['batch'],z['timing'],z['test'])==(b,timing,target)]
 for rule in ['pooled','robust','single_alpaca','single_gsm8k','single_mbpp']:
  rs=[z for z in group if z['rule']==rule];summary.append({'batch':b,'timing':timing,'test':target,'rule':rule,'panels':len(rs),'width_counts':{str(k):sum(z['width']==k for z in rs) for k in range(1,9)},'mean_worst_domain_slowdown_percent':float(np.mean([z['worst_domain_slowdown_percent'] for z in rs])),'max_worst_domain_slowdown_percent':max(z['worst_domain_slowdown_percent'] for z in rs),'mean_pooled_loss_percent':float(np.mean([z['pooled_throughput_loss_percent'] for z in rs]))})
 ps=sorted([z for z in group if z['rule']=='pooled'],key=lambda z:z['panel']);rs=sorted([z for z in group if z['rule']=='robust'],key=lambda z:z['panel']);assert all(p['profile_blocks']==q['profile_blocks'] for p,q in zip(ps,rs))
 paired.append({'batch':b,'timing':timing,'test':target,'same_width_panels':sum(p['width']==q['width'] for p,q in zip(ps,rs)),'total_panels':64,'mean_throughput_gain_robust_vs_pooled_percent':float(np.mean([100*(q['throughput']/p['throughput']-1) for p,q in zip(ps,rs)])),'mean_worst_slowdown_change_pp':float(np.mean([q['worst_domain_slowdown_percent']-p['worst_domain_slowdown_percent'] for p,q in zip(ps,rs)]))})
out={'passed':True,'scope':'Post-hoc comparison at192uniqueprofilingprompts,8widths,same timing passes. Pooled/robust use identicalmixed panels; singles use all4three-block panels ineachdomain. No new GPU timings.','tests':['known robust/pooled disagreement example','nonprofiling outcome poisoning leaves selection invariant','raw output hash equality','pooled androbust paired identical profiling prompts'],'limitations':['64 mixed combinations overlap and are not64 independent experiments.','Equal prompt budget does not imply equal wall cost across domains. Pooled/robust identical data have equal measurement cost; decision overhead not timed.','Both test sets already examined inearlier work; this is exploratory method comparison, not a new prespecified confirmation.','Worst-domain slowdown uses relative time, distinct from throughput loss; test bestwidth is hindsight.','One GPU/model pair; only three domains pertest andtwo timing rounds.'],'summary':summary,'paired':paired}
(r/'robust_v80_summary.json').write_text(json.dumps(out,indent=2)+'\n');(r/'robust_v80_records.json').write_text(json.dumps(records)+'\n');print(json.dumps({'paired':paired,'mean_timing_summary':[x for x in summary if x['timing']=='mean']},indent=2))
