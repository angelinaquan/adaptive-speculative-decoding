"""Exploratory global versus per-request tree allocation at equal total nodes.
Uses exposed fitting-pool states; no throughput or novelty claim.
"""
import json,heapq
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from rerank_v90 import prepare
from tree_v91 import adaptive,reward

def global_trees(xs,budget,temp):
 heap=[];out=[[] for _ in xs]
 def expand(i,path,logp):
  grids=xs[i]['grids'];j=len(path)
  if j==len(grids):return
  row=grids[j][path[-1] if path else 0]/temp;lp=row-logsumexp(row)
  for c,p in enumerate(lp):heapq.heappush(heap,(-logp-float(p),i,path+(c,)))
 for i in range(len(xs)):expand(i,(),0.)
 for _ in range(budget):
  neg,i,path=heapq.heappop(heap);out[i].append(path);expand(i,path,-neg)
 assert sum(map(len,out))==budget
 return out

def groups(xs):
 # At most one state per request per group: same anchor ordinal, randomized requests.
 by_prompt={}
 for x in xs:by_prompt.setdefault(x['prompt'],[]).append(x)
 out=[];rng=np.random.default_rng(1865793)
 for j in range(max(map(len,by_prompt.values()))):
  rows=[v[j] for v in by_prompt.values() if len(v)>j];rng.shuffle(rows)
  out.extend(rows[k:k+16] for k in range(0,len(rows),16))
 return out

def total(gs,w,t,mode):
 n=0
 for xs in gs:
  trees=global_trees(xs,w*len(xs),t) if mode=='global' else [adaptive(x,w,t) for x in xs]
  n+=sum(reward(x,p) for x,p in zip(xs,trees))
 return n

def run():
 r=Path(__file__).resolve().parent;raw=json.loads((r/'path_v89_checks.json').read_text());results=[]
 for w in [3,8]:
  xs=prepare(raw,w);fit=groups([x for x in xs if x['prompt']%2==0]);screen=groups([x for x in xs if x['prompt']%2==1]);vals={};selected={}
  for mode in ['local','global']:
   temps=[.1,.25,.5,1,2];fit_vals=[total(fit,w,t,mode) for t in temps];t=temps[int(np.argmax(fit_vals))];selected[mode]={'temperature':t,'fit_rewards':fit_vals};vals[mode]=total(screen,w,t,mode)
  results.append({'width':w,'mean_nodes_per_request':w,'screen_groups':len(screen),'screen_states':sum(map(len,screen)),'selected':selected,'rewards':vals,'global_gain_percent':100*(vals['global']/vals['local']-1)})
 out={'scope':__doc__,'results':results,'limitations':['Exposed historical training-pool prompts','Replay teacher states are not closed-loop serving traces','Equal verifier node counts do not establish equal runtime','Draft widths are matched within each comparison','Batch-level dynamic allocation has prior art; this is a mechanism screen']};(r/'batch_tree_v93_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
