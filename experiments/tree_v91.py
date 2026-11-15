"""Exploratory equal-node tree screen on already-exposed fitting data.
No target outcomes are used to construct adaptive trees. Greedy decoding only.
"""
import heapq,json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from rerank_v90 import prepare

def adaptive(x,budget,temp):
 grids=x['grids'];heap=[];selected=[]
 def expand(path,logp):
  j=len(path)
  if j==len(grids):return
  row=grids[j][path[-1] if path else 0]/temp;lp=row-logsumexp(row)
  for c,p in enumerate(lp):heapq.heappush(heap,(-logp-float(p),path+(c,)))
 expand((),0.)
 while heap and len(selected)<budget:
  neg,path=heapq.heappop(heap);selected.append(path);expand(path,-neg)
 return selected

def ranks(x):
 path=[];prev=0
 for j,c in enumerate(x['gold']):
  order=np.argsort(-x['grids'][j][prev],kind='stable');path.append(int(np.flatnonzero(order==c)[0]));prev=c
 return tuple(path)

def static_fit(xs,budget):
 counts={}
 for x in xs:
  path=ranks(x)
  for j in range(1,len(path)+1):counts[path[:j]]=counts.get(path[:j],0)+1
 paths=sorted(counts,key=lambda p:(-counts[p],len(p),p))[:budget]
 assert all(len(p)==1 or p[:-1] in paths for p in paths)
 return paths

def static_tree(x,paths):
 result=[]
 for p in paths:
  actual=[]
  for j,rank in enumerate(p):
   row=x['grids'][j][actual[-1] if actual else 0];actual.append(int(np.argsort(-row,kind='stable')[rank]))
  result.append(tuple(actual))
 return result

def chain(x,budget):
 p=();out=[]
 for j in range(min(budget,len(x['grids']))):
  p=p+(int(np.argmax(x['grids'][j][p[-1] if p else 0])),);out.append(p)
 return out

def reward(x,tree):
 assert len(tree)==len(set(tree))
 assert all(len(p)==1 or p[:-1] in tree for p in tree)
 gold=tuple(x['gold']);return 1+sum(gold[:j] in tree for j in range(1,len(gold)+1))

def run():
 r=Path(__file__).resolve().parent;raw=json.loads((r/'path_v89_checks.json').read_text());results=[]
 for w in [3,8]:
  xs=prepare(raw,w);fit=[x for x in xs if x['prompt']%2==0];test=[x for x in xs if x['prompt']%2==1]
  budget=w;temps=[.1,.25,.5,1,2];scores=[sum(reward(x,adaptive(x,budget,t)) for x in fit) for t in temps];t=temps[int(np.argmax(scores))];fixed=static_fit(fit,budget)
  vals={name:sum(reward(x,fn(x)) for x in test) for name,fn in [('chain',lambda x:chain(x,budget)),('static_empirical',lambda x:static_tree(x,fixed)),('adaptive',lambda x:adaptive(x,budget,t))]}
  results.append({'width':w,'node_budget':budget,'fit_states':len(fit),'screen_states':len(test),'selected_temperature':t,'static_rank_paths':fixed,'rewards':vals,'adaptive_vs_static_percent':100*(vals['adaptive']/vals['static_empirical']-1),'adaptive_vs_chain_percent':100*(vals['adaptive']/vals['chain']-1)})
 out={'scope':'Exploratory split of previously exposed v89/v90 training pool; not fresh validation. Equal node counts, no runtime claim. Adaptive product-probability tree is prior-art-style control, not a novelty claim.','results':results}
 (r/'tree_v91_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
