import itertools,copy,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from batch_tree_v93 import global_trees,groups
rng=np.random.default_rng(93)
for trial in range(50):
 xs=[{'grids':[rng.normal(size=(1,2)),rng.normal(size=(2,2))],'gold':[0,1]} for _ in range(2)]
 nodes=[(i,p) for i in range(2) for n in [1,2] for p in itertools.product(range(2),repeat=n)]
 def mass(node):
  i,path=node;v=1.;prev=0
  for j,c in enumerate(path):v*=softmax(xs[i]['grids'][j][prev])[c];prev=c
  return v
 for budget in [2,3,4]:
  trees=global_trees(xs,budget,1);selected=[(i,p) for i,t in enumerate(trees) for p in t]
  best=max(sum(mass(v) for v in combo) for combo in itertools.combinations(nodes,budget) if all(len(p)==1 or (i,p[:-1]) in combo for i,p in combo))
  np.testing.assert_allclose(sum(map(mass,selected)),best,atol=1e-12)
  poison=copy.deepcopy(xs)
  for x in poison:x['gold']=[]
  assert global_trees(poison,budget,1)==trees
out={'passed':True,'exhaustive_global_allocation_checks':150,'checks':['Global node budget','Global mass equals exhaustive optimum','Gold labels do not affect allocation']}
Path(__file__).with_name('batch_tree_v93_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
