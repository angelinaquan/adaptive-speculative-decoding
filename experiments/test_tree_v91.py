import itertools,copy,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from tree_v91 import adaptive,reward,chain,static_fit,static_tree
rng=np.random.default_rng(91)
for trial in range(100):
 grids=[rng.normal(size=(1,2)),rng.normal(size=(2,2)),rng.normal(size=(2,2))]
 x={'grids':grids,'gold':[0,1,0]}
 allpaths=[p for n in [1,2,3] for p in itertools.product(range(2),repeat=n)]
 def mass(path):
  prev=0;v=1.
  for j,c in enumerate(path):v*=softmax(grids[j][prev])[c];prev=c
  return v
 for budget in [1,2,3,4]:
  selected=adaptive(x,budget,1);assert len(selected)==budget
  optimum=max(sum(mass(p) for p in tree) for tree in itertools.combinations(allpaths,budget) if all(len(p)==1 or p[:-1] in tree for p in tree))
  np.testing.assert_allclose(sum(mass(p) for p in selected),optimum,atol=1e-12)
  poisoned=copy.deepcopy(x);poisoned['gold']=[1,0,1]
  assert adaptive(poisoned,budget,1)==selected
  assert 1<=reward(x,selected)<=budget+1
  assert reward(x,chain(x,budget))<=min(budget,3)+1
assert reward({'gold':[0,1]},[(0,),(0,1),(1,)])==3
out={'passed':True,'random_score_grids':100,'exhaustive_optimal_tree_comparisons':400,'checks':['prefix closure','exact node budget','gold poisoning leaves adaptive choices unchanged','heap objective equals exhaustive maximum prefix probability mass','known accepted-prefix reward'],'limitation':'These are replay construction tests, not target attention mask/KV cache correctness or serving timing.'}
Path(__file__).with_name('tree_v91_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
