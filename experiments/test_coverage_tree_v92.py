import copy,json
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from coverage_tree_v92 import coverage_rows,logistic_loss,tree
from tree_v91 import adaptive
rng=np.random.default_rng(92);grids=[rng.normal(size=(1,3)),rng.normal(size=(3,3)),rng.normal(size=(3,3))]
a={'grids':grids,'gold':[1]};b={'grids':grids,'gold':[]};c={'grids':grids,'gold':[0,1,2]}
f,y,p=coverage_rows([a,b,c]);assert y.tolist()==[1.,0.,0.,1.,1.,1.] and len(p)==4
theta=rng.normal(size=4);numerical=approx_derivative(lambda t:np.array([logistic_loss(t,f,y)[0]]),theta).ravel();np.testing.assert_allclose(numerical,logistic_loss(theta,f,y)[1],rtol=1e-5,atol=1e-7)
for budget in range(1,9):
 assert tree(a,budget,1,coverage=np.ones(3))==adaptive(a,budget,1)
 selected=tree(a,budget,1,theta=theta);poison=copy.deepcopy(a);poison['gold']=[2,2,2]
 assert selected==tree(poison,budget,1,theta=theta)
 assert len(selected)==budget and len(set(selected))==budget
 assert all(len(p)==1 or p[:-1] in selected for p in selected)
out={'passed':True,'checks':['First absent candidate is labeled negative; no later censored labels','Logistic analytic gradient matches finite differences','Unit coverage exactly reproduces raw tree','Changing gold labels cannot change frozen tree choices','Unique nodes, prefix closure, equal node budgets']}
Path(__file__).with_name('coverage_tree_v92_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
