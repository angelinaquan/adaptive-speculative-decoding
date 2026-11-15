"""CPU regression: synthetic startup sampling must not activate the observer."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

p=Path(__file__).with_name('tie_audit_v108r2.py')
tree=ast.parse(p.read_text())
definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
calls=[]
def original(*args,**kwargs):
    calls.append(1)
    return 'sampled','processed'
scope={'np':np,'original':original}
exec(compile(ast.Module(body=definitions,type_ignores=[]),str(p),'exec'),scope)
sampler=SimpleNamespace(sampling_states=SimpleNamespace(temperature=SimpleNamespace(np=np.ones(2))))
args=(sampler,None,None,None,np.array([0,1]),None,None,None)
assert scope['audited_sample'](*args)==('sampled','processed')
assert len(calls)==1 and not hasattr(sampler,'_tie_audit')
worker=scope['TieAuditWorker']();worker.model_runner=SimpleNamespace(sampler=sampler)
assert worker.tie_begin()=={'enabled':True}
assert sampler._tie_audit==dict(calls=0,rows=0,mismatches=[])
try:
    scope['audited_sample'](*args)
except AssertionError:
    pass
else:
    raise AssertionError('Armed observer accepted non-greedy real requests')
assert len(calls)==2
print(json.dumps(dict(passed=True,checks=['unarmed warmup returns original outputs',
    'explicit post-init arming resets observer','armed nonzero temperatures rejected'],
    scope='CPU arming logic regression only; GPU audit not executed'),indent=2))
