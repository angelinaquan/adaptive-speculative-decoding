"""Pure-cap correctness without importing GPU-only runtime locally."""
import ast,json
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent;source=ast.parse((r/'fixed_verify_v99.py').read_text());fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='caps_for');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'caps','exec'),ns);rng=np.random.default_rng(99)
for _ in range(1000):
 n=int(rng.integers(1,65));scheduled=rng.integers(0,9,n);legal=rng.integers(0,9,n);reset=rng.integers(0,2,n).astype(bool);actual=ns['caps_for'](scheduled,legal,reset);expected=[0 if reset[i] else min(3,int(scheduled[i]),int(legal[i])) for i in range(n)];np.testing.assert_array_equal(actual,expected)
out={'passed':True,'cases':1000,'scope':'Host-cap function only; actual GPU allocations and target output equality are mandatory in the serving run'};(r/'cost_v99_local_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
