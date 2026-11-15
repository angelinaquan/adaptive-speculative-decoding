"""Input and decision-interface audit; does not inspect experiment outcomes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,itertools
from pathlib import Path
import numpy as np
from correlated_v84 import normalize,replay
from allocation_headroom_v73 import allocate
r=Path(__file__).resolve().parent;z=np.load(r/'labels_v83_features.npz');manifest=json.loads((r/'labels_v83_manifest.json').read_text());ids=z['ids'];tr=np.flatnonzero(z['splits']=='train');va=np.flatnonzero(z['splits']=='validation')
assert not set(ids[tr])&set(ids[va])
def texts(ix):return {' '.join(manifest[int(i)]['text'].split()) for i in set(ids[ix])}
assert not texts(tr)&texts(va)
X=np.concatenate([normalize(z['context']),normalize(z['anchor'])],axis=1)/np.sqrt(2);assert np.isfinite(X).all();np.testing.assert_allclose(np.linalg.norm(X,axis=1),1,atol=1e-6)
for fold in range(3):
 a=set(ids[tr][ids[tr]%3!=fold]);b=set(ids[tr][ids[tr]%3==fold]);assert a and b and not a&b
rng=np.random.default_rng(95)
for trial in range(30):
 pred=rng.uniform(1,2,(4,8));pid=np.array([0,1,0,1]);off=np.array([0,0,8,8]);args=(pred,pid,off)
 first=replay(*args,lambda i,w:1.,np.zeros((8,8)),np.ones(8),0,1);second=replay(*args,lambda i,w:float(w+2),np.zeros((8,8)),np.ones(8),0,1)
 assert [(a['width'],a['prediction']) for a in first]==[(a['width'],a['prediction']) for a in second]
 for o in [0,8]:assert sum(a['width'] for a in first if a['offset']==o)<=6
 # Kernel distance formula is valid for normalized features.
 a=rng.normal(size=(8,5));a=normalize(a);D=np.maximum(0,2-2*a@a.T);np.testing.assert_allclose(D,((a[:,None]-a[None,:])**2).sum(-1),atol=1e-12)
 K=np.exp(-D);assert np.linalg.eigvalsh(K).min()>-1e-10
out={'passed':True,'checks':['Train/evaluation prompt IDs and normalized texts disjoint','Prompt-grouped folds do not split a request','All actual feature vectors finite and unit normalized','Frozen zero-feedback policy unaffected by all reward replacements','Synthetic token budget checks','RBF distance identity and positive-semidefinite synthetic kernels'],'experiment_source_sha256':hashlib.sha256((r/'kernel_v95.py').read_bytes()).hexdigest(),'scope':'Input/decision-interface audit only; no experiment outcomes inspected'}
(r/'kernel_v95_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
