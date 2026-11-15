"""Synthetic mechanism/causality checks; never counted as empirical evidence."""
import numpy as np
from correlated_v84 import covariance,replay,fit_model
ids=np.tile(np.arange(16),4);offsets=np.repeat([0,8,16,24],16);w=np.arange(1,9)
p=np.tile(1+.4*w-.025*w*w,(64,1));loading=.2*w
Y=p+np.where(ids<8,1.,-1.)[:,None]*loading
B=np.outer(loading,loading);noise=np.full(8,.05)
frozen=replay(p,ids,offsets,lambda i,j:Y[i,j],B,noise,0,1)
adaptive=replay(p,ids,offsets,lambda i,j:Y[i,j],B,noise,1,1)
assert sum(x['reward'] for x in adaptive)>sum(x['reward'] for x in frozen)
assert [x['width'] for x in adaptive if x['offset']==0]==[x['width'] for x in frozen if x['offset']==0]
assert any(x['width']!=y['width'] for x,y in zip(adaptive,frozen) if x['offset']>0)
for cut in [0,8,16,24]:
 rr=replay(p,ids,offsets,lambda i,j:100. if offsets[i]>=cut else Y[i,j],B,noise,1,1)
 sig=lambda rows:[(x['index'],x['width'],x['prediction']) for x in rows if x['offset']<=cut]
 assert sig(rr)==sig(adaptive)
observed={(x['index'],x['width']-1):x['reward'] for x in adaptive}
assert replay(p,ids,offsets,lambda i,j:observed[(i,j)],B,noise,1,1)==adaptive
estimated,R=covariance(Y-p,ids)
np.testing.assert_allclose(estimated,B,atol=1e-12)
assert np.linalg.eigvalsh(estimated).min()>-1e-10 and (R>0).all()
# Contrasting invariant: a constant offset across widths cancels if there is no clipping.
from allocation_headroom_v73 import allocate
base=p[:16];shift=np.linspace(-.1,.1,16)[:,None]
assert allocate(base.tolist(),48)['widths']==allocate((base+shift).tolist(),48)['widths']
# Deterministic fit and prompt-disjoint cross-fitting on a small numerical example.
X=np.column_stack([np.ones(len(ids)),ids/16,offsets/24]);m=fit_model(X,Y,ids,[.1,1.]);n=fit_model(X,Y,ids,[.1,1.])
assert m['lambda']==n['lambda'];np.testing.assert_array_equal(m['coef'],n['coef'])
print('PASS: allocation changes with width-dependent feedback; known covariance recovery; counterfactual/future guards; width-independent cancellation; deterministic fit')
