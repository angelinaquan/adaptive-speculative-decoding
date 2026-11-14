"""Gupta & Ramdas (ICML 2023), Algorithm 1 and Eq. 11.
One independent binary calibrator per prefix position. No automatic monotonicity
or adversarial calibration guarantee is claimed for this application.
"""
import numpy as np

EPS = 1e-5  # released utils.logit convention

def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -100, 100)))

def logit(p):
    p = np.clip(p, EPS, 1-EPS)
    return np.log(p/(1-p))

def metric_project(z, A, radius=100.):
    """Exact A-norm projection onto Euclidean ball, to bisection precision."""
    if np.linalg.norm(z) <= radius:
        return z.copy()
    values, vectors = np.linalg.eigh(A)
    coordinates = vectors.T @ z
    def candidate(lam):
        return values*coordinates/(values+lam)
    lo, hi = 0., max(float(values.max()), 1.)
    while np.linalg.norm(candidate(hi)) > radius:
        hi *= 2
    for _ in range(80):
        mid = (lo+hi)/2
        if np.linalg.norm(candidate(mid)) > radius:
            lo = mid
        else:
            hi = mid
    return vectors @ candidate(hi)

def isotonic_decreasing(p):
    """Unweighted Euclidean projection onto nonincreasing vectors (PAV)."""
    levels, counts = [], []
    for x in p:
        levels.append(float(x)); counts.append(1)
        while len(levels)>1 and levels[-2]<levels[-1]:
            n=counts[-2]+counts[-1]
            v=(levels[-2]*counts[-2]+levels[-1]*counts[-1])/n
            levels[-2:]=[v];counts[-2:]=[n]
    return np.repeat(levels, counts)

class OPS:
    def __init__(self, positions=7, convention='paper'):
        assert convention in ('paper','released_code')
        self.convention=convention
        self.theta=np.zeros((positions,2))
        if convention=='paper':self.theta[:,0]=1.
        self.A=np.repeat((100.*np.eye(2))[None],positions,axis=0)
        self.count=np.zeros((positions,10))
        self.total=np.zeros((positions,10))
        if convention=='released_code':
            self.count[:]=1.
            self.total[:]=np.linspace(.1,1.,10)-.05/10  # literal upstream convention
        self.projections=0
    def predict(self, base):
        x=np.stack([logit(base),np.ones(len(base))],axis=1)
        p=sigmoid(np.sum(self.theta[:len(base)]*x,axis=1))
        if self.convention=='paper':
            bins=np.minimum((p*10).astype(int),9)
        else:
            bins=np.sum(p[:,None]>np.linspace(.1,1.,10),axis=1)
        n=self.count[np.arange(len(base)),bins]
        totals=self.total[np.arange(len(base)),bins]
        tops=np.divide(totals,n,out=(bins+.5)/10.,where=n>0)
        return p,tops,(x,p,bins)
    def update(self, context, y):
        x,p,bins=context;n=len(y);idx=np.arange(n)
        # All predictions are emitted before this block's labels are exposed.
        self.count[idx,bins]+=1
        self.total[idx,bins]+=y
        g=(p-y)[:,None]*x
        self.A[:n]+=g[:,:,None]*g[:,None,:]
        A=self.A[:n];det=A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]
        v=np.stack([A[:,1,1]*g[:,0]-A[:,0,1]*g[:,1],
                    A[:,0,0]*g[:,1]-A[:,1,0]*g[:,0]],axis=1)/det[:,None]
        self.theta[:n]-=v/.1
        for k in np.where(np.sum(self.theta[:n]**2,axis=1)>100**2)[0]:
            self.theta[k]=metric_project(self.theta[k],self.A[k]);self.projections+=1
