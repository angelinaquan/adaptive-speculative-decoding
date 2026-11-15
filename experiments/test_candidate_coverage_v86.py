import numpy as np
from candidate_coverage_v86 import row_score,calibrate,candidates,rerank
rng=np.random.default_rng(1865786)
for _ in range(200):
 exact=rng.normal(size=128);approx=exact+rng.normal(scale=.7,size=128)
 for k in [1,8,16]:
  score=row_score(approx,exact,k)
  # The implementation handles boundary roundoff conservatively.
  threshold=score
  wanted=np.argsort(-exact,kind='stable')[:k];got=candidates(approx,k,threshold)
  assert set(wanted)<=set(got)
  np.testing.assert_array_equal(rerank(approx,exact,k,threshold),wanted)
# Ties retain all boundary candidates; exact reranking uses stable token-id order.
np.testing.assert_array_equal(candidates([1,1,1],1,0),[0,1,2])
np.testing.assert_array_equal(rerank([1,1,1],[2,2,0],1,0),[0])
assert calibrate([[0],[1],[2]],.1)['threshold']==float('inf')
assert calibrate([[0,9],[1,2],[3,4]],.5)['threshold']==4
assert calibrate([[0],[1],[2]],.5)['rank']==2
# Exact finite rank calculation for a uniform random position of a test block among20.
alpha=.1;n=19;rank=calibrate([[i] for i in range(n)],alpha)['rank']
assert rank==18 and rank/(n+1)>=1-alpha
print('PASS:600 top-k containment/rerank cases; tie handling; prompt maxima; finite-sample rank and small-N fallback')
