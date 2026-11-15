"""Prompt-block split-conformal candidate coverage. CPU reference, not a serving implementation."""
import math
import numpy as np

def row_score(approx,exact,k):
 approx=np.asarray(approx,float);exact=np.asarray(exact,float)
 assert approx.ndim==exact.ndim==1 and approx.shape==exact.shape
 assert np.isfinite(approx).all() and np.isfinite(exact).all() and 1<=k<=len(approx)
 top=np.argsort(-exact,kind='stable')[:k]
 kth=np.partition(approx,len(approx)-k)[len(approx)-k]
 return max(0.,float(kth-approx[top].min()))

def calibrate(prompt_scores,alpha):
 assert 0<alpha<1 and len(prompt_scores)>0
 maxima=[]
 for scores in prompt_scores:
  scores=np.asarray(scores,float);assert scores.ndim==1 and len(scores)>0 and np.isfinite(scores).all() and (scores>=0).all()
  maxima.append(float(scores.max()))
 rank=math.ceil((len(maxima)+1)*(1-alpha))
 q=float('inf') if rank>len(maxima) else sorted(maxima)[rank-1]
 return {'threshold':q,'calibration_prompts':len(maxima),'rank':rank,'alpha':alpha,'unit':'Maximum score over a prespecified set of rows within each prompt'}

def candidates(approx,k,threshold):
 approx=np.asarray(approx,float);assert approx.ndim==1 and np.isfinite(approx).all() and 1<=k<=len(approx) and threshold>=0
 kth=np.partition(approx,len(approx)-k)[len(approx)-k]
 if np.isinf(threshold):return np.arange(len(approx))
 # Conservative FP64 boundary padding for the reference implementation. GPU kernels
 # need their own numerical audit; conformal coverage does not imply bitwise parity.
 padding=4*np.finfo(float).eps*(abs(kth)+abs(threshold)+1)
 return np.flatnonzero(approx>=kth-threshold-padding)

def rerank(approx,exact,k,threshold):
 # Reference implementation has full exact logits for checking. A serving implementation
 # must compute them only for selected rows; this function provides no speed evidence.
 subset=candidates(approx,k,threshold);exact=np.asarray(exact,float)
 return subset[np.argsort(-exact[subset],kind='stable')[:k]]
