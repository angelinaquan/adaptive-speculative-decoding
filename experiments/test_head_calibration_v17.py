"""CPU gates for feature mapping, censored labels, and pre-update scoring.

Does not test CUDA capture or the upstream GPU IRLS kernels.
"""
import json
from unittest.mock import patch
import torch
from head_calibration_v17 import HeadCalibrator, OnlineAcceptanceEstimator

state = HeadCalibrator(4,7,torch.device('cpu'))
slots = torch.tensor([2,0],dtype=torch.int64)
raw = torch.tensor([[.7,.6,.5,.4,.3,.2,.1],[.2,.3,.4,.5,.6,.7,.8]])
prediction = state.predict_head(raw,slots)
assert torch.allclose(prediction,raw,atol=1e-6)
assert torch.allclose(state.predictions[slots],raw,atol=1e-6)
# -1 padding must not write into the last persistent request slot.
last_features=state.features[-1].clone();last_predictions=state.predictions[-1].clone()
state.predict_head(torch.cat([raw,torch.full((2,7),.99)]),torch.tensor([2,0,-1,-1]))
assert torch.equal(state.features[-1],last_features)
assert torch.equal(state.predictions[-1],last_predictions)
# Request 1: one accepted, one first rejection, suffix unobserved.
# Request 2: three verified drafts all accepted, remaining positions censored.
sampled = torch.tensor([2,4]); rejected = torch.tensor([1,0])
state.observe(slots,sampled,rejected)
expected_sse = (.7-1)**2+.6**2+(.2-1)**2+(.3-1)**2+(.4-1)**2
assert state.scores[0].item()==5
assert abs(state.scores[1].item()-expected_sse)<1e-6
assert abs(state.scores[2].item()-expected_sse)<1e-6
# Change only unobserved suffixes: observed score must remain unchanged.
state.scores.zero_()
raw2 = raw.clone();raw2[0,2:]=.99;raw2[1,3:]=.01
state.predict_head(raw2,slots);state.observe(slots,sampled,rejected)
assert abs(state.scores[2].item()-expected_sse)<1e-6
# Simulate an update that would make rescored probabilities very different.
# Scores must still reflect the predictions stored before feedback arrived.
state.scores.zero_();state.predict_head(raw,slots);state.learning=True
def fake_update(self,*args):
    assert abs(self.scores[2].item()-expected_sse)<1e-6
    self.intercepts.fill_(10.)
    self.predictions.zero_()
with patch.object(OnlineAcceptanceEstimator,'step',fake_update):
    state.observe(slots,sampled,rejected)
assert abs(state.scores[2].item()-expected_sse)<1e-6
gpu_padding=False
if torch.cuda.is_available():
    gpu=HeadCalibrator(4,7,torch.device('cuda'))
    padded=torch.cat([raw,torch.full((2,7),.99)]).cuda()
    gpu_slots=torch.tensor([3,0,-1,-1],device='cuda')
    out=gpu.predict_head(padded,gpu_slots)
    assert torch.allclose(out[:2].cpu(),raw,atol=1e-6)
    assert torch.allclose(gpu.predictions[3].cpu(),raw[0],atol=1e-6)
    assert torch.allclose(torch.sigmoid(gpu.features[3]).cpu(),raw[0],atol=1e-6)
    assert torch.all(out[2:]==0)
    gpu.observe(gpu_slots[:2],sampled.cuda(),rejected.cuda())
    assert abs(gpu.scores[2].item()-expected_sse)<1e-6
    gpu_padding=True
print(json.dumps({'passed':True,'gpu_padding_kernel_checked':gpu_padding,'checks':['identity feature map','persistent request-slot mapping','padding cannot overwrite the last live slot',
    'unverified suffix is not labeled negative','score before parameter update'],
    'scope':'CPU semantics only; no GPU/capture/performance claim'}))
