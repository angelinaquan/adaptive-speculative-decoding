"""Experimental vLLM 0.33.0 worker extension; NOT yet GPU validated.

Reuses upstream IRLS updates, replacing the input feature with the pretrained
DSpark confidence-head logit. Does not alter the sampler or target model.
Select with worker_extension_cls='head_calibration_v17.HeadCalibrationWorker'.
"""
import torch
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import OnlineAcceptanceEstimator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.triton_utils import tl,triton

_instances = []


@triton.jit
def _predict_head_kernel(raw_ptr,slots_ptr,slope_ptr,bias_ptr,features_ptr,predictions_ptr,out_ptr,
                         K:tl.constexpr,RAW_STRIDE:tl.constexpr,SLOT_STRIDE:tl.constexpr,
                         FEATURE_STRIDE:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    slot=tl.load(slots_ptr+row*SLOT_STRIDE)
    step=tl.arange(0,BLOCK)
    if slot>=0:
        raw=tl.load(raw_ptr+row*RAW_STRIDE+step,mask=step<K,other=.5)
        raw=tl.minimum(tl.maximum(raw,1e-6),1.-1e-6)
        feature=tl.log(raw)-tl.log(1.-raw)
        slope=tl.load(slope_ptr)
        bias=tl.load(bias_ptr+step,mask=step<K,other=0.)
        prediction=tl.sigmoid(slope*feature+bias)
        tl.store(features_ptr+slot*FEATURE_STRIDE+step,feature,mask=step<K)
        tl.store(predictions_ptr+slot*FEATURE_STRIDE+step,prediction,mask=step<K)
        tl.store(out_ptr+row*K+step,prediction,mask=step<K)
    else:
        # Captured graphs use -1 for padded requests. Python indexing would
        # incorrectly write these rows into the final persistent request slot.
        tl.store(out_ptr+row*K+step,0.,mask=step<K)


class HeadCalibrator(OnlineAcceptanceEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slope.fill_(1.)
        self.intercepts.zero_()
        self.learning = False  # Explicitly enable only after initialization/warmup.
        # observed count, raw SSE, calibrated SSE, raw logloss, calibrated logloss
        self.scores = torch.zeros(5, device=self.device, dtype=torch.float64)
        _instances.append(self)

    def predict_head(self, raw, slots):
        if raw.is_cuda:
            predicted=torch.empty_like(raw,dtype=torch.float32)
            _predict_head_kernel[(raw.shape[0],)](raw,slots,self.slope,self.intercepts,
                self.features,self.predictions,predicted,K=self.num_speculative_steps,
                RAW_STRIDE=raw.stride(0),SLOT_STRIDE=slots.stride(0),
                FEATURE_STRIDE=self.features.stride(0),BLOCK=triton.next_power_of_2(self.num_speculative_steps))
            return predicted
        feature = torch.logit(raw.float().clamp(1e-6, 1-1e-6))
        predicted = torch.sigmoid(self.slope*feature+self.intercepts)
        valid=slots>=0
        self.features[slots[valid]] = feature[valid]
        self.predictions[slots[valid]] = predicted[valid]
        predicted[~valid]=0.
        return predicted

    def observe(self, slots, num_sampled, num_rejected):
        # Same observation rule as upstream: accepted prefix and first rejection
        # only; a truncated/unverified suffix supplies no conditional labels.
        accepted = (num_sampled-1).clamp_min(0)
        admitted = accepted+num_rejected
        position = torch.arange(self.num_speculative_steps, device=self.device)[None,:]
        observed = (position <= accepted[:,None]) & (position < admitted[:,None])
        label = (position < accepted[:,None]).float()
        raw = torch.sigmoid(self.features[slots])
        prediction = self.predictions[slots]
        def logloss(prob):
            prob = prob.clamp(1e-6,1-1e-6)
            return -(label*prob.log()+(1-label)*torch.log1p(-prob))
        mask = observed.to(torch.float64)
        # Score predictions stored at draft time, BEFORE any parameter update.
        self.scores.add_(torch.stack([mask.sum(),
            (mask*(raw-label).square()).sum(), (mask*(prediction-label).square()).sum(),
            (mask*logloss(raw)).sum(), (mask*logloss(prediction)).sum()]))
        if self.learning:
            super().step(slots,num_sampled,num_rejected)


def _install():
    if getattr(DSparkSpeculator,'_head_calibration_v17_installed',False):
        return
    original_load = DSparkSpeculator.load_draft_model
    original_sample = DSparkSpeculator._sample_sequential
    original_observe = DSparkSpeculator.observe_verification

    def load(self,*args,**kwargs):
        model = original_load(self,*args,**kwargs)
        assert self.enable_adaptive_verification and self.use_confidence_head
        assert self.num_speculative_steps == 7, 'Pilot is restricted to the released block7 head'
        assert self._draft_topk is None, 'Top-k draft approximation is outside this pilot'
        self.head_calibration_v17 = HeadCalibrator(self.max_num_reqs,self.num_speculative_steps,self.device)
        return model

    def sample(self,num_reqs,head_hidden):
        original_sample(self,num_reqs,head_hidden)
        slots = self.sample_idx_mapping[:num_reqs*self.num_speculative_steps].view(num_reqs,self.num_speculative_steps)[:,0]
        # This feature depends on the backbone and previous draft token, not on
        # the candidate token at the position being admitted.
        raw = self.draft_token_confidence_probs[:num_reqs]
        calibrated = self.head_calibration_v17.predict_head(raw,slots)
        self.draft_token_confidence_probs[:num_reqs] = calibrated

    def observe(self,slots,num_sampled,num_rejected):
        original_observe(self,slots,num_sampled,num_rejected)
        self.head_calibration_v17.observe(slots,num_sampled,num_rejected)

    DSparkSpeculator.load_draft_model = load
    DSparkSpeculator._sample_sequential = sample
    DSparkSpeculator.observe_verification = observe
    DSparkSpeculator._head_calibration_v17_installed = True


class HeadCalibrationWorker:
    def calibration_control_v17(self, *, learning=None, slope=None, intercepts=None,
                               reset_statistics=False, refit_interval=None):
        """Call only between generate requests, outside measured timing windows."""
        assert len(_instances)==1, f'Expected one calibrator, found {len(_instances)}'
        state = _instances[0]
        if learning is not None:
            state.learning = bool(learning)
        if slope is not None:
            assert float(slope)>=0
            state.slope.fill_(float(slope))
        if intercepts is not None:
            assert len(intercepts)==state.num_speculative_steps
            state.intercepts.copy_(torch.tensor(intercepts,device=state.device))
        if refit_interval is not None:
            assert int(refit_interval)>0
            state.REFIT_INTERVAL = int(refit_interval)
        if reset_statistics:
            for tensor in [state.info,state.grad,state.counts,state.scores]:
                tensor.zero_()
            state._steps_since_refit = 0
            state._refits = 0
        return dict(slope=state.slope.cpu().tolist(),intercepts=state.intercepts.cpu().tolist(),
                    scores=state.scores.cpu().tolist(),refits=state._refits,
                    learning=state.learning,refit_interval=state.REFIT_INTERVAL)


_install()
