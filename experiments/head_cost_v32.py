"""Calibration plus measured cost table without diagnostic scoring overhead."""
import head_calibration_v17 as head
from cost_only_v26 import CostOnlyWorker
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import OnlineAcceptanceEstimator

def observe_without_scores(self,slots,num_sampled,num_rejected):
 if self.learning:
  OnlineAcceptanceEstimator.step(self,slots,num_sampled,num_rejected)
head.HeadCalibrator.observe=observe_without_scores

class HeadCostWorker(CostOnlyWorker,head.HeadCalibrationWorker):
 pass
