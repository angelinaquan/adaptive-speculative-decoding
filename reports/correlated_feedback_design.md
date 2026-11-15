# Correlated feedback correction: candidate, not a result

The proposed addition is a request-specific residual model over the rewards of different draft lengths. The frozen context predictor supplies an eight-dimensional mean. Training residuals at different offsets estimate which length-specific errors persist within a request. Observing reward at the selected length updates the residual vector through its learned covariance. The next budgeted allocation uses the corrected mean.

This is a standard Gaussian random-effects/filtering construction applied to speculative-decoding feedback. The filtering equations are not a new statistical algorithm. A contribution would require evidence that this particular feedback transfer improves decisions and actual serving performance over strong controls.

## Closest verified prior work

- [ForeSpark](https://arxiv.org/html/2611.13442v1), sections 4.2–4.5, already predicts per-length acceptance from context/anchor features, jointly schedules draft and verification work, and calibrates the estimates. Its non-causal drafts need separate labels for each width. Thus context features, per-length prediction, calibration, and batch allocation are prior art. Our candidate addition is a request-specific cross-width residual update from selected-width feedback; the inspected sections do not establish whether this exact update appears elsewhere. The current ridge control is not a full ForeSpark reproduction.
- [BanditSpec](https://proceedings.mlr.press/v267/hou25h.html) formulates online configuration selection as a bandit problem and studies UCBSpec/EXP3Spec with stopping-time regret. Online feedback-driven adaptation is therefore not new. Our greedy posterior-mean allocator currently has no exploration or regret guarantee; it must not be described as a new bandit algorithm or as improving BanditSpec without a matched implementation.
- [TurboSpec](https://arxiv.org/abs/2406.14066) is direct prior work on closed-loop speculation control. A claim that runtime feedback or load-aware control itself is new would be incorrect.

The search also surfaced secondary descriptions without a verified primary paper. Those were not used as evidence of a method or benchmark result. This search does not prove novelty or establish workshop readiness.

## Why v82 was structurally weak

If every width for request i gets the same additive correction c_i, then any complete allocation adds exactly sum_i c_i to the objective. Its maximizer is unchanged. The v82 implementation could change allocations only because it clipped corrected predictions to width-dependent reward bounds. Its negative results remain valid for that implemented rule, but they are a weak test of whether feedback contains useful allocation information.

The v84 correction is a vector. With mean m, covariance C, selected width w, observed residual e, and observation variance R_w, the gain is C[:,w]/(C[w,w]+R_w). The updated mean is m + gain*(e-m[w]); the covariance update subtracts the corresponding rank-one matrix. Cross-time covariance excludes same-offset residual products to distinguish persistent request effects from observation noise. PSD projection and a noise floor stabilize this approximation. Bounded integer rewards are not Gaussian, so posterior uncertainty is a modeling device, not a calibrated confidence interval.

## Evaluation and interpretation

The code/plan were frozen before inspecting v83 labels. Training uses384 prompts and validation 96 prompts; smaller training sizes are secondary learning curves. No validation outcome chooses regularization, residual scale, decay, or training size. Residual settings maximize a training replay objective using cross-fitted baseline predictions; this is fitting, not an unbiased estimate of performance.

Controls are fixed3, training-mean allocation, and the identical frozen context predictor. All share prompt groups and draft budgets. Tests cover selected-feedback-only access, current/future outcome poisoning, deterministic model fitting, known covariance recovery, and the constant-shift invariance above.

The primary gate is at least 10% validation accepted-token reward over fixed3 and a gain over the frozen predictor. It is only a gate: equal total draft tokens do not imply equal latency, teacher offsets are exogenous, and FP32 training trajectories differ from BF16 serving. A positive result must be followed by fresh held-out closed-loop evaluation, controller overhead measurements, tuned native widths and matched adaptive controls, and repeated balanced timing orders. The requested 10% throughput result remains unproven.

## What information can change an allocation?

Under the Gaussian surrogate, let C be the current residual covariance and let observing width w have noise variance R_w. For two candidate widths d and e, their residual difference has variance

    V(d,e) = C[d,d] + C[e,e] - 2*C[d,e].

After one observation at w, the reduction is

    V_before(d,e) - V_after(d,e)
      = (C[d,w] - C[e,w])^2 / (C[w,w] + R_w).

This follows by substituting the rank-one covariance update. It is nonnegative, and is zero when feedback correlates equally with both candidate widths. In particular, a pure common-mode request effect has C = sigma^2 * 1*1^T and provides no information about width differences. This is a stronger explanation of v82's weakness than merely saying it had too few observations. It also identifies a falsifiable requirement for v84: persistent residual covariance must have a component outside the common-mode direction. This diagnostic is model-based; it does not validate the Gaussian assumption or establish throughput gain.

For any fixed feasible allocation set, define true expected per-request rewards r_i(d) and estimates rhat_i(d). If the implemented allocator exactly maximizes the estimated sum and max_d |r_i(d)-rhat_i(d)| <= epsilon_i, then its expected reward gap to the true optimal allocation is at most 2*sum_i epsilon_i. Proof: add and subtract estimated rewards for the optimal and selected allocations; the difference between their estimated sums is nonpositive, and the two remaining error sums are bounded by sum_i epsilon_i each. This elementary bound is not a novelty claim and our experiments have not established such error bounds.

Both statements concern reward at a fixed resource constraint. Neither controls wall-clock latency, predictor overhead, or changing generation trajectories. The empirical serving experiment remains necessary.
