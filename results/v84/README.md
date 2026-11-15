# Width-correlated feedback correction (pending)

This repairs a structural limitation of v82: a width-independent additive residual contributes the same constant for every allocation and can only affect choices through clipping. v82 remains archived as a negative result.

The revised method estimates a persistent request-effect covariance from cross-time products of training cross-fitted prediction residuals. A Gaussian posterior update uses feedback only at the selected width; covariance transfers that correction to the other widths. PSD projection and a noise floor stabilize updates. Covariance scale/decay are fitted using training replay, which is explicitly a training objective, not an unbiased CV estimate. No validation outcomes choose parameters.

The frozen plan was written before any v83 labels were inspected. Primary:384training/96validation prompts. Secondary learning curves:96/192training prompts. Matched frozen predictor, training-mean allocator, and fixed3 controls use identical groups and draft budgets. A positive reward screen is insufficient: fresh evaluation and measured end-to-end throughput against tuned native and frozen controls must follow.

Synthetic mechanism and information-flow tests pass. Job3064611 is submitted with afterok dependency on label-collection job3064552; status/results have not been inspected. No report changes or empirical performance claim.
