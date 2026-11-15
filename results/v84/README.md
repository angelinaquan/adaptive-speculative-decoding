# Width-correlated feedback correction (pending)

This repairs a structural limitation of v82: a width-independent additive residual contributes the same constant for every allocation and can only affect choices through clipping. v82 remains archived as a negative result.

The revised method estimates a persistent request-effect covariance from cross-time products of training cross-fitted prediction residuals. A Gaussian posterior update uses feedback only at the selected width; covariance transfers that correction to the other widths. PSD projection and a noise floor stabilize updates. Covariance scale/decay are fitted using training replay, which is explicitly a training objective, not an unbiased CV estimate. No validation outcomes choose parameters.

The frozen plan was written before any v83 labels were inspected. Primary:384training/96validation prompts. Secondary learning curves:96/192training prompts. Matched frozen predictor, training-mean allocator, and fixed3 controls use identical groups and draft budgets. A positive reward screen is insufficient: fresh evaluation and measured end-to-end throughput against tuned native and frozen controls must follow.

Synthetic mechanism and information-flow tests pass. Job3064611 is submitted with afterok dependency on label-collection job3064552; status/results have not been inspected. No report changes or empirical performance claim.

## Completed result: primary gate failed

Job3064611 completed0:0. [W&B](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/3nye1siq).

| Train prompts/domain | Frozen reward gain vs fixed3 | Correlated reward gain vs fixed3 | Correlated vs frozen |
|---|---:|---:|---:|
|32|-0.475%|-0.475%|0.000%|
|64|-1.010%|-0.891%|+0.120%|
|128 (primary)|+0.535%|+0.654%|+0.118%|

The primary raw rewards are fixed3=1683, frozen=1692, correlated=1694. Training-mean allocation matches fixed3 at all sizes. Local reproduction matches every width choice, reward and selected parameter; predictions agree within1e-10. The candidate fails its10% reward gate and should not proceed to serving validation on this evidence. No throughput, significance, or novelty claim. Do not tune more parameters on these validation outcomes. Report unchanged.
