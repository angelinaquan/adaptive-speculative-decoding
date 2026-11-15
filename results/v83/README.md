# Larger training/validation collection (pending)

Job3064552 was submitted to collect independent-width acceptance labels on480prompts:384training and96validation, balanced across conversation/math/code. All prompts come from the historical v77 profiling pool, exclude the47/48prompt v73 screen, and are distinct across splits. They are not new inference workloads; per-state training labels have not previously been used by these predictors. No test outcomes are evaluated.

The fixed plan in experiments/labels_v83_plan.json specifies training-size comparisons, frozen/train-mean/fixed3 controls, causal residual correction, and the gate for further serving work. All model/update selection must use training prompts alone. This addresses the small-data limitation of v75/v81/v82; it is not evidence that scaling will succeed.

The collector inherits v73 checkpoint, causal-feature, independent-verification, and target replay checks. It stops on failures and retains EOS-censored prompts. An independent exporter reconstructs every accepted prefix from saved token IDs. Labels use FP32 HF teacher trajectories, not BF16 serving; future end-to-end validation must establish transfer, output correctness, and actual latency including overhead.

Status: submitted, not yet inspected. No performance conclusion or report change.

## Completed result

Job3064552 completed0:0 in29m16s. All480prompts processed,478usable,2767states; early-EOS prompts1and16 retained. Train2218states, validation549states. Independent local reconstruction checks all22,136 width labels and source/plan/manifest hashes. [W&B](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/woa3n7n4).
