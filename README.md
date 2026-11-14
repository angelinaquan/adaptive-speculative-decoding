# Adaptive verification for speculative decoding

Experiments by **Angelina Quan** on confidence-based verification lengths for an open-weight LLM.

**Updated validation result:** after testing every fixed proposal width from 1 through 7 on all 32 validation prompts, adaptive verification is **2.22% faster than the best fixed width at batch 8**, but **11.63% slower at batch 1** and **11.17% slower at batch 32**. These are descriptive validation measurements, not statistically established held-out gains. The earlier 11–15% improvements below compare against fixed length 7, not the strongest tuned baseline.

This repository reproduces **vLLM's existing adaptive-verification implementation** with a released DSpark drafter. It does not claim that the scheduler is new, or that online calibration caused these gains. The broader project asks whether online confidence calibration under workload shifts can improve verification decisions; that contribution remains unproven.

**Held-out correctness update:** the fresh 96-prompt run stopped before speculative timing: fixed width 3 at batch 8 differed from ordinary greedy decoding on one prompt (index 88, GSM8K item 901), starting at output token 26. Both outputs reached the 128-token cap. The cause is under investigation; no held-out speedup is established. [Preserved outputs](results/v16/incomplete/) · [failed correctness run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/nue79syr).

The diagnostic reproduced the same single mismatch in two complete repeats. At the first divergence, the target log-probability gap between ` a` and ` enough` changes from +0.000469 in ordinary decoding to −0.002220 in speculative execution. This is consistent with numerical sensitivity between execution paths, but does not establish the root cause or satisfy the strict output-equality gate. [Diagnostic outputs and summary](results/v19/).

## Completed results

Throughput is total generated tokens divided by total measured wall time across three timing repetitions. Higher is better. The calls include prefill and decoding; model loading, compilation, and warmup are excluded.

| Concurrent requests | Ordinary decoding | Fixed length 7 | Adaptive verification | Adaptive vs. fixed 7 | Adaptive vs. ordinary |
|---:|---:|---:|---:|---:|---:|
| 1 | 187.43 tok/s | 184.30 tok/s | 212.13 tok/s | **+15.10%** | +13.18% |
| 8 | 567.87 tok/s | 1,010.31 tok/s | 1,107.50 tok/s | **+9.62%** | +95.03% |
| 32 | 1,885.48 tok/s | 1,641.21 tok/s | 1,824.39 tok/s | **+11.16%** | **−3.24%** |

All batch sizes are shown, including the negative comparison with ordinary decoding at batch 32. The first fixed-7 timing at batch 8 was slower than the other two (1.078 s versus about 0.910 s); it is retained in the table. Thus the pooled +9.62% at batch 8 should not be read as a stable effect size.

- [Ordinary-decoding W&B run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/hossc6i8)
- [Fixed-7 W&B run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/6pb6wa6c)
- [Adaptive-verification W&B run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/itgkdqhi)
- [GPU sampling-correctness W&B run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/hvf8or17)
- [Raw measurements and output token IDs](results/v14/raw/) · [computed summary](results/v14/summary.json)

## Stronger baseline sweep (v15)

All methods processed the same 32 validation prompts at each batch size, with two full-workload warmups and three timing repetitions. All nine methods matched the ordinary-decoding output tokens. Fixed widths were selected on these validation measurements only; engine order was not balanced, so this sweep is for selection rather than significance testing.

| Batch | Best fixed width | Best fixed throughput | Adaptive throughput | Adaptive vs. best fixed |
|---:|---:|---:|---:|---:|
| 1 | 5 | 337.41 tok/s | 298.17 tok/s | **−11.63%** |
| 8 | 3 | 1,056.36 tok/s | 1,079.83 tok/s | **+2.22%** |
| 32 | 1 | 2,039.42 tok/s | 1,811.65 tok/s | **−11.17%** |

This is evidence that the original fixed-7 comparison was insufficient to establish a general advantage. It is not evidence of an improvement from online confidence-head recalibration. The smaller configured widths also reduce drafting work; adaptive verification at width 7 only trims verification.

[All raw v15 measurements](results/v15/raw/) · [full summary and W&B links](results/v15/summary.json)

```bash
python experiments/analyze_serving_v15.py results/v15/raw --output /tmp/serving-v15-audit.json
```

## What is the idea?

Speculative decoding generates draft tokens with a smaller model, then checks them with the target model. Verifying every draft position can waste compute when later tokens are unlikely to survive rejection.

The released drafter has a pretrained confidence head. vLLM's adaptive scheduler uses estimated prefix-survival probabilities and profiled execution costs to choose a verification budget and allocate slots across requests. This changes **how much to verify after drafting**; it does not magically know an optimal draft length in advance or eliminate all drafting work. See the [vLLM adaptive-verification documentation](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/).

## Setup and correctness

The completed probe used:

- NVIDIA H200, driver 580.178.04; vLLM 0.33.0; PyTorch 2.14.0+cu130. [Full installed package list](docs/environment-v14.txt).
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- [DSpark Qwen3-4B block-7 drafter](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7), revision `3457dff1417cb84927f6098a5fcb7cee85c934b7`.
- Released checkpoint weights promoted to **FP32**, TF32 disabled, Triton attention, CUDA graphs, prefix caching disabled, non-thinking chat template.
- Greedy generation, a 128-token output cap, and batches drawn from 32 validation prompts interleaved across Alpaca, GSM8K, and HumanEval. The original probe uses the first 1, 8, or 32 prompts at the corresponding batch size.

Two checks passed before timing conclusions:

1. Fixed-7 and adaptive generation matched ordinary greedy decoding token-for-token at each evaluated batch size. Repeated timed outputs matched as well. These output IDs are committed so this comparison can be independently checked.
2. The actual installed GPU rejection kernel was tested against a known three-state Markov target with 20,000 draws each for fixed and causally chosen variable proposal lengths. Total-variation deviations were 0.01325 and 0.01385, with maximum standardized cell errors of 2.17 and 2.02. Both passed the predeclared numerical thresholds.

These checks have limits: greedy agreement is not a proof of full-model stochastic equivalence; the finite-state kernel test does not validate every cache or scheduler path. Earlier work with a different BF16 backend failed a numerical check, so these FP32 results must not be relabeled as BF16 results.

## Reproduce

Use a CUDA-capable Linux GPU environment with enough memory for this FP32 configuration. The measured configuration was an H200; other hardware can change the ranking.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
wandb login
bash scripts/run_probe.sh v14
```

The experiment scripts use the original W&B entity/project, `aquan-world-model/18657-speculative-pilot`. If running under another account, change the `wandb.init` entity/project in the scripts, or use `WANDB_MODE=offline`. No credentials are included. Model paths are created locally and ignored by Git.

The wrapper puts temporary files and caches in `.local/`; set `SERVING_SCRATCH` to a suitable local disk to override it. Outputs go to `$SERVING_SCRATCH/probe-results`. These wrapper paths differ from the original cluster-specific launcher, but preserve its model and inference settings; inference initialization is excluded from timing.

To audit the committed measurements without a GPU or extra packages:

```bash
python scripts/verify_results.py
```

## Follow-up experiments and current status

The v15 validation sweep is completed and audited above. The repository also includes the v16 follow-up code; no held-out result is claimed yet.

- **v15 (completed):** run every fixed proposal width from 1 through 7, ordinary decoding, and adaptive verification. Evaluate all 32 validation prompts at every batch size, rather than one prompt at batch 1. A shorter configured proposal width may change the parallel drafter computation, not only truncate verification.
- **v16:** 96 frozen fresh prompts, 32 per domain; batch 8 is the predeclared primary comparison, with batches 1 and 32 secondary. Choose fixed widths using v15 validation only. Compare methods in three balanced engine-order rounds and retain per-batch timings for paired analysis. The driver declines to launch fresh evaluation if adaptive fails to beat both validation comparators at the primary batch size.

```bash
bash scripts/run_probe.sh v15
python experiments/analyze_serving_v15.py \
  .local/probe-results-v15 --output .local/serving_v15_summary.json
# After inspecting validation and preserving the frozen test plan:
bash scripts/run_probe.sh v16
```

The wrapper retains the cache and numerical settings across these experiments. The held-out runner and manifest are included for transparency, but no held-out performance result is claimed in this revision. The shell wrapper assumes the default `.local` output directory in the analysis example above; substitute your `SERVING_SCRATCH` path if you changed it.

## Sources and attribution

Inference and adaptive scheduling are provided by [vLLM](https://github.com/vllm-project/vllm). The released drafter and pretrained confidence head come from [DeepSpec](https://github.com/deepseek-ai/DeepSpec). The prompt selection uses the Alpaca, GSM8K, and HumanEval files in DeepSpec's evaluation datasets, from commit `005e03b81cec38b7da6399833d609ee89a2587f2`. Dataset hashes and held-out selection provenance are recorded in [the v16 plan](experiments/serving_v16_plan.json). Upstream model, software, and dataset licenses remain applicable.
