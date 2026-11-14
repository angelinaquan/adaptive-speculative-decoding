# Reliable draft-length selection for batched speculative decoding

Literature review and project framing, November 14, 2026. Author: Angelina Quan.

## Research question

How much profiling is needed to choose a fast draft length reliably, and does that choice transfer across prompt domains? This is an empirical extension and tuning study, not a claim that batch-aware speculative decoding is new.

The course assignment permits empirical questions and proposed initial experiments. The statistical component here is selection under noisy runtime measurements, held-out performance, and uncertainty in the resulting throughput gains.

## Closest prior work

1. **Su, Giannoula, and Pekhimenko (2023), The Synergy of Speculative Decoding and Batching in Serving Large Language Models.** [Paper](https://arxiv.org/html/2310.18813v1). This is the most direct antecedent. It studies batch size and speculation length across models and GPUs, finds that larger batches often favor shorter speculation, and profiles configurations to build a batch-size-to-length mapping. Both the qualitative hypothesis and a simple lookup-table selection rule are already present. Our project must cite this as the starting point, not present either as a new method.

2. **Liu et al., TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput (2024 preprint; 2025 revision).** [Paper](https://arxiv.org/abs/2406.14066). TurboSpec profiles the execution environment and uses runtime feedback to adjust speculation. It addresses both acceptance and execution cost, making it relevant to any proposed cost-based adaptive controller. Offline batch-size measurements alone do not replicate its live-serving claims.

3. **Hou et al., BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms (ICML 2025).** [Paper](https://proceedings.mlr.press/v267/hou25h.html). BanditSpec formulates configuration selection as online learning and provides UCBSpec and EXP3Spec with stopping-time regret analysis. Simply adding bandit selection would not establish novelty. A complete implementation is needed before reporting a numerical comparison under that name.

4. **Song et al., ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding (2026).** [Paper](https://arxiv.org/abs/2611.13442). ForeSpark predicts acceptance for candidate draft lengths from context and jointly allocates draft and verification budgets. This is especially close to our modern parallel-drafter setting. We use its runtime; native fixed-width experiments are configuration controls, not a reproduction of its learned scheduler or evidence that our method outperforms ForeSpark.

## What existing evidence supports

- v70: at batch 64, forced width 3 was 16.38% faster than forced width 8 on 192 reused prompts. This was exploratory configuration tuning.
- v72: the native width sweep recovered the benefit. Custom forced width 3 differed from native width 3 by only +0.056%. There is no demonstrated custom-runtime advantage.
- v69: width 8 won the small batch-8 forced-width screen. This motivates checking a batch-by-width interaction; it does not establish an optimal width across workloads.
- v76: the frozen confirmation protocol uses 768 fresh-to-our-runs prompts, native widths 3 and 8, batch sizes 8 and 64, four engine orders, two warmup passes and two timed passes. Results must pass the complete output/provenance audit before promotion. This two-width comparison cannot establish global optimality.

These experiments concern offline batched inference. Batch size is not a substitute for a measured arrival-rate workload, and combining timings from separately initialized engines omits switching costs.

## Proposed empirical extension

After completing the frozen confirmation, measure selection quality as a function of profiling budget. Partition any new workload into profiling, validation, and test prompts before timing or configuration selection. Evaluate the native width grid on all required partitions with balanced engine orders. Select using profiling/validation data only; retain test measurements solely for evaluation.

Compare a default fixed width, a global fixed width selected on validation data, a batch-specific profiling lookup table, and domain-specific profiling tables. Measure cross-domain transfer by selecting on one domain and evaluating on another. An exhaustive test-set winner is a hindsight reference, not a deployable baseline. A learned or online policy would require an additional closed-loop experiment including its overhead.

Report held-out throughput, relative loss to the hindsight width, profiling GPU time, and selection stability across independent profiling subsets. Keep prompts/batches and timing rounds paired when estimating uncertainty; repeated passes over the same prompts are not independent new workloads. Report negative results and all prespecified batch sizes/domains. If selection improves throughput, report how many generated tokens are required to amortize profiling cost under the measured workload.

Data-efficient profiling and transfer are proposed evaluation questions. This review does not establish that they are novel. The project's defensible contribution is the measured evidence, reproducible implementation, and careful account of when configuration tuning helps.
