# Draft-stage profiling gate (pending)

Job3065022 submitted. Compare native width3 with the previously parity-checked forced-width3 runtime, in both engine orders. Both use BF16, batch64, the same192reused prompts,2warmup/2timed passes, isolated compiled caches, and fresh AR output checks. Then run one additional forced3 pass with CUDA events around pre-draft, context, backbone and tail graph replays. Events are read only after generation, with no per-step synchronization. Target parameter signatures and output equality remain checked.

The instrumented pass is excluded from throughput comparisons. Tail time includes output projection AND candidate selection; it is not an isolated LM-head measurement. Eager fallback, verification and host work are not separately attributed. CUDA durations are not a complete end-to-end decomposition. Compare instrumentation overhead and the native/forced throughput gap before interpreting component fractions. This is a headroom measurement, not an optimization or novelty claim.

Potential low-precision drafters and speculative vocabularies have prior art, including [QSpec](https://aclanthology.org/2025.emnlp-main.240.pdf) and [Speculative Vocabulary](https://arxiv.org/abs/2602.13836). Any new statistical control mechanism must be distinguished from those methods and evaluated against tuned alternatives. No performance results yet; report unchanged.

## Completed result

Job3065022 completed0:0 in9minutes. [W&B](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/zoe1c4a2). Independent local audit passes5sessions/63batch-output checks and source/plan hashes. Native3:9319.41tokens/s; forced3:9065.74tokens/s (-2.72%). Instrumented forced3 differs by-0.082% from its uninstrumented throughput.

Of1279.27ms measured profile wall time, tail65.08ms (5.09%), backbone143.81ms, pre6.77ms, context5.91ms were attributed to graph replays. Eager, target and host work are not individually attributed. Hypothetically deleting the entire tail gives only5.36% improvement within this measured forced run; this is not a native-runtime speed bound or realizable head-only saving. It provides no support for a10% output-head optimization at this batch size. One low-batch check is queued separately; do not generalize this component profile across loads.
