# 8B first-divergence diagnostic (completed)

Job3068835 completed. With logprobs20, all128 AR/native3 sequences match exactly. Native3 reproduces its prior outputs; AR changes exactly requests70 and97. At both first-divergence positions, instrumented AR/native top20 probability records match, and the two competing tokens have exactly equal log probabilities. This is consistent with a sampling-path/tie issue, but the original no-logprob logits were not captured. It is not proof of the specific faulty operation, a correctness fix, or a performance result.

The follow-up audit compares original no-logprob sampler outputs with torch.argmax and checks zero greedy temperatures, without substituting tokens. [W&B diagnostic](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/de1vcozn).
