# No-logprob greedy sampling audit

Initial job3069158 failed during engine initialization because the observer incorrectly asserted that synthetic warmup requests were greedy. It produced no model evidence. The retry explicitly arms observation only after initialization; CPU regression checks cover that lifecycle.

Retry3069514 completed. Every observed Sampler.sample result agreed with torch.argmax and CPU/GPU greedy temperatures were zero. All128 observed AR/native outputs agree, but AR again differs from the original uninstrumented output while native reproduces it. This audit covers Sampler.sample, not every rejection-kernel decision. It does not justify a sampler patch. The observer synchronizes GPU work, so its changed AR output motivates a controlled uninstrumented reproduction.

[Failed diagnostic](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/38k69mbq), [completed retry](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/f3vy8yis). No performance claim; report unchanged.
