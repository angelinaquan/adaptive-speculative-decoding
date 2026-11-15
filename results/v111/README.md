# BF16 diagnostic: original failure not reproduced

[W&B](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/ioaaeppr). Four raw repeats and one log-probability repeat on the original first eight profiling prompts, AR and native width2, with graph and eager execution. AR/native agree within each mode on all repeats. Eager differs from the original graph reference on indices1,3,5,6. Graph mode has tied reported log probabilities for tokens323 and334 at the historical divergence. This is a numerical-stability lead, not proof of the original cause or a sampler bug. No timing claim.
