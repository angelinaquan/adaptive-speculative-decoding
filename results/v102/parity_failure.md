# Native 8B parity gate failed

Pinned checkpoints downloaded successfully to node-local storage (about451GiB free before,434GiB after). Job3068300 completed the fresh AR reference, then failed during width3 warmup on two GSM8K requests (global indices70 and97). First differences occurred at generated positions37 and5, zero-based. No 8B throughput claim is permitted, and v103 remains unsubmitted.

The cause is unresolved. The next diagnostic reproduces the first128 prompts with top20 log probabilities under the same BF16 native/AR configurations. It records whether instrumentation preserves the historical outputs; it does not relax exact-output checks or constitute a timing benchmark.

[Failed run and artifacts](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/imytnd4f).
