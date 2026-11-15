# Frozen-score residual reranking: negative result

Fit on all 96 v89 prompts, including its formerly internal validation prompts; evaluate on 96 distinct prompts from the historical training pool. This is an internal screen, not a fresh final test. Models were frozen before evaluation.

| Width | Native reward | CE reward | Prefix reward | Prefix gain |
|---|---:|---:|---:|---:|
| 3 | 1646 | 1646 | 1646 | 0.00% |
| 8 | 2371 | 2418 | 2363 | -0.34% |

At width 3, grouped training cross-validation selected the native fallback for both objectives. At width 8, CE improves reward 1.98%, while the proposed prefix objective loses 0.34%. Both prespecified gates fail. Retire this candidate; no serving-speedup claim is justified.

Checks cover analytical gradients, exhaustive prefix expectation, saved model/input/source hashes, disjoint prompt identities/texts, optimizer convergence, and exact reconstruction of native proposals. Data collection passed its native-output checks. See [collection W&B run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/ueda54c1). Evaluation upload was submitted as CPU job 3066538 and remains unconfirmed.

The training-only error diagnostic finds 251/285 width-3 first errors and 390/466 width-8 first errors still have the correct token in the candidate set. Most of these correct tokens rank second or third. This establishes selection headroom only: oracle rewards cannot be presented as deployable improvements. Cheap score calibration has not recovered this headroom. The report PDF is unchanged.

Upload verified complete: https://app.forge.coreweave.com/aquan-world-model/18657-speculative-pilot/runs/w9464hrc
