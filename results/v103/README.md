# 8B profiling sweep: failed correctness gate

[W&B logs](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/5y5lxy7r). Job 3070220 failed after 28 completed sessions. Batch 8, width 2, reverse order diverged from the fresh AR reference during warmup at profile index 6, token index 57. The cause is unresolved. No relaxed correctness gate or selective rerun is used to turn this into a passed experiment.

## Completed batch-64 subset (diagnostic only)

All 14 native sessions passed; 168 batch hashes were independently checked against the AR reference. Width 3 was fastest in both orders: 1.695 and 1.701 seconds per 192 profiling prompts. Maximum width 7 took 1.993 and 1.972 seconds. The full sweep selected the same width 3 already suggested by 4B evidence; it found no better configuration in this subset.

This drafter supports widths 1–7, so the proposed literal {3,8} transfer comparison is unavailable. Reinterpreting it as {3,maximum legal width} gives {3,7}, also selecting 3 in both orders. That reinterpretation is explicitly exploratory, not a preregistered test.

These are profiling-split timings, not held-out inference or a matched total-cost result. The global correctness gate failed, so this partial evidence cannot support a final generalization or end-to-end speedup claim. The report PDF is unchanged.

`model_profile_v103_transfer_audit.json` records the audit and timings. `model_profile_v103_raw.json.gz` contains the stage records, including the failed stage. `../../experiments/analyze_transfer_v103.py` reproduces this analysis from individual records placed beside the script.
