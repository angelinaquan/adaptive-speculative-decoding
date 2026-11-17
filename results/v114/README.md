# Full FP32-head correctness regression passed

All32 engine sessions completed, with1856 batch output checks passing against a fresh AR reference. This covers200 prompts (192 profiling plus8 historical regression cases), batches8/64, native widths1–7, two engine loads, and two warmup/two timed passes per engine. The local independent audit reconstructs all record hashes, token counts, phase/repeat/batch coverage, configurations, and source/plan hashes.

The original Slurm job3072601 failed **after** this gate was saved: the W&B finalizer incorrectly used keyword arguments with `SummaryDict.update`. As a result, held-out inference never started. The finalizer now passes a dictionary; success and failure paths were exercised with a mapping-only summary fixture. `repair_publish_v114.py` audits and republishes the existing records to [the original W&B run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/hj28pf3h), preserving the original job failure in metadata. No GPU correctness rerun is needed.

Job3089108 repairs publication and starts only the held-out v112 comparison. No held-out speed result is available at submission. The FP32 output head is an empirically validated numerical mitigation; the original intermittent BF16 mismatch remains unproven at the root-cause level. This does not justify claiming an upstream sampler bug was fixed.
