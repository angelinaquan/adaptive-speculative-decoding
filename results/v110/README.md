# Matched total-cost comparison: completed, negative result

Job 3070561 completed successfully. [W&B run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/g6d5cg5y).

Shared-engine screening plus two native confirmations fails against both cheap controls. All four methods selected width 3 in both opposite execution orders. The prespecified requirement of at least 10% less total time than both cheap controls in both orders was not met.

## Primary endpoint: 3,072 requests

| Method | Order 0 total seconds | Order 1 total seconds |
|---|---:|---:|
| Zero-profile transfer: fixed width 3 | 57.85 | 58.60 |
| Native shortlist: widths 3 and 8 | 155.85 | 152.55 |
| Proposed screen + native top-two confirmation | 219.26 | 214.64 |
| Full native sweep: widths 1 through 8 | 410.44 | 416.13 |

Screening takes 40.69% and 40.70% more time than the native shortlist, and 3.79 and 3.66 times the zero-profile transfer time. It saves 46.58% and 48.42% versus the full sweep, which alone is insufficient to support the method. Screening adds cost without improving the selected deployment configuration in this setting.

## Protocol and accounting

The same 4B model, H200 GPU, batch size 64, BF16, and maximum 64 output tokens apply to all methods. Each method runs its own pipeline. Selection uses 192 profiling prompts; deployment uses 192 disjoint held-out prompts repeated 1, 4, and 16 times. These are 192 unique test prompts, not 3,072 independent samples. All shorter endpoints are included in the summary. The test split is consumed by this frozen comparison and must not be reused as an untouched test set for subsequent method tuning.

Total wall time runs from pipeline start through completion of useful inference, including imports, engine startup, profiling warmups and timed passes, selection, profiling-stage teardown, and deployment startup. No deployment warmup is excluded. Deployment process shutdown after the inference endpoint is not included. Common AR correctness-reference construction cost is 39.52 seconds, reported separately rather than charged to each method. Queue time is excluded. Existing disk caches are shared consistently; this is a cached-start result, not a cold-start claim.

The local audit passed 630 batch output hash checks (including six reference construction batches), 32 forced-target parameter checks, and selection reconstruction from profiling data only across 35 engine stages. All outputs match the fresh AR reference. Two opposite orders establish order agreement here, not population statistical significance or generalization across models, hardware, and workloads.

## Artifacts

- `matched_v110_summary.json`: all timings, endpoints, selections, and comparisons.
- `matched_v110_raw.json.gz`: keyed archive of all raw JSON records, frozen plan, and audits.
- `matched_v110_local_audit.json`: independent local audit receipt.
- `matched_v110_run.json`: completed W&B receipt.
- `../../experiments/audit_matched_v110.py`: audit implementation; expects the original individual JSON files beside the script.

The project report PDF is unchanged. No further experiment is launched on the basis of this negative result.
