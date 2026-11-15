# Second-model profiling and evaluation split (prepared, not run)

The Qwen3-8B native sweep is gated on v102 exact-output correctness. It will measure native widths 1–7 at batches 8 and 64 on the profiling split only. No proposed method or speedup is established by this preparation.

Each domain has 64 profiling, 32 validation and 64 test prompts. Validation and test texts are disjoint from the saved workload-manifest and fit-JSONL inventory after lowercasing and whitespace normalization; MBPP task IDs also exclude known full/sanitized task overlaps. This is not an audit of unrecorded historical inputs or model pretraining. All splits exclude v102 smoke prompts.

Profiling reuses 2 Alpaca, 5 GSM8K and all 64 MBPP prompts. Validation may be used for development. Test outcomes must remain uncollected until a separate candidate and analysis plan is frozen. The primary target remains a measured 10% throughput improvement over strong tuned controls, including overhead.

Preparation assertions, Python syntax checks and ten fake-engine CPU integration sessions pass. Injected output mismatches and stale AR references are rejected. These checks cover harness bookkeeping only, not GPU decoding correctness. The new GPU runner has not executed and has no W&B run yet. It logs timings and artifacts when launched; it aborts on any exact-output mismatch. The report is unchanged.
