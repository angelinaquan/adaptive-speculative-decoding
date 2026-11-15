# Draft8/verify3 serving cost gate (submitted)

Job3067480, H200, batch64,192reused prompts,64generated-token cap. Seven engine sessions: fresh AR reference, native3/forced3/forced8, then reverse order. Both forced arms verify exactly min(3, legal scheduled drafts) per request. Warmup and complete-generation timing are separate. Native3 is the strong baseline; forced3 measures custom-manager overhead.

The fixed verification manager caps every request and sets total quota to the sum of those caps, so confidence cannot choose the allocation. Random confidence heads are never interpreted as learned probabilities. A separate untimed pass checks actual GPU query lengths minus non-draft tokens against expected verification lengths. Every warmup/timed/audit output must match the fresh AR token IDs exactly; target parameter pointers/versions are audited unchanged.

1000 CPU cap tests and syntax checks pass. GPU allocation and output correctness remain unverified until the run completes. This is a cost gate, not a learned-policy evaluation. Do not multiply replay reward gains into measured throughput and present that as observed performance.

Submitted2026-11-15 12:17:39UTC. Next monitoring no earlier than12:48UTC. W&B logged by runner. Report unchanged.

## Completed serving result

Job3067480 completed exit0 in13m21s. Native3:9346.69tokens/s; forced draft3/verify3:8945.44; forced draft8/verify3:8473.84. Draft8/verify3 is9.34% slower than native3 and5.27% slower than matched forced3. Both orders agree versus native3 (-9.32%,-9.36%). Forced3 overhead itself is4.29%, with order variation.

Local artifact audit passes:7sessions,90batch output-hash checks against fresh AR,178untimed exact GPU allocation checks, target unchanged, source/plan hashes consistent, and throughput arithmetic exactly matches remote. These are192reused prompts on one GPU/model; no significance/generalization claim.

The v98 replay reward gain is therefore not a demonstrated speedup. This constant-verification cost gate does not directly measure the adaptive policy or mathematically bound all optimized implementations, but shows substantial overhead in the current route. Do not promote it as a10% method improvement. Report unchanged.

[W&B run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/1c18m153).
