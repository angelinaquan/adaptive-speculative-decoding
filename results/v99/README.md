# Draft8/verify3 serving cost gate (submitted)

Job3067480, H200, batch64,192reused prompts,64generated-token cap. Seven engine sessions: fresh AR reference, native3/forced3/forced8, then reverse order. Both forced arms verify exactly min(3, legal scheduled drafts) per request. Warmup and complete-generation timing are separate. Native3 is the strong baseline; forced3 measures custom-manager overhead.

The fixed verification manager caps every request and sets total quota to the sum of those caps, so confidence cannot choose the allocation. Random confidence heads are never interpreted as learned probabilities. A separate untimed pass checks actual GPU query lengths minus non-draft tokens against expected verification lengths. Every warmup/timed/audit output must match the fresh AR token IDs exactly; target parameter pointers/versions are audited unchanged.

1000 CPU cap tests and syntax checks pass. GPU allocation and output correctness remain unverified until the run completes. This is a cost gate, not a learned-policy evaluation. Do not multiply replay reward gains into measured throughput and present that as observed performance.

Submitted2026-11-15 12:17:39UTC. Next monitoring no earlier than12:48UTC. W&B logged by runner. Report unchanged.
