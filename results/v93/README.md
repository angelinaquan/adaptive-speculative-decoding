# Sharing tree nodes across requests: negative feasibility screen

The global allocator expands the highest cumulative-probability frontier node across requests, while the control gives each request the same fixed number of nodes and builds its own dynamic tree. Both tune temperature using the same five settings on fitting prompts. Draft width and total verification-node budget match within each comparison.

Use the previously exposed v89 even/odd fitting-pool split, grouping requests at the same anchor ordinal, at most one state per request per group, up to 16 requests. Results are exploratory replay rewards, not independent validation or throughput.

| Width / mean nodes | Per-request tree | Shared allocation | Gain |
|---|---:|---:|---:|
| 3 | 826 | 839 | 1.57% |
| 8 | 1289 | 1306 | 1.32% |

150 exhaustive small-case comparisons verify maximum surrogate mass under the global budget. Gold-label poisoning leaves decisions unchanged. This does not verify a serving implementation or attention/KV-cache behavior. Dynamic batch allocation already has prior art; no novelty is claimed. The screen does not justify a GPU implementation for the 10% goal. W&B upload submitted as CPU job 3066610; completion unconfirmed. Report unchanged.
