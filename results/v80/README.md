# Equal-budget robust-rule comparison

This is an exploratory replay of existing full-width timings, not a new GPU benchmark or independent confirmation. Every rule uses 192 unique profiling prompts and the same eight candidate widths and timing passes. Robust and pooled selection use identical mixed-domain profiling panels (64 prompts/domain); single-domain controls use 192 prompts from their respective domains. Equal prompt counts need not imply equal profiling wall time across domains.

The robust rule minimizes the largest domain-relative profiling time ratio. Pooled selection minimizes summed profiling time. Test outcomes never enter selection; changing non-profiling outcomes leaves choices unchanged. A synthetic example where the two rules should disagree passed. All input output hashes were checked. The remote replay and local analysis agree exactly on all records and summaries; all 68 input-file hashes match.

## Result

At batch 8, all rules select width 8. At batch 64, using mean timings:

| Rule | Width choices | Mean pooled throughput loss, v77 | Mean pooled throughput loss, v78 |
|---|---|---:|---:|
| Pooled mixed192 | 3 in all 64 panels | 0.00% | 0.00% |
| Robust mixed192 | 3 in 58 panels; 7 in 6 | 0.31% | 0.38% |
| Alpaca-only192 | 3 in all 4 panels | 0.00% | 0.00% |
| GSM8K-only192 | 3 in 2 panels; 7 in 2 | 1.65% | 2.05% |
| MBPP-only192 | 3 in all 4 panels | 0.00% | 0.00% |

Robust selection also worsens its intended test metric: mean worst-domain relative slowdown is 2.73% versus 2.09% for pooled on v77, and 0.64% versus 0.00% on v78. Selecting on one timing round and testing on the other yields the same direction: robust loses 0.24–0.39% pooled throughput depending on test set and order. No improvement is demonstrated. Pooled remains the preferred rule in this setup.

The 64 mixed panels and four single-domain panels are overlapping combinations, not independent trials. The means are descriptive finite-pool sensitivity summaries; no significance claim is made. Test-set optimal widths are hindsight references. The comparison is post-hoc because these outcomes were examined in earlier work.

[W&B tables and artifacts](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/m2gezfap). The project report was deliberately left unchanged.
