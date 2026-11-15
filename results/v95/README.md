# Nonlinear feature audit: negative result

On the larger v83 dataset (2218 training states, 549 screen states), compare linear and RBF kernel ridge under the same prompt-grouped folds, penalty grid, and allocation reward selection. RBF bandwidth is the median training-feature squared distance, without labels. Both allocate under 3*n tokens.

| Model | Reward | Gain vs fixed3 |
|---|---:|---:|
| Fixed width3 | 1683 | 0% |
| Linear | 1692 | 0.53% |
| RBF | 1700 | 1.01% |

RBF gains only 0.47% over linear. This does not support the 10% goal and does not justify serving integration. Retire this candidate. The screen outcomes were previously exposed in v84, so these are exploratory results. No inference-throughput or novelty claim.

Audits pass for input hashes, disjoint prompt IDs and normalized texts, grouped folds, actual feature normalization, frozen decisions independent of feedback labels, allocation budgets, and synthetic RBF kernel properties. The bandwidth uses all training features across folds, so this is not strictly inductive nested CV. Source hash is recorded in checks.

Local session 21781 exited0. W&B publication submitted as job3067020 alongside repair of the failed v90 artifact upload. The v90 failure was a missing source file, not an experiment failure; missing files were copied and the original run is resumed. Next check no earlier than 2026-11-15 12:03 UTC. Report unchanged.
