# Equal-node dynamic tree feasibility screen

Exploratory reuse of the v90 fitting pool: even prompt IDs fit temperature and the empirical static rank tree, odd IDs score them. These prompts have already been examined; this is not independent validation. Node budgets 3 and 8 use their corresponding draft score grids. No hidden target labels enter dynamic construction.

Dynamic expansion uses cumulative conditional softmax probability. The static control chooses the most frequent training gold rank prefixes under the same node budget; here it selects a chain. Temperature is selected from 0.1, 0.25, 0.5, 1, 2 on fitting prompts only.

| Draft width / verification nodes | Chain/static reward | Dynamic reward | Gain |
|---|---:|---:|---:|
| 3 | 816 | 826 | 1.23% |
| 8 | 1193 | 1289 | 8.05% |

400 exhaustive small-tree objective comparisons pass, along with budget, closure, gold-poisoning, and reward checks. This does not validate a serving verifier, attention masks, or KV-cache handling. Tree verification overhead has not been measured, and reward is not throughput.

Dynamic tree expansion is prior art: [DySpec](https://arxiv.org/abs/2410.11744); [Sequoia](https://arxiv.org/abs/2402.12374) studies optimized tree structures and hardware-aware sizing. This implementation is a feasibility baseline, not a faithful full-system reproduction or novel method. It does not meet the 10% goal.

Potential follow-up: learn calibrated conditional acceptance probabilities from training feedback to allocate tree nodes, compared with this raw-score baseline and a tuned static tree. Any follow-up needs fresh evaluation and must establish benefit beyond published dynamic-tree methods. No report change is justified yet.
