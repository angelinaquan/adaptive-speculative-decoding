# Expected-prefix single-path selection (pending)

A backward dynamic program chooses one path to maximize modeled expected prefix length: V_j(a)=max_b p_j(b|a)*(1+V_next(b)). Existing candidate transition scores provide row-normalized probabilities. Temperature is selected only on fitting prompts. This is not an acceptance guarantee: the model probabilities can be wrong.

Controls: native greedy, normalized MAP, raw-score global sum and geometrically discounted raw sum. [DCOCO section5.3](https://arxiv.org/html/2610.12570v1) already compares greedy/global/geometric paths. The proposed objective differs from raw-score sums, but dynamic programming and path lookahead are not broadly claimed as new. Novelty remains unproven.

CPU reference passes400exhaustive small cases and a counterexample where prefix reward prefers a different path than greedy. Data collection uses96differentprompts from the v83 training pool, disjoint from v88, with48fit/48internal-validation. Original backbone and candidate scores remain frozen. Native proposals and greedy paths on exported grids must exactly match v83 outputs. No fresh final test, serving benchmark or performance result yet.
