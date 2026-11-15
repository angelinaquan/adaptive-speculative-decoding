# Causal request-residual correction screen

A frozen context/anchor ridge predictor is corrected using the request's past prediction residuals. Each update uses only the acceptance reward at the previously selected width. A shrunk, decayed scalar residual shifts predicted rewards across widths before budgeted allocation. Inner prompt-grouped validation chooses update strength, shrinkage and decay, including a no-update candidate; outer prompts never select parameters.

Fixed3 earns848 reward; frozen predictor853 (+0.590%); residual correction852 (+0.472%). Correction is -0.117% versus frozen. This fails the 10% reward gate and supplies no evidence of throughput improvement.

Passed checks: prompt-disjoint fitting, outer-label poisoning, unchosen-counterfactual poisoning, current/future-feedback poisoning at every offset, and exact allocation-budget checks. These teacher anchors are exogenous, eight tokens apart; this is not closed-loop decoding. The small reused sample cannot establish generalization. Context-aware allocation and feedback adaptation have prior art; no novelty claim.

To reproduce, place experiments/residual_v82.py, experiments/allocation_headroom_v73.py, residual_v82_plan.json and results/v81/context_v75_features.npz in a scratch directory and run residual_v82.py with NumPy. Remote reproduction and W&B upload job3064511 is submitted but not yet verified. The report remains unchanged.
