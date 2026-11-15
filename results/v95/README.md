# Nonlinear feature audit (submitted)

Compare linear and RBF kernel ridge on the larger v83 context/anchor feature dataset. Use all training prompts, three prompt-grouped folds, the same penalties (0.01,0.1,1), and allocation reward to choose the penalty. RBF bandwidth is fixed by the median training-feature squared distance, without labels. Each method allocates under the same 3*n token budget. This is a representation feasibility audit, not a novel method.

The v83 validation outcomes have already been examined in v84, so these results will be exploratory. Median bandwidth uses the whole training-feature pool; folds are not strictly inductive nested CV. Teacher anchors and replay rewards cannot establish serving speed.

Local process session 21781 was launched at approximately 2026-11-15 11:00 UTC. No completion check yet; next monitoring no earlier than 11:31 UTC. Report unchanged.
