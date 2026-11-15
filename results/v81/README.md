# Context-predictor selection screen

Exploratory nested prompt-grouped cross-validation on 277 teacher anchors from 47 reused prompts. This measures accepted-token reward under draft budget 3 per state, not throughput. Linear/RBF kernel ridge models use identical feature and hyperparameter sets; inner selection minimizes prediction MSE or maximizes allocation reward.

| Model | Reward | Gain over fixed3 |
|---|---:|---:|
| Fixed3 | 848 | 0% |
| Anchor, MSE selection | 839 | -1.06% |
| Anchor, reward selection | 845 | -0.35% |
| Context+anchor, MSE selection | 853 | +0.59% |
| Context+anchor, reward selection | 844 | -0.47% |

Outer-label poisoning leaves fitting and predictions unchanged. No candidate meets the 10% reward screening gate. Oracle reward1024 is hindsight, not a deployable result. Remote reproduction and W&B upload submitted as job3064511; completion has not yet been checked.

To reproduce locally, create a scratch directory containing experiments/decision_v81.py, experiments/allocation_headroom_v73.py, this directory's decision_v81_plan.json and context_v75_features.npz, and results/v75/context_v75_summary.json; run decision_v81.py with NumPy installed. Keep output separate from archived results. No GPU needed.

Remote reproduction job3064511 completed0:0; all model rewards match local expected outputs. [W&B](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/0mvuwuyq).
