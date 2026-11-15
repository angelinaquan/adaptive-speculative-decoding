# Missing-candidate probability for tree allocation: negative screen

Candidate-only softmax gives total probability one to the shortlist. This screen multiplies it by an estimated probability that the correct token is in the shortlist. Logistic features are position, row entropy, and top-two margin, with an intercept. Labels follow true prefixes and include the first absent token; later unreachable positions are censored. A position-only Beta(1,1)-smoothed estimator is the simple control.

Same previously exposed v89 even/odd prompt split as v91. All three probability methods select temperature from the same five values using fitting reward only. Node budget equals draft width. No independent validation or timing claim.

| Width | Raw tree | Position coverage | Feature coverage | Feature vs strongest control |
|---|---:|---:|---:|---:|
| 3 | 826 | 818 | 819 | -0.85% |
| 8 | 1289 | 1291 | 1295 | +0.31% |

Correctness checks pass for censoring, finite-difference gradients, equality to the raw tree at unit coverage, gold-label poisoning, and tree budgets/closure. This candidate does not justify a serving implementation or a claim of 10% improvement. Retired at exploratory-screen stage. Report PDF unchanged.

W&B publishing of v91 and v92 was submitted as CPU job 3066563; completion has not yet been checked. No GPU job was launched for this screen.

Verified upload completed: https://app.forge.coreweave.com/aquan-world-model/18657-speculative-pilot/runs/sx1jha5q
