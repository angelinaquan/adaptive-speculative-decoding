# Learned frozen-score residual reranking (pending)

Fit a small linear correction to frozen candidate-transition logits. Five fixed score/rank/entropy features have shared and position-specific coefficients:20parameters at width3 and45at width8. The neural drafter and target remain frozen. Greedy selection uses the corrected scores.

Compare ordinary teacher-prefix cross-entropy to negative expected accepted-prefix length under the corrected softmax. Both use identical capacity, three ridge penalties, three prompt-grouped fitting folds, and a native fallback. All96v89prompts are now fitting data, explicitly including that earlier internal validation. The new96v90prompts are disjoint, taken from another part of the historical v83training pool; they never select parameters. This is an internal held-out screen, not a fresh final benchmark.

Analytic CE/prefix gradients match finite differences. Expected-prefix loss matches exhaustive path expectation on a synthetic chain. These tests validate implementation, not model accuracy. The prefix objective must gain>=10%held-out reward over native and improve over CE before fresh serving investigation. Acceptance-oriented training/correctors have prior art; no novelty or speed claim yet.

Local training session85224 started. GPUcollection job3066201 submitted with W&B; native proposal and exported-grid reproduction gates retained. Report unchanged.
