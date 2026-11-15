# Context-conditioned score adapter (submitted)

The global score calibration in v90 failed. This screen adds 80 interactions between five candidate-score features and a fixed 16-dimensional random projection of causal pre-draft context. Existing shared/position features remain, giving 100/125 parameters for widths3/8. The drafter and candidate lists stay frozen.

Fit on even v89 prompts; screen on odd prompts. Both sets have already been exposed, so this is exploratory. Three grouped folds within fitting prompts select from penalties .1,.01,1 or native fallback (wins ties). Compare matched-capacity CE and expected-prefix losses. A >=10% reward improvement over native plus improvement over CE is required before fresh-data work, and cannot establish serving speed.

[DCOCO](https://arxiv.org/html/2610.12570v1) already studies contextual candidate correction; this is not a broad novelty claim. The potential contribution would need to be demonstrated as effective lightweight adaptation of a released fixed drafter, with appropriate controls and independent timing.

The feature audit passes on four prompts at both widths: exact identity joins, native preservation at zero residual, original feature preservation, finite features, and label poisoning. The original full-pool audit was terminated to avoid duplicating large interaction tensors while fitting; this narrower test is explicitly not a full-pool audit. Main fitting validates every prompt/anchor join and requires every optimizer to converge.

Local fitting session70864 submitted, output /tmp/context_rerank_v96_output.txt. Do not inspect before 2026-11-15 12:05 UTC. No results or report changes yet.
