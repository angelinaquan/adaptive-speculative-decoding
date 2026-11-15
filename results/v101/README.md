# Matched width3 drafting, shorter verification

Exploratory v89 fitting / v90 screen, already exposed prompts. Draft exactly3, allocate verification widths0..3 under budget2*n. The target emits one correction/bonus token even for zero verification. Marginal survival probabilities decrease with depth, so sorting them with parent-first ties solves this allocation exactly. Compare tuned raw-confidence and logistic hazard calibration, using the same procedures as v98.

Native3 reward1646; raw1465 (-11.00%); calibrated1471 (-10.63%). Calibration beats raw by only0.410%. Fixed verify2 scores1377. The initial restricted1..3 screen is preserved separately and explicitly superseded by the runtime-compatible0..3 result; it did not improve the conclusion.

200 exhaustive allocation tests pass, as do native path reconstruction on561states and outcome-poisoning invariance. The calibrated reward ratio is0.89368. Under the simplified reward/cycle-time model, a10% speedup would require cycle time below0.89368/1.1=0.81243 of native3, or an18.76% saving. This is a feasibility requirement, not a timing prediction. Current startup profiling also has nonmonotonic192/256-token costs and is not a reliable substitute for measured end-to-end timing.

No10% improvement over the strong raw-confidence control is demonstrated, and no GPU run is justified as a calibration-method success test. W&B upload pending. Report unchanged.
