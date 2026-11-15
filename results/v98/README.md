# Current-draft confidence for verification allocation

Draft width8, then allocate verification lengths1..8 under a shared3*n verification-token budget. Labels are prefixes of the actual width8 proposal; **never substitute independently drafted width-k labels**. Compare native width3 as the stronger compute baseline, fixed draft8/verify3, raw confidence with a fitted logit temperature, and a five-feature logistic conditional-hazard calibrator. Censor training rows after the first rejection. Allocation uses cumulative survival probabilities.

Train on all96v89 prompts, screen on96disjoint v90 prompts. Both historical pools have already been examined: exploratory only. Three penalties are chosen using prompt-grouped OOF reward; three raw temperatures use fitting reward. Selection procedures differ as disclosed; neither reads screen labels.

| Rule | Screen reward | Gain vs native3 |
|---|---:|---:|
| Native draft3/verify3 | 1646 | 0% |
| Draft8/verify3 | 1672 | 1.58% |
| Raw-confidence allocation | 1819 | 10.51% |
| Calibrated allocation | 1828 | 11.06% |

Calibration improves only0.495% over the tuned raw-confidence control. This is **not a10% method improvement over the strongest baseline**, and is not a throughput gain: width8 drafting costs more, while packing/decision overhead remains unmeasured. Dynamic verification and calibration are established mechanisms. Do not promote these numbers to the report as a novel speedup.

On the separate full v83 diagnostic split, the draft8 adaptive-verification hindsight oracle gives16.76% reward over native3, still excluding extra drafting cost. The immediate decision gate before any elaborate implementation is a paired serving cost test of draft8/verify3 versus native3, with exact output parity and fixed verification quotas. If that overhead consumes the reward opportunity, retire the mechanism in this regime.

Checks pass for disjoint identities/texts, native score-path reconstruction, finite monotone reward curves, label-poisoning invariance of frozen allocation, and every deterministic truncation length. No serving correctness has been tested. W&B upload submitted as3067468; check no earlier than12:45UTC. Report unchanged.

Upload verified complete: https://app.forge.coreweave.com/aquan-world-model/18657-speculative-pilot/runs/j0kvd8d7
