# Causal repetition lookup: negligible reward headroom

Diagnostic baseline on previously exposed v89 fitting-pool traces, with even/odd prompt split. Retrieve the longest matching suffix (2–32 tokens), breaking ties by newest occurrence; copy only a continuation fully inside the observed prompt/verified prefix. Fit minimum-match threshold from 2,3,4,8,16, plus native-only fallback. Native wins ties.

Fitting selects native-only at widths 3 and 8, giving 0% improvement. Even hindsight routing between these two proposals adds only 0.12% reward at width 3 and 0.34% at width 8 on the screen split. This is not a bound on all retrieval methods, longer-generation workloads, or runtime gains from skipping the drafter. No speedup claim follows.

500 brute-force lookup tests pass. Retrieval and match-length routing are prior art: [SAM Decoding](https://aclanthology.org/2025.acl-long.595/) and [SuffixDecoding](https://arxiv.org/abs/2411.04975). This is a simple feasibility baseline, not a reproduction of their full methods. No GPU serving implementation is justified by this screen. W&B publication submitted as CPU job 3066633; completion unconfirmed; report unchanged.

Verified upload completed: https://app.forge.coreweave.com/aquan-world-model/18657-speculative-pilot/runs/bh0imtmh
