# Matched total-cost comparison (submitted)

Job3070561 tests shared-engine screening plus two native confirmations against zero-profile transfer(width3), native shortlist{3,8}, and full native1..8 profiling. Each method runs as its own actual pipeline in two opposite method orders on the same4B/H200/b64 setting. Selection uses192 profile prompts only. Deployment uses the192 held-out test prompts in the v103 manifest, repeated1,4,16times; those test outcomes are now reserved for this frozen comparison rather than future method tuning.

Total wall time includes engine startup, imports, all profiling warmups and timings, selection, deployment startup and useful inference. The first deployment pass is counted, not excluded as warmup. Common AR correctness-reference cost is reported separately. Existing per-width disk caches are shared consistently; this is not a cold-start claim. Strict fresh-AR equality applies throughout.

The prespecified primary endpoint is total elapsed time through3072 requests. A positive claim requires at least10% lower total time than both cheap controls in both orders. All shorter endpoints and the full-sweep comparison are also reported. Two orders and repeated prompts do not establish a population confidence interval. No result exists yet. Report unchanged.
