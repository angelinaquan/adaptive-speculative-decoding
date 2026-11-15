# Candidate-coverage calibration: contingent experiment

This candidate is prepared only as a CPU reference while v85 measures output-stage headroom. No GPU implementation, accuracy result, or speedup is claimed.

Use an approximate draft output projection to rank vocabulary tokens, then recompute accurate scores only on a selected subset. This broad approach is prior art: [SpecVocab](https://arxiv.org/html/2602.13836v1) uses a trained low-dimensional projection, a fixed top-k shortlist, and accurate rescoring, and explicitly allows varying k. [DynaSpec](https://arxiv.org/abs/2510.13847) routes contexts to token clusters. [SlimSpec](https://nebius.com/blog/posts/slimspec-faster-speculative-decoding-without-cutting-the-vocabulary) uses a low-rank full-vocabulary head. [Coverage/latency optimization](https://arxiv.org/abs/2603.05210) is also established. Neither dynamic vocabulary selection nor low-rank projection is our novelty claim.

The specific hypothesis is that a calibrated score-gap threshold can use fewer accurate dot products than a fixed shortlist at matched candidate-set coverage. A possible cheap ranker is an FP8 projection; this remains unimplemented and its cost/quality is unknown in this runtime. The appropriate controls include full BF16, approximate-only, and fixed-size approximate-then-rescore; the method must not win merely because a strong approximate-only baseline is omitted.

For one draft row let s be approximate logits, l accurate logits, T the accurate top-k set, and s_(k) the kth largest approximate score. Define the nonconformity score

    a = max(0, s_(k) - min_{v in T} s_v).

At threshold q, retain C = {v : s_v >= s_(k)-q}. Then a <= q implies T is contained in C. Accurate reranking recovers T provided the same numerical logits and tie rule are used. This is a deterministic containment statement, not a claim that different GPU matrix shapes give bitwise identical logits.

For calibration, take the maximum row score within each prompt, and select order statistic ceil((n+1)*(1-alpha)) across n independent/exchangeable calibration prompt blocks. If the rank exceeds n, use infinity and retain the full vocabulary. Standard split-conformal rank reasoning gives marginal coverage of a new exchangeable prompt block. The calibration construction is not a new statistical theorem; see [Romano, Sesia and Candes,2020](https://arxiv.org/abs/2006.02544).

The block must be defined before outcomes are seen. A guarantee over a few teacher anchors does not cover an entire decoding trajectory. For a whole-trace claim, calibration must cover every relevant position within a fixed output cap and fixed width, or use a separately justified sequential procedure. Correlated tokens within a prompt are not independent calibration examples. Model fitting must use separate prompts. Workload distribution shift invalidates an unqualified exchangeability claim. No conditional coverage guarantee is asserted.

Full target verification remains unchanged. Output correctness must come from the verifier and independent parity tests; a calibrated shortlist is not a substitute for verification. In this DFlash2 pipeline, preserving the complete unary candidate set matters because a predecessor-conditioned selector can choose a candidate other than the highest unary logit.

Proceed only if measured component costs leave credible overall headroom. First evaluate candidate coverage, retained-set size, exact-rescore cost, and acceptance on separate fitting/calibration/validation prompts. Freeze all settings before a fresh test. Include candidate selection, gathers, quantization, accurate rescoring, graph buckets and fallback in end-to-end timings. A slower method with a smaller candidate set is a negative result. The requested >=10% throughput gain against tuned controls remains the final gate.
