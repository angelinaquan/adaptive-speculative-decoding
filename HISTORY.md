# Research history

The original question was whether serving-time confidence recalibration improved speculative verification under workload shifts. Refitting a pretrained head changed its coefficients, but a batch-64 pilot on reused prompts was 0.24% slower than freezing the fit. An earlier greedy-output mismatch motivated the batch-invariant backend and token-by-token parity checks used in the confirmation studies.

The cost-model study isolated the target-time table while retaining the native confidence head. An exploratory batch-64 comparison estimated a 4.98% improvement over native adaptive verification. The frozen comparison on 768 separate prompts yielded 3.17%. An exploratory batch-128 screen beat native adaptive verification by 2.09% but lost 4.10% to fixed width 3. The paper reports the latter screen as a limitation.

Subsequent branches investigated vocabulary restriction, draft-path selection, feedback-based allocation, and the cost of profiling fixed draft widths. Their gains and failures concern different interventions. The current tree contains the cost-table paper and its confirmation experiments.

The complete pre-cleanup code, run records, reports, and chronological notes remain accessible in the [archived research tree](https://github.com/angelinaquan/adaptive-speculative-decoding/tree/671068dc8fd239b91e63131d016c1ddb0ab7f615) and its [experiment log](https://github.com/angelinaquan/adaptive-speculative-decoding/blob/671068dc8fd239b91e63131d016c1ddb0ab7f615/README.md).
