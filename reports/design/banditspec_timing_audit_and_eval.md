# BanditSpec timing audit and proposed fair evaluation

Status: source audit and evaluation design, not a completed reproduction or new submitted experiment. All existing 4B/8B results are development/exploratory evidence for this design. No shortlist is used in the proposed confirmatory experiment.

## Audited source

Repository: https://github.com/sail-sg/BanditSpec

Pinned commit: `4e6dd869a3a1677ca62ee8055713cec439f26ccc` (2025-09-02).

This audit describes the released snapshot, not necessarily the code that produced every ICML table. No GPU reproduction of this repository has been performed.

### Timing and switching

- `generate_utils.py:145–175`: target/draft cache setup, target prefill, first target token, and EAGLE prefill occur before the synchronized outer timer starts.
- `generate_utils.py:174–252`: the outer timer covers the decoding loop, initial exploration, width selection, drafting, verification, cache bookkeeping, GPU synchronization and bandit updates. Exploratory accepted tokens contribute to the generated response, rather than being discarded profiling work.
- `generate_utils.py:180–191`: statistics reset on every generation call. The first seven loop iterations select1,2,3,4,1,2,3, if generation lasts that long. This is not two complete rounds over all four arms.
- `generate_utils.py:193–218`: width changes alter loop bounds and the target verification-buffer slice, while reusing the same resident target, drafter and caches. There is no engine reload for a switch and no separate switch-cost subtraction. Any work executed inside the loop is included in its wall time.
- `generate_utils.py:244–249`: per-round throughput used as feedback is measured through the synchronized end of decoding work. Updating the statistics follows that inner timer but remains inside the outer elapsed time.
- `inference_length.py:23–60`: model construction/loading, the `torch.compile` wrapper call, tokenization and input transfer happen outside generation's timer. Lazy compilation is not thereby guaranteed absent from later timed operations.
- `inference_length.py:79–100`: the LLaMA driver calls adaptive generation, then fixed lengths0–4 separately. Its aggregate omits the first generation batch via `num_run > 1`; it does not omit the early exploration rounds of subsequent calls. The code does not randomize method order here. Printed per-call figures and aggregate figures therefore have different warmup treatment.

### Discrepancies and pitfalls

1. The released `ucb_length_spec_generate` has four statistics entries and only chooses1–4, whereas the paper describes0–4. Zero is a separately run fixed baseline in this driver.
2. At line190 the bonus uses `ucb_calltime.sum()`, producing one scalar added equally to every arm. Consequently it cannot change the argmax relative to the empirical means. This is not an arm-specific uncertainty bonus. It is a direct code finding, not evidence that the published experiments used this exact defect.
3. Both fixed and adaptive routines may break on any EOS in the batch. The adaptive length stopping condition uses a minimum where the fixed routine uses a maximum. Those differences can affect completed work.
4. Termination checks precede updates to `count` and `num`, so a final timed iteration can be absent from the reported throughput numerator. Count actual emitted tokens instead of trusting these counters in a reproduction.
5. The current driver contains dataset and batch-size choices that differ from the paper's prose. It is not a turn-key exact reproduction of its500-instance/16-repeat figure.

Permalinks:
- https://github.com/sail-sg/BanditSpec/blob/4e6dd869a3a1677ca62ee8055713cec439f26ccc/generate_utils.py#L138-L255
- https://github.com/sail-sg/BanditSpec/blob/4e6dd869a3a1677ca62ee8055713cec439f26ccc/inference_length.py#L23-L100

## Evaluation question

Can a full-candidate profiler or online selector improve useful inference throughput and total completion time on a new workload, after paying its actual exploration and switching costs?

Do not attempt to demonstrate that a previously selected shortlist is universally sufficient. Separate online useful exploration from offline discarded calibration; they have different cost accounting.

## Feasibility gate before experiments

Implement and validate genuine resident-engine switching over the full legal set `K={0,1,...,Kmax}`. For the current8B drafter Kmax=7; zero must actually disable drafting. Determine the range from backend support and checkpoint metadata, never speed results. If a candidate is unsupported, document this before evaluation and do not describe the experiment as covering that candidate. Do not simulate short drafts by drafting the maximum and discarding work.

Check arbitrary randomized switch sequences,0-to-positive and positive-to0 transitions, KV rollback, per-sequence EOS, truncation and actual output-token counting against fresh AR references. Check fixed versus switching paths under the same runtime settings. Validate timing boundaries with synchronized GPU completion. A comparison on different kernel paths must disclose that difference; prefer one implementation for every fixed and adaptive policy.

If resident switching is unavailable, the existing per-engine restart experiments remain a separately labelled deployment-cost study. They are not a faithful online BanditSpec-style comparison.

## Proposed bounded evaluation (freeze manifest before any new timing)

- Use one validated target/drafter pair initially; make no unseen-model-transfer claim.
- Predeclare batch sizes1,8,32,64 and output caps128,512, yielding eight conditions. These conditions are not claimed to be wholly unseen: earlier results are development evidence. Use independently sampled, deduplicated held-out prompts and freeze the policies and parameters before their timing. Verify overlap against the saved workload inventory.
- Use a declared balanced chat/math/code workload, not handpicked prompts with favorable acceptance. Natural EOS remains active; caps are caps, not guaranteed output lengths. Include all conditions in the report.
- Use five independent prompt/ordering seeds per condition, paired across methods. Counterbalance execution order to reduce hardware and cache-order effects. Uncertainty is computed over independent workload replicates, not repeated copies of the same96 prompts.
- A separate source-only dataset selects any global fixed default or batch-aware prior rule. No test measurements update those choices. Freeze source-rule fitting too. All old inspected prompts can be source/development data; never relabel them as untouched.

## Policies and controls

1. Every fixed candidate0..Kmax: show the complete response surface, with AR0 as an operational baseline.
2. Best single fixed width chosen on source/development data: primary no-new-profiling control.
3. Source-only batch-aware rule: optional strong transfer control, explicitly informed and frozen.
4. Uniform explore-then-commit across ALL candidates: the simple profiler. One and four complete rounds are two predeclared exploration budgets. Evaluate each separately; do not choose the best budget on test.
5. A paper-faithful BanditSpec baseline: first reconcile the paper algorithm and released implementation, validate arm-specific exploration, and include0 if the evaluation declares it. Any modified rate-based controller must be named as a variant, not an exact reproduction.
6. Hindsight best fixed width is an oracle diagnostic, never a deployable baseline or free source of the initial configuration. Distinguish a noisily estimated best fixed width from a true oracle; use independent repetitions for selection and evaluation where possible.

There is no{3,7}, no min/mid/max shortlist, and no range pruning based on old speed results.

## Two clearly separated experiments

A. Online useful exploration (closest to the audited code): select a width each decoding round. Exploration generates committed valid output for the same requests. Every exploratory round, policy update, synchronization and runtime switch is charged; outputs are counted once. Reset policy state at the beginning of each episode. Report full-episode performance, not just the favorable suffix after learning.

B. Offline profile then deploy: profile ALL candidates on disjoint calibration prompts, freeze the selected width, then run the held-out workload. Charge the measured calibration work and all actual switching/setup costs to total completion time. Calibration outputs do not count as useful held-out output. Use equal profiling wall-time budgets for budget-efficiency claims, not merely equal prompt counts. Profile-only and online results are not interchangeable.

## Metrics and accounting

Primary practical metric: synchronized wall time to complete the same prescribed request batch/workload, including prefill and every emitted output through EOS/cap. Count tokens directly from output buffers, excluding padding and rejected drafts. Require equal output sequences under greedy decoding so that faster termination is not a quality/length artifact.

Secondary paper-like metric: decode-only throughput, starting after prefill and the first token, with the numerator restricted to exactly the corresponding decoded tokens. This allows a timing-boundary comparison with BanditSpec without pretending it measures full serving latency.

Warm-resident and cold-start are separate reported regimes. For warm-resident measurements, initialize and prepare every eligible shape consistently before timing; report that preparation cost and memory separately. For cold-start measurements, charge actual loading, allocation, compilation and switching to each policy. Never charge repeated engine restarts to one method while granting another free resident switching.

Report per-condition total time, selected-arm frequencies, tokens/s, exploration fraction, switches, and adaptation curves. Also report a ratio of summed completion times across the fixed workload mixture, with all cell values visible. No selection of the worst fixed baseline after results. Treat startup savings, inference improvements, and search-cost savings as different quantities.

For payback, show observed completion-time curves over predeclared workload horizons. Estimates outside measured horizons remain extrapolations. If the same configuration is selected, do not convert timing jitter into a finite payback claim. A benefit is credible only when it survives all exploratory/switching costs against the source-trained default, not merely against the worst arm.

## Status and claim boundary

This is a design, not a launched GPU run. Existingv112/v115 jobs are not modified by this audit and remain exploratory evidence under their original protocols. A resident-switching correctness gate, source/test manifests, fixed controller parameters, and resource budget must be completed before submitting the new evaluation. No report PDF or speedup claim is updated by this document.
