# Final-project experiment plan: profiling cost and payback

**Frozen question:** When does draft-length tuning pay for itself, and when are prior choices or a two-width shortlist enough?

Job3089218 is submitted after successful completion of the existing held-out run3089108. It cannot run after a correctness failure. No result is claimed yet. No change to the running v112 plan or report PDF.

| Question | Comparison | Evidence |
|---|---|---|
| How much does profiling cost? | Candidate engines for pair versus full sweep | Actual profiling wall seconds split into engine initialization, warmup generation, timed generation, and residual overhead |
| Does a smaller budget suffice? |64 versus192 profile prompts, pair and full | Same nested domain-balanced ordering; selected widths and held-out total time |
| Does prior knowledge suffice? | Fixed3, fixed7, and prior load rule versus tuned choices | Two batch sizes8/64; prior rule selects7 at8 and3 at64 |
| When does tuning pay back? | Actual total elapsed time for all six methods |96,384,1536 requests; two opposite method orders; labelled extrapolation for break-even estimates |

Methods are fixed3, fixed7, pair64, full64, pair192, full192. Pair profiles3/7; full profiles1–7. There are24 actual pipelines and98 engine stages on one GPU. Both budgets use the same192-prompt permutation, interleaved by domain with seed20261116. The first64 contain22 Alpaca,21 GSM8K,21 MBPP prompts. Each candidate runs one charged warmup and one timed pass. Every selection is saved before held-out inference. The same96 validation prompts used by v112 are explicitly reused for this supplemental evaluation; these are not a new independent confirmation set. The budget contrast is measured within v115, not by comparing mixed-domain64 with v112 clustered-domain192.

## Cost interpretation

The primary outcome is actual monotonic pipeline wall time through useful inference, including imports, engine startup, profiling warmup/timing, selection, profiling teardown, deployment startup, and checks/file overhead. There is no excluded deployment warmup. Queue time, final deployment shutdown, and common correctness/reference work are outside operational pipeline time. Existing compilation caches are reused. These results concern native engines initialized separately for each candidate width, not an intrinsic lower bound on tuning cost or a runtime capable of free width changes.

Report observed totals at all three horizons, both orders separately, and profiling cost components. Estimate a slope from the384-to1536-request wall-time interval; fit an intercept and estimate a crossing only with explicit extrapolation labels. If two methods select the same width, do not convert timing noise into an inference-based payback claim. If the tuned configuration is not faster, report no fitted payback. Two execution orders and repeated96-prompt workloads do not support population confidence intervals.

Success is an answer, not a positive speedup: the shortlist may match the sweep, the larger budget may change the choice, profiling may beat a universal default yet lose to an informed prior, or profiling may not pay back within measured horizons. All are reportable. Earlier4B measurements are supporting evidence under their own precision/runtime conditions, not numerically interchangeable with these8B timings.

The runner publishes raw artifacts, partial pipeline progress, a cost-breakdown table, observed total-time curves, and a JSON analysis to W&B. Any exact output mismatch aborts; no filtering or selective retry is permitted. The final report should use three figures: profiling-cost breakdown, observed total time versus workload size, and selection/payback table. No new algorithm search is part of this suite.
