# Width-allocation headroom by policy class

Use the full v83 labels with the same fixed groups of at most16 requests and per-anchor budget3*n. All figures below are **infeasible diagnostics**, not proposed-method gains, and use already exposed teacher-state data.

| Diagnostic | Train reward gain | Validation reward gain |
|---|---:|---:|
| Exact per-state oracle | 22.84% | 17.83% |
| Exact one-width-per-request oracle | 7.45% | 6.12% |
| Previous-state full counterfactual feedback | -7.19% | -11.65% |

The persistent oracle is a binary integer program with one width per request, all per-anchor budget constraints (including requests absent at later anchors), and total realized reward as objective. Every solve reports optimality with zero relative gap and passes integer feasibility/objective reconstruction. It matches exhaustive enumeration on100 synthetic cases.

Thus, on these recorded states and constraints, even perfect hindsight cannot obtain10% reward improvement by assigning each request a fixed width. This narrows the search toward state-dependent information or a different computational mechanism. It is not a bound on throughput, other workloads, changing groups, or state-dependent policies. The previous-feedback policy is one privileged heuristic, not an upper bound on all causal policies.

A request-mean hindsight heuristic is also saved for transparency. It can change widths as group membership changes, so its6.71% validation gain is not a valid fixed-per-request upper bound; use the exact6.12% result.

W&B upload submitted as3067086; completion unconfirmed. Report unchanged.
