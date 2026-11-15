# Shared-engine profiling screen (exploratory gate passed)

Candidate: screen all widths in one reusable forced-width engine, then measure only shortlisted widths with native engines. This targets profiling startup cost, not faster inference. Its central unresolved assumption is that shared-engine timings preserve the useful native ranking despite different execution overhead.

The v79 diagnostic shows that generation accounts for only 6.62–7.17% of total measured profiling time. Eliminating all generation cannot save 10% under the unchanged eight-engine lifecycle. This is arithmetic on prior timings, not a universal lower bound or measured racing result.

The v105 exploratory gate uses the existing 4B model and 192 reused prompts, batch64, max64 output tokens. It switches widths1–8 in one isolated-cache engine, uses two warmups and two timed passes in each of two opposite orders, and requires exact fresh-AR outputs throughout. The forced planner verifies all legal drafts; untrained confidence heads do not choose budgets. It records full subprocess time separately from initialization and generation.

Before collecting outcomes, we fix the shortlist to the two widths with smallest total generation time. The historical v72 native winner must appear in both order-specific shortlists. That only warrants fresh native confirmation; historical rank agreement is not held-out evidence or a measured cost improvement. A successful final evaluation must include zero-profile transfer, an equal-cost native shortlist, and a full native sweep. Test data remain reserved.

Generic bandit configuration selection is prior work ([BanditSpec, ICML2025](https://proceedings.mlr.press/v267/hou25h.html)); this screen carries no novelty claim. A search did not establish that this exact engine-screening application is new. The original >=10% steady-state inference target is still unmet, and profiling savings are a separate exploratory outcome.

Job3068356 completed. Shortlists were [3,5] and [3,4] in opposite orders; pooled shortlist [3,4]. Historical native winner3 survived both. Local audit reproduced all shortlists and checked162 output hashes and48 target-parameter audits. The shared-engine subprocess took236.81s, including158.18s initialization; separate AR reference generation took118.81s. These timings are not a matched cold/warm comparison against the full native sweep, and there is no measured profiling or inference improvement yet. [W&B run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/hky0zfhg). Report unchanged.
