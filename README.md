# Using Measured GPU Costs to Speed Up Speculative Decoding

[Core implementation](src/verification_costs.py) · [Benchmark and analysis code](src/) · [Paper](paper/paper.pdf)

**Research question.** Does enforcing a nondecreasing GPU timing curve lead a speculative decoder to choose slower verification budgets? The native vLLM scheduler uses a running maximum over profiled target times. We replace that running maximum with the median measured at each captured size. The confidence head, token verifier, and allocation rule stay fixed.

**Setting.** Qwen3-4B with the seven-token DSpark drafter on one H200, using greedy decoding and a 128-token output cap. The batch-64 comparison used 768 held-out prompts, equally divided among conversation, math, and code. Each method ran in four balanced engine orders, with two timed passes after warmup. All timed outputs matched the ordinary greedy reference.

**Main result.** At batch 64, measured costs raised warmed-up throughput from **6,059 to 6,251 tokens/s** versus native adaptive verification, a **3.17% gain** (95% paired batch-and-round bootstrap interval **1.13% to 5.45%**). The gain over a globally tuned fixed width was 2.56%. Against prespecified domain-specific widths, the interval included zero. A separate batch-32 experiment found a 1.35% gain over native adaptive verification. Timing excludes startup and profiling.

![Held-out batch-64 throughput gains with 95% paired intervals](results/throughput.png)

[Course paper](paper/paper.pdf) · [LaTeX](paper/main.tex) · [Cost-table implementation](src/verification_costs.py) · [Research history](HISTORY.md)

## Reproduce

From the repository root, recompute both published summaries from the saved session records and regenerate the graph on CPU:

```sh
python -m pip install numpy matplotlib
python scripts/verify.py
```

For a fresh timing run, use an H200 with the paper's vLLM 0.33.0 GPU environment (PyTorch 2.14.0). The runner downloads the pinned checkpoints and stages each experiment in a new directory. W&B logging defaults to offline.

```sh
python -m pip install -r requirements.txt
python scripts/run_cost_study.py --batch 64 --run-dir /tmp/cost-batch64
python scripts/run_cost_study.py --batch 32 --run-dir /tmp/cost-batch32
```
