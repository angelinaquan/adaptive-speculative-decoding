# Uninstrumented cache-control reproduction (completed)

Job3069891 completed all six AR/native3 sessions on the original192prompts at b64/BF16, with two warmups and two timed passes per session and no worker observer/logprob output. Existing v102 caches, new caches on first load, and the same new caches on reload all passed exact output comparisons. The three AR references agree. Local audit checked72 batch hashes, including9 initial reference-construction batches and36 native/reference comparison batches.

All three AR references differ from the original v102 reference at requests70,97,149,165. The original discrepancy remains unexplained; this experiment does not show a cache-dependent difference or prove a runtime fix. [W&B run](https://wandb.ai/aquan-world-model/18657-speculative-pilot/runs/qm316i2z).

The consistent fresh references justify proceeding with profiling-only v103 under fresh AR checks for every width and batch and immediate abort on any mismatch. They do not justify relaxing correctness or claiming a speedup. Original failed artifacts remain archived; report unchanged.
