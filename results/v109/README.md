# Uninstrumented cache-control reproduction (submitted)

Job3069891 repeats the original192prompts, two warmups and two timed passes at b64/BF16 for AR and native3. It uses no worker observer and no logprob output. Six sessions compare existing v102 caches, new caches on first load, and those same new caches on reload. Each native session checks against its corresponding fresh AR reference. Parity failures are preserved and remaining controlled comparisons continue; unrelated failures abort. Timing data are diagnostic only, not a performance result.

Next check no earlier than2026-11-15 16:24UTC. The full8B sweep is still unsubmitted.
