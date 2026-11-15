# No-logprob greedy sampling audit (submitted)

Job3069158 observes Sampler.sample in fresh AR and native3 engines on the first128 reused v102 prompts. It compares emitted samples to torch.argmax of the same processed logits, verifies CPU/GPU temperatures are zero, and records any discrepancies with logit values. It never alters sampled tokens. Synchronization may perturb scheduling; historical reproduction is checked separately. No fix or performance claim yet. Next check no earlier than2026-11-15 15:19UTC.
