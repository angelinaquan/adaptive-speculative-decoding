# Second-model baseline: Qwen3-8B (submitted)

Target `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`; drafter `RedHatAI/Qwen3-8B-speculator.dflash2` revision `e1c737a13c34ba1e1f0f739e02af0bf082fe8080`. Public metadata is archived. This drafter has block_size8 with a bonus anchor and **seven** speculative tokens, unlike the earlier4B width8 configuration. Compatibility checks cover verifier identity, vocabulary, layer indices, and proposal length.

CPU preparation3068012 downloads pinned weights, validates configs, and publishes pending v101 W&B logs. GPU job3068013 depends on successful preparation and is configured to cancel on failed dependency. Batch64 H200,192reused prompts,64token cap. Fresh AR reference plus native3/native7 in both orders; two warmup and two timed passes, separate compiler caches per arm. Every batch must exactly match AR token IDs. No learned policy is evaluated.

This is a correctness/cost smoke gate, not a full width sweep, independent generalization result, or new-method improvement. If it passes, widths1..7 and disjoint profiling/test prompts are required before claiming a tuned native baseline. It cannot satisfy the10% proposed-method goal by itself.

Submitted2026-11-15 13:03UTC. Next monitoring no earlier than13:34UTC. Report unchanged.
