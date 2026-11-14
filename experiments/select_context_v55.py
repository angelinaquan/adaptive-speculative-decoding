from collections import Counter

def select(prompt_ids, ranking, budget, use_context=True):
    counts=Counter(v for seq in prompt_ids for v in seq) if use_context else Counter()
    context=sorted(counts,key=lambda v:(-counts[v],v))[:budget]
    chosen=set(context)
    for v in ranking:
        if len(chosen)==budget:break
        chosen.add(v)
    assert len(chosen)==budget
    return sorted(chosen),dict(unique_prompt_tokens=len(counts),included_prompt_tokens=len(context))
