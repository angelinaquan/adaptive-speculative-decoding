"""Prefix-loss objective for choosing a compact greedy draft vocabulary.

Identity applies to one block with unchanged backbone features and Markov
correction, identical arithmetic and tie-breaking, and restricted argmax.
It is not a closed-loop speed prediction or a stochastic-sampling theorem.
"""
import numpy as np

def lost_prefix_tokens(accepted_prefix, allowed):
    """Positive loss relative to full-vocab draft along this fixed block."""
    for j, token in enumerate(accepted_prefix):
        if int(token) not in allowed:
            return len(accepted_prefix)-j
    return 0

def prefix_weights(prefixes, vocab_size):
    """Additive union-bound weights: early mistakes destroy longer suffixes."""
    weights=np.zeros(vocab_size,dtype=np.float64)
    for prefix in prefixes:
        for j,token in enumerate(prefix):
            weights[int(token)]+=len(prefix)-j
    return weights

def choose_vocabulary(prefixes,vocab_size,budget,weighted=True):
    assert 1<=budget<=vocab_size
    weights=prefix_weights(prefixes,vocab_size) if weighted else np.bincount(
        [int(t) for p in prefixes for t in p],minlength=vocab_size)
    return set(np.lexsort((np.arange(vocab_size),-weights))[:budget].tolist())

def draft(base, transition, anchor, allowed):
    previous=anchor;tokens=[]
    ids=np.array(sorted(allowed))
    for logits in base:
        previous=int(ids[np.argmax((logits+transition[previous])[ids])])
        tokens.append(previous)
    return tokens

def acceptance(proposals,target):
    for j,(x,y) in enumerate(zip(proposals,target)):
        if x!=y:return j
    return len(target)

def check_identity(seed=18657,cases=1000):
    rng=np.random.default_rng(seed);checked=0
    for case in range(cases):
        v,k=4,5
        base=rng.normal(size=(k,v));transition=rng.normal(size=(v,v))
        if case%10==0:base[:]=0;transition[:]=0
        anchor=int(rng.integers(v));full=draft(base,transition,anchor,set(range(v)))
        target=full.copy()
        mismatch=int(rng.integers(k+1))
        if mismatch<k:target[mismatch]=(target[mismatch]+1)%v
        full_len=acceptance(full,target);prefix=full[:full_len]
        weights=prefix_weights([prefix],v)
        for mask in range(1,1<<v):
            allowed={i for i in range(v) if mask&(1<<i)}
            restricted=draft(base,transition,anchor,allowed)
            actual=max(0,full_len-acceptance(restricted,target))
            predicted=lost_prefix_tokens(prefix,allowed)
            assert actual==predicted,(case,allowed,actual,predicted)
            assert predicted<=sum(weights[i] for i in range(v) if i not in allowed)
            checked+=1
    return dict(passed=True,cases=checked,includes_exact_ties=True,
        scope='Single-block restricted greedy Markov draft, fixed backbone features. Exact positive-prefix-loss identity and additive upper bound; NOT end-to-end speed evidence.')

if __name__=='__main__':
    import json
    print(json.dumps(check_identity(),indent=2))
