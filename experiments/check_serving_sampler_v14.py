"""Finite-state distribution check of the installed vLLM rejection kernel.

Exercises probabilistic drafts and variable causal prefix lengths. This does
not prove model KV-cache correctness or adaptive scheduler correctness.
"""
import itertools
import json
import os
from pathlib import Path


def main():
    import numpy as np
    import torch
    import wandb
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    rng = np.random.default_rng(141314)
    p = np.array([[.2, .5, .3], [.6, .1, .3], [.25, .35, .4]])
    q = np.array([[.7, .2, .1], [.1, .6, .3], [.4, .5, .1]])
    sequences = list(itertools.product(range(3), repeat=3))
    lookup = {s:i for i,s in enumerate(sequences)}
    expected = np.array([p[0,a]*p[a,b]*p[b,c] for a,b,c in sequences])
    run = wandb.init(entity='aquan-world-model', project='18657-speculative-pilot', group='serving-v14',
                     name='actual-kernel-finite-vocab-check', dir=os.environ['SERVING_SCRATCH'])
    result = {'passed':False, 'cases':[], 'run_url':run.url,
              'scope':'Installed GPU rejection kernel, three-state Markov target; not full model or scheduler proof'}
    print('RUN_URL', run.url, flush=True)
    try:
        for mode in ['fixed3', 'causal_variable']:
            counts = np.zeros(27, dtype=int)
            lengths = np.zeros(4, dtype=int)
            for chunk in range(20):
                n = 1000
                targets, drafts, local_pos, mapping, cu, q_logits = [], [], [], [], [0], []
                for i in range(n):
                    prev = 0
                    tokens, probabilities = [0], []
                    for j in range(3):
                        # Admission uses previous tokens only, before drawing candidate j.
                        if mode == 'causal_variable' and j > 0 and prev == 1:
                            break
                        probabilities.append(q[prev])
                        token = int(rng.choice(3, p=q[prev]))
                        tokens.append(token)
                        prev = token
                    k = len(tokens)-1
                    lengths[k] += 1
                    targets.extend([np.log(p[t]) for t in tokens])
                    drafts.extend(tokens)
                    mapping.extend([i]*(k+1))
                    local_pos.extend(range(k+1))
                    cu.append(cu[-1]+k+1)
                    q_logits.append(np.log(probabilities+[q[0]]*(3-k)))
                def tensor(x, dtype):
                    return torch.tensor(np.asarray(x), dtype=dtype, device='cuda')
                sampled, num_sampled = rejection_sample(
                    tensor(targets, torch.float32), tensor(q_logits, torch.float32),
                    tensor(drafts, torch.int64), tensor(cu, torch.int32),
                    tensor(local_pos, torch.int64), torch.arange(n, device='cuda', dtype=torch.int32),
                    tensor(mapping, torch.int32), tensor(local_pos, torch.int32),
                    torch.ones(n, device='cuda'),
                    tensor(rng.integers(1, 2**60, size=n), torch.int64), 3)
                values = sampled.cpu().tolist()
                ns = num_sampled.cpu().tolist()
                for ids, count in zip(values, ns):
                    assert 1 <= count <= 4
                    seq = ids[:count]
                    assert all(0 <= t < 3 for t in seq)
                    # Complete this one-round sample from the exact target.
                    while len(seq) < 3:
                        seq.append(int(rng.choice(3, p=p[seq[-1]])))
                    counts[lookup[tuple(seq[:3])]] += 1
            obs = counts/counts.sum()
            tv = float(np.abs(obs-expected).sum()/2)
            zmax = float(np.max(np.abs(obs-expected)/np.sqrt(expected*(1-expected)/counts.sum())))
            case = dict(mode=mode, draws=int(counts.sum()), tv=tv, max_standardized_error=zmax,
                        proposal_length_counts=lengths.tolist(), counts=counts.tolist())
            result['cases'].append(case)
            assert tv < .03 and zmax < 6.5, case
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        out = Path(os.environ['SERVING_SCRATCH'])/'sampler-check.json'
        out.write_text(json.dumps(result, indent=2))
        run.summary.update(result)
        artifact = wandb.Artifact('serving-v14-sampler-check', type='correctness')
        artifact.add_file(str(out)); artifact.add_file(__file__)
        run.log_artifact(artifact); run.finish()
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
