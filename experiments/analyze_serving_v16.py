"""Paired held-out batch bootstrap, conditional on observed engine sessions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare_seconds(candidate, baseline, seed=160016):
    """Arrays are rounds x repeats x prompt groups. Identical output lengths required upstream."""
    assert candidate.shape == baseline.shape and candidate.ndim == 3
    assert np.all(candidate > 0) and np.all(baseline > 0)
    # Keep all measurements of a resampled prompt group together.
    c = candidate.sum(axis=(0,1))
    b = baseline.sum(axis=(0,1))
    indices = np.random.default_rng(seed).integers(len(c), size=(10000,len(c)))
    samples = 100*(b[indices].sum(axis=1)/c[indices].sum(axis=1)-1)
    return dict(throughput_gain_pct=float(100*(b.sum()/c.sum()-1)),
        paired_group_bootstrap_95ci_pct=np.quantile(samples,[.025,.975]).tolist(),
        round_gain_pct=(100*(baseline.sum(axis=(1,2))/candidate.sum(axis=(1,2))-1)).tolist(),
        groups=len(c), resamples=10000,
        uncertainty_scope='Prompt-group bootstrap conditional on three observed engine rounds; not a hardware-session population CI')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roles = ['ar','fixed','adaptive']
    records = {}
    for batch in [1,8,32]:
        for role in roles:
            for round_index in range(3):
                path = args.directory/f'{role}-b{batch}-round{round_index}.json'
                record = json.loads(path.read_text())
                assert record['status'] == 'completed', str(path)
                meta = record['metadata']
                assert (meta['role'],meta['batch'],meta['round']) == (role,batch,round_index)
                assert all(c['passed'] for c in record['checks']), str(path)
                record['file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                records[batch,role,round_index] = record
    assert len({r['metadata']['manifest_sha256'] for r in records.values()}) == 1
    assert len({r['metadata']['selection_sha256'] for r in records.values()}) == 1
    assert len({r['metadata']['source_sha256'] for r in records.values()}) == 1
    assert len({json.dumps(r['metadata']['versions'],sort_keys=True) for r in records.values()}) == 1
    target_configs = []
    for record in records.values():
        config = dict(record['metadata']['config'])
        config.pop('speculative_config',None)
        target_configs.append(json.dumps(config,sort_keys=True))
    assert len(set(target_configs)) == 1, 'Target configuration differs across comparisons'
    summary = dict(scope='Fresh held-out greedy inference; existing vLLM scheduler reproduction',
        primary_batch_size=8, batches={}, run_urls=[r['run_url'] for r in records.values()],
        file_sha256={f'{b}/{m}/{s}':r['file_sha256'] for (b,m,s),r in records.items()})
    for batch in [1,8,32]:
        reference = records[batch,'ar',0]['outputs']
        assert len(reference) == 96
        groups = 96//batch
        times = {role:np.zeros((3,2,groups)) for role in roles}
        tokens = {role:0 for role in roles}
        selected_methods = set()
        for role in roles:
            for round_index in range(3):
                record = records[batch,role,round_index]
                assert record['outputs'] == reference
                if role == 'fixed':
                    selected_methods.add(record['metadata']['method'])
                assert len(record['records']) == 2*groups
                seen = set()
                for row in record['records']:
                    repeat, group = row['repeat'],row['group']
                    assert repeat in [0,1] and group in range(groups)
                    assert (repeat,group) not in seen
                    seen.add((repeat,group))
                    assert row['indices'] == list(range(group*batch,(group+1)*batch))
                    assert row['tokens'] == sum(len(reference[i]) for i in row['indices'])
                    assert row['seconds'] > 0
                    times[role][round_index,repeat,group] = row['seconds']
                    tokens[role] += row['tokens']
        assert len(selected_methods) == 1 and len(set(tokens.values())) == 1
        summary['batches'][str(batch)] = dict(selected_fixed=next(iter(selected_methods)),
            tokens_per_second={role:float(tokens[role]/times[role].sum()) for role in roles},
            adaptive_vs_fixed=compare_seconds(times['adaptive'],times['fixed']),
            adaptive_vs_ar=compare_seconds(times['adaptive'],times['ar']))
    primary = summary['batches']['8']
    summary['primary_positive_in_all_observed_rounds_and_bootstrap'] = all(
        primary[key]['paired_group_bootstrap_95ci_pct'][0] > 0 and min(primary[key]['round_gain_pct']) > 0
        for key in ['adaptive_vs_fixed','adaptive_vs_ar'])
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['batches'],indent=2))


if __name__ == '__main__':
    main()
