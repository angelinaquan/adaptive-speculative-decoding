"""Select static widths on validation only; never infer significance here."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    methods = ['ar', 'adaptive'] + [f'fixed{k}' for k in range(1,8)]
    results = {m:json.loads((args.directory/f'{m}.json').read_text()) for m in methods}
    reference = results['ar']
    for method, result in results.items():
        assert result['status'] == 'completed_validation_only', method
        assert result['greedy_gate_passed'], method
        assert result['outputs'] == reference['outputs'], method
        assert result['plan']['prompt_sha256'] == reference['plan']['prompt_sha256'], method
        assert result['plan']['batches'] == [1,8,32], method
        assert len(result['records']) == 9, method
        for batch in [1,8,32]:
            assert len(result['outputs'][str(batch)]) == 32, method
            rows = [r for r in result['records'] if r['batch'] == batch]
            assert sorted(r['repeat'] for r in rows) == [0,1,2], method
            expected_tokens = sum(map(len, result['outputs'][str(batch)]))
            assert all(r['tokens'] == expected_tokens and r['seconds'] > 0 for r in rows), method
    summary = dict(scope='Validation selection only; repeats are timing repeats, not independent prompts or evidence of significance',
                   output_equality_passed=True, batches={},
                   input_sha256={m:hashlib.sha256((args.directory/f'{m}.json').read_bytes()).hexdigest() for m in methods},
                   run_urls={m:r['run_url'] for m,r in results.items()})
    for batch in [1,8,32]:
        rates = {}
        for method, result in results.items():
            rows = [r for r in result['records'] if r['batch'] == batch]
            rates[method] = sum(r['tokens'] for r in rows)/sum(r['seconds'] for r in rows)
        best = max((m for m in methods if m.startswith('fixed')), key=lambda m:rates[m])
        summary['batches'][str(batch)] = dict(tokens_per_second=rates, selected_fixed=best,
            adaptive_vs_selected_fixed_pct=100*(rates['adaptive']/rates[best]-1),
            adaptive_vs_ar_pct=100*(rates['adaptive']/rates['ar']-1),
            adaptive_beats_both_descriptively=rates['adaptive']>max(rates[best],rates['ar']))
    args.output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary['batches'], indent=2))


if __name__ == '__main__':
    main()
