"""Recompute every reported v14 throughput from committed raw measurements."""
import json
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
raw = root/'results/v14/raw'
results = {name:json.loads((raw/f'{name}.json').read_text())
           for name in ['ar','fixed7','adaptive']}
reference = results['ar']
rows = json.loads((root/'experiments/splits.json').read_text())
domains = ['alpaca','gsm8k','humaneval']
grouped = [[r for r in rows if r['domain']==d and r['split']=='validation'] for d in domains]
tasks = [grouped[i%3][i//3] for i in range(32)]
assert hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest() == reference['plan']['prompt_sha256']
for name,result in results.items():
    assert result['status'] == 'completed_validation_only', name
    assert result['greedy_gate_passed'], name
    assert result['plan']['prompt_sha256'] == reference['plan']['prompt_sha256'], name
    assert result['outputs'] == reference['outputs'], name
    for batch in [1,8,32]:
        assert len(result['outputs'][str(batch)]) == batch
        rows = [r for r in result['records'] if r['batch']==batch]
        assert sorted(r['repeat'] for r in rows) == [0,1,2]
        expected = sum(map(len,result['outputs'][str(batch)]))
        assert all(r['tokens']==expected and r['seconds']>0 for r in rows)
sampler = json.loads((raw/'sampler-check.json').read_text())
assert sampler['passed']
for case in sampler['cases']:
    assert sum(case['counts']) == case['draws'] == 20000
    assert case['tv'] < .03 and case['max_standardized_error'] < 6.5
stored = json.loads((root/'results/v14/summary.json').read_text())['results']
for batch in [1,8,32]:
    rates = {}
    for name,result in results.items():
        rows = [r for r in result['records'] if r['batch']==batch]
        rates[name] = sum(r['tokens'] for r in rows)/sum(r['seconds'] for r in rows)
    expected = next(r for r in stored if r['batch']==batch)
    for name,rate in rates.items():
        assert abs(rate-expected['rates'][name]) < 1e-9
    delta = 100*(rates['adaptive']/rates['fixed7']-1)
    assert abs(delta-expected['adaptive_vs_fixed7_pct']) < 1e-9
    print(f'Batch {batch:2}: AR {rates["ar"]:.2f}, fixed7 {rates["fixed7"]:.2f}, '
          f'adaptive {rates["adaptive"]:.2f} tok/s; adaptive/fixed7 {delta:+.2f}%')
print('PASS: recorded outputs agree and published throughput numbers reproduce.')
print('This audits existing records; it does not rerun GPU inference or establish significance.')
