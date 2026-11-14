"""Reconstruct the two v48 target tables from saved profiling measurements."""
import json
from pathlib import Path
import numpy as np

def audit(root):
    checked = []
    for role in ['original', 'measured']:
        x = json.loads((root / f'cost_v48_{role}.json').read_text())
        assert x['status'] == 'completed' and x['costs']['mode'] == role
        samples = x['costs']['raw_target_samples']
        sizes = sorted({n for n, value in samples})
        medians = np.array([np.median([v for n, v in samples if n == size]) for size in sizes])
        assert np.isfinite(medians).all() and (medians > 0).all()
        if role == 'original':
            medians = np.maximum.accumulate(medians)
        expected = medians[np.minimum(np.searchsorted(sizes, np.arange(1025)), len(sizes) - 1)]
        np.testing.assert_array_equal(expected, np.array(x['costs']['cost_tables'][1])[:1025])
        checked.append(role)
    return dict(passed=True, checked=checked, positions=1025,
                scope='Independent target-table reconstruction from raw medians at/below graphlimit. No draft/above-limit reconstruction.')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.root), indent=2))
