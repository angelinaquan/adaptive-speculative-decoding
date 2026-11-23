"""Recompute both paper summaries from the archived sessions on CPU."""
import json
from pathlib import Path
from run_cost_study import stage_inputs
import subprocess
import sys
import tempfile
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_result(results):
    summary = json.loads((results / 'batch64_summary.json').read_text())
    fig, ax = plt.subplots(figsize=(8, 3), layout='constrained')
    baselines = [('original', 'Native adaptive'), ('fixed7', 'Global fixed width'),
                 ('domain_fixed', 'Domain-specific widths')]
    for row, (key, label) in enumerate(baselines):
        comparison = summary['comparisons'][key]
        gain = comparison['gain_percent']
        low, high = comparison['batch_and_round_bootstrap_95']
        ax.hlines(row, low, high, color='#275b70', linewidth=3)
        ax.plot(gain, row, 'o', color='#275b70')
        ax.annotate(f'{gain:+.2f}%', (gain, row), xytext=(0, 8),
                    textcoords='offset points', ha='center')
    ax.axvline(0, color='gray', linestyle=':', zorder=0)
    ax.set(yticks=range(3), yticklabels=[label for _, label in baselines],
           ylim=(2.45, -.55), xlabel='Throughput gain (%) with 95% paired bootstrap interval',
           title='Measured verification costs on 768 held-out prompts at batch 64')
    ax.spines[['right', 'top', 'left']].set_visible(False)
    fig.savefig(results / 'throughput.png', dpi=180)


repo = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    with zipfile.ZipFile(repo / 'results/sessions.zip') as records:
        records.extractall(work)
    for batch in (32, 64):
        version = stage_inputs(repo, work, batch)
        summary = work / f'{version}.json'
        subprocess.run([sys.executable, str(work / f'analyze_cost_{version}.py'),
                        str(work), '--output', str(summary)], stdout=subprocess.DEVNULL, check=True)
        published = repo / 'results' / f'batch{batch}_summary.json'
        if json.loads(summary.read_text()) != json.loads(published.read_text()):
            raise ValueError(f'{version} differs from the published analysis')
        print(f'Batch {batch}: reproduced the complete published summary')
    subprocess.run([sys.executable, str(repo / 'src/audit_cost_tables.py'), str(work),
                    '--output', str(work / 'tables.json')], stdout=subprocess.DEVNULL, check=True)
    print('Verified all eight stored batch-64 cost tables against raw profile samples')
plot_result(repo / 'results')
