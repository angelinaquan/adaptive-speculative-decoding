"""Draw the README's batch-64 comparisons from the published summary."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = Path(__file__).resolve().parents[1] / 'results'
summary = json.loads((results / 'v38/cost_v38_summary.json').read_text())
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
