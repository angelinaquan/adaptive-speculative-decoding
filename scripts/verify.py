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
    gains = summary['domain_gain_percent']
    domains = [('alpaca', 'Chat'), ('gsm8k', 'Math'), ('mbpp', 'Code')]
    baselines = [('fixed7', 'Fixed width 7', '#396CB1'),
                 ('domain_fixed', 'Domain-specific width', '#238447'),
                 ('original', 'Native adaptive', '#7956A2')]
    fig, ax = plt.subplots(figsize=(8.4, 4.7), layout='constrained')
    for index, (key, label, color) in enumerate(baselines):
        positions = [group + (index - 1) * 0.25 for group in range(len(domains))]
        values = [gains[domain][key] for domain, _ in domains]
        bars = ax.bar(positions, values, width=0.235, label=label, color=color)
        ax.bar_label(bars, labels=[f'{value:+.1f}%' for value in values], padding=4)
    ax.set_title('Measured-cost verification: throughput gain at batch 64', pad=12)
    ax.set_ylabel('More output tokens per second (%)')
    ax.set_xticks(range(len(domains)), [label for _, label in domains])
    ax.set_ylim(0, 6.6)
    ax.legend(title='Comparison baseline', loc='upper right', frameon=False)
    ax.grid(axis='y', alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(results / 'throughput.png', dpi=180)
    plt.close(fig)


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
