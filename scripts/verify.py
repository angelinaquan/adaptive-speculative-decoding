"""Recompute both paper summaries from the archived sessions on CPU."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

repo = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    with zipfile.ZipFile(repo / 'results/sessions.zip') as records:
        records.extractall(work)
    for source in (repo / 'experiments').iterdir():
        if source.is_file():
            shutil.copy2(source, work / source.name)
    for version in ('v26', 'v38'):
        summary = work / f'{version}.json'
        subprocess.run([sys.executable, str(work / f'analyze_cost_{version}.py'),
                        str(work), '--output', str(summary)], stdout=subprocess.DEVNULL, check=True)
        published = repo / 'results' / version / f'cost_{version}_summary.json'
        if json.loads(summary.read_text()) != json.loads(published.read_text()):
            raise ValueError(f'{version} differs from the published analysis')
        print(f'{version}: reproduced the complete published summary')
