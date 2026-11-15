"""Collect native profiling baselines only, after the 8B smoke gate passes."""
import json
import os
import subprocess
from pathlib import Path
import wandb

r = Path(__file__).resolve().parent
plan = json.loads((r / 'model_split_v103_plan.json').read_text())
assert json.loads((r / 'model_v102_summary.json').read_text())['passed']
assert not list(r.glob('model_profile_v103_b*.json')), 'Existing run: inspect instead of overwrite'
run = wandb.init(entity='aquan-world-model', project='18657-speculative-pilot',
                 group='second-model-v103', name='8b-native-profile-only-width-sweep',
                 dir=os.environ['SERVING_SCRATCH'], config=plan)
status, sessions = 'failed', []
schedule = [(b, 'ar', 0, -1) for b in [8, 64]]
for order in range(2):
    batches = [8, 64] if order == 0 else [64, 8]
    widths = list(range(1, 8)) if order == 0 else list(range(7, 0, -1))
    schedule.extend((b, 'native', w, order) for b in batches for w in widths)
try:
    for batch, role, width, order in schedule:
        subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',
                        str(r / 'benchmark_model_profile_v103.py'), '--batch', str(batch),
                        '--role', role, '--width', str(width), '--round', str(order)], check=True)
        result = json.loads((r / f'model_profile_v103_b{batch}_{role}{width}_round{order}.json').read_text())
        assert result['status'] == 'completed' and all(x['parity'] for x in result['records'])
        sessions.append(result)
        run.log({'completed_sessions': len(sessions)})
    baselines = {}
    for batch in [8, 64]:
        costs, rates = {}, {}
        for width in range(1, 8):
            rows = [x for s in sessions if (s['batch'], s['role'], s['width']) == (batch, 'native', width)
                    for x in s['records'] if x['phase'] == 'timed']
            costs[width] = sum(x['seconds'] for x in rows)
            rates[width] = sum(x['tokens'] for x in rows) / costs[width]
        baselines[str(batch)] = dict(selected_width=min(costs, key=lambda w: (costs[w], w)),
                                     profiling_seconds=costs, profiling_tps=rates)
    summary = dict(passed=True, scope='Profiling only: no held-out speedup or method claim',
                   baselines=baselines, sessions=len(sessions),
                   initialization_seconds=sum(s['initialization_seconds'] for s in sessions),
                   generation_seconds_including_warmup=sum(x['seconds'] for s in sessions for x in s['records']),
                   validation_or_test_outcomes_collected=False)
    (r / 'model_profile_v103_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    run.summary.update(summary)
    status = 'completed'
finally:
    run.summary['status'] = status
    artifact = wandb.Artifact('qwen3-8b-v103-native-profile', 'baseline-profiling')
    for pattern in ['model_profile_v103*.json', 'model_split_v103*.json',
                    'benchmark_model_profile_v103.py', 'run_model_profile_v103.py']:
        for path in r.glob(pattern):
            artifact.add_file(str(path))
    run.log_artifact(artifact)
    (r / 'model_profile_v103_run.json').write_text(json.dumps(dict(url=run.url, status=status), indent=2) + '\n')
    run.finish()
