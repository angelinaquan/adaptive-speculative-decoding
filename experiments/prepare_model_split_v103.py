"""Reserve fresh evaluation texts before any second-model outcome collection."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = 'model_split_v103'
norm = lambda s: ' '.join(s.lower().split())
sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    destinations = [ROOT / f'{PREFIX}_{s}.json' for s in ('manifest', 'audit', 'plan')]
    assert not any(p.exists() for p in destinations), 'Do not overwrite frozen split'
    seen, task_ids, scanned = set(), set(), {}

    def collect(x):
        if isinstance(x, dict):
            for key in ('text', 'prompt', 'question'):
                if isinstance(x.get(key), str):
                    seen.add(norm(x[key]))
            if x.get('domain') == 'mbpp' and 'task_id' in x:
                task_ids.add(x['task_id'])
            for value in x.values():
                collect(value)
        elif isinstance(x, list):
            for value in x:
                collect(value)

    # Workload manifests and fit datasets are the text-exposure inventory.
    # Timing outputs contain token arrays and may be cloud-offloaded; avoid
    # treating their absence as evidence of unseen prompts.
    for path in sorted(ROOT.glob('*manifest*.json')) + sorted(ROOT.glob('*fit.jsonl')):
        if path.name.startswith(('mbpp_official', PREFIX)):
            continue
        raw = path.read_bytes()
        try:
            values = [json.loads(line) for line in raw.splitlines()] if path.suffix == '.jsonl' else [json.loads(raw)]
        except (ValueError, UnicodeError):
            continue
        scanned[path.name] = sha(raw)
        for value in values:
            collect(value)

    source_paths = {
        'alpaca': ROOT.parent / 'speculative-baseline/DeepSpec/eval_datasets/alpaca.jsonl',
        'gsm8k': ROOT / 'gsm8k_official_train.jsonl',
        'mbpp': ROOT / 'mbpp_official_full.jsonl',
    }
    full = [json.loads(line) for line in source_paths['mbpp'].read_text().splitlines()]
    sanitized = json.loads((ROOT / 'mbpp_official_sanitized.json').read_text())
    task_ids.update(x['task_id'] for x in full if norm(x['text']) in seen)
    task_ids.update(x['task_id'] for x in sanitized if norm(x['prompt']) in seen)
    smoke = json.loads((ROOT / 'vocab_v54_manifest.json').read_text())[:192]
    smoke_texts = {norm(x['text']) for x in smoke}
    output, audit = [], {}
    for domain, path in source_paths.items():
        source = [json.loads(line) for line in path.read_text().splitlines()]
        rows = []
        for i, x in enumerate(source):
            text = x['turns'][0] if domain == 'alpaca' else x['question'] if domain == 'gsm8k' else x['text']
            row = dict(text=text, source_index=i)
            if domain == 'mbpp':
                row['task_id'] = x['task_id']
            if len(text) <= 3000:
                rows.append(row)
        rows = list({norm(x['text']): x for x in rows}.values())
        fresh = [x for x in rows if norm(x['text']) not in seen and (domain != 'mbpp' or x['task_id'] not in task_ids)]
        random.Random(f'v103/fresh/{domain}').shuffle(fresh)
        assert len(fresh) >= 96, (domain, len(fresh), 'Fresh evaluation pool exhausted')
        validation, test = fresh[:32], fresh[32:96]
        held = {norm(x['text']) for x in validation + test}
        pool = [x for x in rows if norm(x['text']) not in held | smoke_texts]
        random.Random(f'v103/profile/{domain}').shuffle(pool)
        profile = pool[:64]
        assert len(profile) == 64
        audit[domain] = dict(fresh_available=len(fresh), profile_reused_texts=sum(norm(x['text']) in seen for x in profile))
        for split, subset in [('profile', profile), ('validation', validation), ('test', test)]:
            for x in subset:
                output.append(dict(domain=domain, split=split, **x, prompt_sha256=sha(x['text'].encode())))
    assert len(output) == 480 and len({norm(x['text']) for x in output}) == 480
    assert not ({norm(x['text']) for x in output} & smoke_texts)
    assert all(norm(x['text']) not in seen for x in output if x['split'] != 'profile')
    mbpp_ids = [x['task_id'] for x in output if x['domain'] == 'mbpp']
    assert len(mbpp_ids) == len(set(mbpp_ids))
    manifest = (json.dumps(output, indent=2) + '\n').encode()
    plan = dict(
        scope='Frozen second-model data split; no proposed-method or speedup claim.',
        manifest_sha256=sha(manifest),
        counts_per_domain=dict(profile=64, validation=32, test=64),
        widths=list(range(1, 8)), batches=[8, 64], max_tokens=64,
        model_gate='v102 fresh-AR exact parity must pass before collecting profiles.',
        profiling='Sweep native widths 1..7 on profile split only; two opposite engine orders, two warmup and two timed passes. Account for startup and generation separately.',
        selection='Choose strong fixed width separately by batch from pooled profiling seconds; tie chooses smaller width. No validation/test outcomes enter this baseline.',
        validation='Available for explicitly labelled method development; not independent confirmation after use.',
        test='Do not collect test outcomes until method, hyperparameters, baselines, primary metric, timing orders and analysis are frozen in a separate plan.',
        primary_target='At least 10 percent measured steady-state throughput gain over strongest properly tuned applicable baseline, including method overhead. No change to target without user clarification.',
        limitations=['Single H200, second model; freshness is relative to saved workload manifests and fit JSONL inventory, not unrecorded historical inputs or pretraining.', 'Only 64 test prompts per domain; repeated timings are not independent prompt samples.', 'Profiling can reuse prior 4B texts; exact counts are audited.', 'A native width sweep is a baseline, not a proposed method.'],
    )
    audit_output = dict(passed=True, domains=audit,
                        freshness_scope='Disjoint normalized texts and MBPP task IDs against saved workload manifests and fit JSONL files; not a claim about unrecorded historical inputs.',
                        historical_input_sha256=scanned,
                        sources={k:dict(path=str(v), sha256=sha(v.read_bytes())) for k,v in source_paths.items()},
                        sanitized_sha256=sha((ROOT / 'mbpp_official_sanitized.json').read_bytes()),
                        manifest_sha256=sha(manifest), results_collected=False)
    destinations[0].write_bytes(manifest)
    destinations[1].write_text(json.dumps(audit_output, indent=2) + '\n')
    destinations[2].write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(passed=True, domains=audit, manifest_sha256=sha(manifest)), indent=2))


if __name__ == '__main__':
    main()
