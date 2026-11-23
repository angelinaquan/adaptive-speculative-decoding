"""Stage an isolated reproduction of the cost batch-32 or batch-64 confirmation study."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['confirmation', 'confirmation-v38'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True, help='New directory; existing directories are never overwritten')
    parser.add_argument('--model-paths', type=Path, help='JSON mapping model IDs to local directories; otherwise use pinned Hugging Face checkpoints')
    parser.add_argument('--stage-only', action='store_true', help='Copy inputs without importing vLLM, downloading models, or starting inference')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    revisions = {
        'Qwen/Qwen3-4B': '1cfa9a7208912126459214e8b04321603b3df60c',
        'deepseek-ai/dspark_qwen3_4b_block7': '3457dff1417cb84927f6098a5fcb7cee85c934b7',
    }
    if args.model_paths:
        models = json.loads(args.model_paths.read_text())
    elif args.stage_only:
        models = {name: f'/path/to/{name}' for name in revisions}
    else:
        from huggingface_hub import snapshot_download
        models = {name: snapshot_download(repo_id=name, revision=revision)
                  for name, revision in revisions.items()}
    assert set(models) == {'Qwen/Qwen3-4B', 'deepseek-ai/dspark_qwen3_4b_block7'}
    assert all(isinstance(p, str) and p for p in models.values())
    version = 'v26' if args.phase == 'confirmation' else 'v38'
    files = [repo/'experiments'/name for name in (
        'cost_only_v26.py', f'benchmark_cost_{version}.py', f'analyze_cost_{version}.py',
        f'cost_{version}_manifest.json', f'cost_{version}_plan.json',
    )]
    if version == 'v26':
        files.append(repo/'results/v25/cost_v25_summary.json')
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.stage_only:
        for path in models.values():
            if not Path(path).is_dir():
                raise FileNotFoundError(f'Model directory unavailable: {path}')
    work = args.run_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=False)
    for path in files:
        shutil.copy2(path, work/path.name)
    (work/'model_paths.json').write_text(json.dumps(models, indent=2)+'\n')
    entry = f'benchmark_cost_{version}.py'
    scratch = work/'scratch'
    (scratch/'tmp').mkdir(parents=True)
    (scratch/'cache').mkdir()
    overrides = {
        'WANDB_MODE': os.environ.get('WANDB_MODE', 'offline'),
        'SERVING_SCRATCH': str(scratch), 'TMPDIR': str(scratch/'tmp'),
        'XDG_CACHE_HOME': str(scratch/'cache'), 'VLLM_CACHE_ROOT': str(scratch/'cache/vllm'),
        'TORCHINDUCTOR_CACHE_DIR': str(scratch/'cache/inductor'),
        'TRITON_CACHE_DIR': str(scratch/'cache/triton'), 'CUDA_CACHE_PATH': str(scratch/'cache/cuda'),
        'FLASHINFER_WORKSPACE_BASE': str(scratch),
        'FLASHINFER_AUTOTUNE_CACHE_DIR': str(scratch/'cache/flashinfer-autotune'),
        'VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN': '512',
        'VLLM_BATCH_INVARIANT': '1', 'NVIDIA_TF32_OVERRIDE': '0', 'OMP_NUM_THREADS': '4',
        'PYTHONPATH': str(work)+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''),
    }
    (work/'reproduction.json').write_text(json.dumps(dict(
        phase=args.phase, benchmark=entry, environment=overrides,
        note='Confirmation reuses the published validation selection and frozen prompts; a reproduction is not a new unseen test.',
    ), indent=2)+'\n')
    print(f'Staged {args.phase} in {work}', flush=True)
    if args.stage_only:
        return
    plan = json.loads((work/f'cost_{version}_plan.json').read_text())
    sessions = [(role, rd) for rd, order in enumerate(plan['orders']) for role in order]
    if version == 'v38':
        sessions.insert(0, ('ar', -1))
    for role, rd in sessions:
        command = [sys.executable, str(work/entry), '--role', role, '--round', str(rd)]
        subprocess.run(command, cwd=work, env={**os.environ, **overrides}, check=True)
    analyzer = f'analyze_cost_{version}.py'
    subprocess.run([sys.executable, str(work/analyzer), str(work), '--output', str(work/'summary.json')], check=True)


if __name__ == '__main__':
    main()
