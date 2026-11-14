"""Stage an isolated reproduction of the cost validation or confirmation study."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['validation', 'confirmation'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True, help='New directory; existing directories are never overwritten')
    parser.add_argument('--model-paths', type=Path, help='Defaults to experiments/model_paths.json from download_models.py')
    parser.add_argument('--stage-only', action='store_true', help='Copy inputs without importing vLLM, downloading models, or starting inference')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    model_paths = args.model_paths or repo/'experiments/model_paths.json'
    models = json.loads(model_paths.read_text())
    assert set(models) == {'Qwen/Qwen3-4B', 'deepseek-ai/dspark_qwen3_4b_block7'}
    assert all(isinstance(p, str) and p for p in models.values())
    files = [repo/'experiments'/name for name in [
        'analyze_cost_v25.py', 'benchmark_cost_v25.py', 'run_cost_v25.py',
        'cost_policy_v24.py', 'diagnostic_budget_v22.py', 'head_calibration_v17.py',
        'head_calibration_v18_manifest.json',
    ]]
    files += [repo/'results/v18/head_calibration_v18_ar.json']
    if args.phase == 'confirmation':
        files += [repo/'experiments'/name for name in [
            'cost_only_v26.py', 'benchmark_cost_v26.py', 'run_cost_v26.py',
            'analyze_cost_v26.py', 'cost_v26_manifest.json', 'cost_v26_plan.json',
        ]]
        files += [repo/'results/v25'/f'cost_v25_{method}.json'
                  for method in ['ar', 'original', 'measured']+[f'fixed{i}' for i in range(1, 8)]]
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
    entry = 'run_cost_v25.py' if args.phase == 'validation' else 'run_cost_v26.py'
    scratch = work/'scratch'
    (scratch/'tmp').mkdir(parents=True)
    (scratch/'cache').mkdir()
    overrides = {
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
        phase=args.phase, command=[sys.executable, str(work/entry)], environment=overrides,
        note='Confirmation reuses the published validation selection and frozen prompts; a reproduction is not a new unseen test.',
    ), indent=2)+'\n')
    print(f'Staged {args.phase} in {work}', flush=True)
    if args.stage_only:
        return
    subprocess.run([sys.executable, str(work/entry)], cwd=work, env={**os.environ, **overrides}, check=True)
    analyzer = 'analyze_cost_v25.py' if args.phase == 'validation' else 'analyze_cost_v26.py'
    if (work/'cost_v26_not_launched.json').exists():
        print('Confirmation skipped: published validation gate failed.', flush=True)
        return
    subprocess.run([sys.executable, str(work/analyzer), str(work), '--output', str(work/'summary.json')], check=True)


if __name__ == '__main__':
    main()
