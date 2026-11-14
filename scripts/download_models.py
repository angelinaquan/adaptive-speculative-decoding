"""Download the exact released checkpoints used in the experiment."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

revisions = {
    'Qwen/Qwen3-4B': '1cfa9a7208912126459214e8b04321603b3df60c',
    'deepseek-ai/dspark_qwen3_4b_block7': '3457dff1417cb84927f6098a5fcb7cee85c934b7',
}
paths = {model: snapshot_download(repo_id=model, revision=revision)
         for model, revision in revisions.items()}
destination = Path(__file__).resolve().parents[1]/'experiments/model_paths.json'
destination.write_text(json.dumps(paths, indent=2)+'\n')
print(f'Wrote {destination}')
