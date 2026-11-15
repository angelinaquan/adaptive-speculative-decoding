"""Download pinned public8B checkpoints; fail before GPU work if metadata differs."""
import json,hashlib,shutil
from pathlib import Path
from huggingface_hub import snapshot_download
r=Path(__file__).resolve().parent;metadata=json.loads((r/'model_v102_metadata.json').read_text());cache=Path('/data/scene-rep/u/aquan/.cache/huggingface/hub');cache.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(cache).free>24*1024**3,'Insufficient free space for pinned8B checkpoint preparation'
paths={}
for name,info in metadata.items():
 path=Path(snapshot_download(name,revision=info['revision'],cache_dir=cache,allow_patterns=['*.safetensors','*.json','*.model','*.txt','tokenizer*']))
 assert path.name==info['revision'];config=json.loads((path/'config.json').read_text());assert config==info['config'];assert list(path.glob('*.safetensors'))
 paths[name]=str(path)
draft=metadata['RedHatAI/Qwen3-8B-speculator.dflash2']['config'];target=metadata['Qwen/Qwen3-8B']['config'];assert draft['block_size']==8 and draft['sample_from_anchor'] is False;assert draft['speculators_config']['proposal_methods'][0]['speculative_tokens']==7;assert draft['speculators_config']['verifier']['name_or_path']=='Qwen/Qwen3-8B';assert draft['draft_vocab_size']==target['vocab_size'];assert max(draft['aux_hidden_state_layer_ids'])<target['num_hidden_layers']
out={'passed':True,'paths':paths,'draft_width_max':7,'metadata_sha256':hashlib.sha256((r/'model_v102_metadata.json').read_bytes()).hexdigest()};(r/'model_v102_paths.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
