import importlib.util,json,hashlib,shutil
from pathlib import Path
root=Path(importlib.util.find_spec('vllm').origin).parent
out=Path('/data/scene-rep/u/aquan/speculative-tonight-20261109/source_v100');out.mkdir(exist_ok=True)
paths=list((root/'v1/worker/gpu').glob('*cudagraph*.py'))+[root/'v1/worker/gpu/model_runner.py',root/'v1/worker/gpu/dp_utils.py',root/'v1/cudagraph_dispatcher.py',root/'config/compilation.py']
records=[]
for p in paths:
 if p.is_file():
  name=str(p.relative_to(root)).replace('/','__');shutil.copy2(p,out/name);records.append({'source':str(p),'file':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))
