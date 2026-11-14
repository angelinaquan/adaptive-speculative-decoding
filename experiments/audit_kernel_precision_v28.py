"""Read-only cache/source audit; does not launch kernels or alter the running study."""
import json,hashlib,os,re
from pathlib import Path
s=Path(os.environ['SERVING_SCRATCH']);root=Path(__file__).resolve().parent
records=[]
for path in sorted((s/'cache/triton').glob('*/matmul_kernel_persistent.json')):
 meta=json.loads(path.read_text());ptx=path.with_suffix('.ptx');ttir=path.with_suffix('.ttir')
 text=ptx.read_text() if ptx.exists() else ''
 opcodes=sorted(set(re.findall(r'\b(?:wgmma|mma)\.[A-Za-z0-9_.]+',text)))
 records.append(dict(cache_key=path.parent.name,metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),name=meta.get('name'),architecture=meta.get('arch'),default_dot_input_precision=meta.get('default_dot_input_precision'),ptx_sha256=hashlib.sha256(ptx.read_bytes()).hexdigest() if ptx.exists() else None,opcodes=opcodes,ttir_float32_pointers='!tt.ptr<f32>' in ttir.read_text() if ttir.exists() else None))
assert records
source=s/'env/lib/python3.11/site-packages/vllm/model_executor/determinism/batch_invariant.py'
compiler=s/'env/lib/python3.11/site-packages/triton/backends/nvidia/compiler.py'
result=dict(scope='Cached-kernel and installed-source inspection, not a profiler trace linking every cached kernel to every timed step',kernels=records,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,compiler]},finding='Cached persistent matmul kernels include TF32 PTX despite requested NVIDIA_TF32_OVERRIDE=0. Report FP32 tensors and exact backend/config, not proven globally IEEE FP32 arithmetic. This audit does not change frozen v26 execution or its parity comparisons.')
(root/'kernel_precision_v28.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
