"""Execute the installed graph-candidate builder with CPU-only stand-ins.
This audits candidate construction, not actual GPU dispatch frequencies.
"""
import ast,json,hashlib
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from collections import defaultdict
from itertools import product,groupby
r=Path(__file__).resolve().parent;src=r/'source_v100/v1__worker__gpu__cudagraph_utils.py';tree=ast.parse(src.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CudaGraphManager');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_init_candidates')
class Mode(Enum):
 FULL=1
 PIECEWISE=2
@dataclass(frozen=True)
class Desc:
 cg_mode:Mode
 num_tokens:int
 num_reqs:int|None=None
 uniform_token_count:int|None=None
 max_query_len:int|None=None
 num_active_loras:int=0
class Combined:
 def decode_mode(self):return Mode.FULL
 def mixed_mode(self):return Mode.PIECEWISE
 def separate_routine(self):return True
ns={'defaultdict':defaultdict,'product':product,'groupby':groupby,'round_up':lambda x,n:((x+n-1)//n)*n,'BatchExecutionDescriptor':Desc,'CUDAGraphMode':Mode};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'),ns)
results=[]
for batch,width in [(8,8),(64,3)]:
 sizes=[1,2,4,8,16,32,64,128,192,256,384,512,576]
 obj=SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=sizes,max_cudagraph_capture_size=576),max_num_reqs=batch,decode_query_len=width+1,cudagraph_mode=Combined(),vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(uses_dynamic_speculative_decoding=lambda:False),cache_config=SimpleNamespace(use_kda_recoverssm=False)),varlen_decode=False,lora_capture_cases=[0],_capture_descs={},_candidates={},_maybe_ubatch_twin=lambda d:None)
 ns['_init_candidates'](obj);full=sorted(set(d.num_tokens for d in obj._capture_descs[Mode.FULL]));need=batch*(width+1);assert need in full
 results.append({'batch':batch,'draft_width':width,'uniform_target_tokens':need,'full_graph_candidate_tokens':full,'exact_full_batch_candidate_exists':True,'slack_in_exact_candidate':0})
out={'passed':True,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'results':results,'scope':__doc__,'limitations':['Assumes native uniform decode with FULL_AND_PIECEWISE and adaptive verification disabled, matching saved native benchmark configuration','Does not observe runtime dispatch, tail batches, prefill, or attention work','Does not prove that arbitrary extra tree nodes would be free even when padding exists']};(r/'graph_shapes_v100_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
