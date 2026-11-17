"""Reconstruct the completed gate from raw records without rerunning inference."""
import json,hashlib
from pathlib import Path

def audit(root):
 gate=json.loads((root/'correctness_v112_gate.json').read_text())
 assert gate['passed'] and gate['sessions']==32
 for key,name in [('manifest_sha256','model_split_v103_manifest.json'),('plan_sha256','matched_v112_plan.json'),('benchmark_sha256','benchmark_matched_v112.py')]:
  assert gate[key]==hashlib.sha256((root/name).read_bytes()).hexdigest()
 ref=json.loads((root/'correctness_v114_b8_ar0_round0.json').read_text());assert len(ref['outputs'])==200
 source_hash=hashlib.sha256((root/'benchmark_correctness_v114.py').read_bytes()).hexdigest()
 n=0;files=[]
 for b in [8,64]:
  for role,w in [('ar',0)]+[('native',w) for w in range(1,8)]:
   for order in [0,1]:
    p=root/f'correctness_v114_b{b}_{role}{w}_round{order}.json';s=json.loads(p.read_text());files.append(p.name)
    assert (s['batch'],s['role'],s['width'],s['round'])==(b,role,w,order)
    assert s['status']=='completed' and s['source_sha256']==source_hash and s['plan_sha256']==gate['plan_sha256'] and s['versions']==ref['versions']
    assert s['config']['hf_overrides']=={'head_dtype':'float32'}
    assert s['config']['dtype']=='bfloat16' and s['config']['max_num_seqs']==b
    if role=='ar':assert s['outputs']==ref['outputs']
    else:assert s['config']['speculative_config']['num_speculative_tokens']==w
    expected_keys={(phase,rep,start) for phase in ['warmup','timed'] for rep in [0,1] for start in range(0,200,b)}
    assert len(s['records'])==len(expected_keys)
    assert {(z['phase'],z['repeat'],z['start']) for z in s['records']}==expected_keys
    for z in s['records']:
     ids=ref['outputs'][z['start']:z['start']+b]
     assert z['parity'] and z['output_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
     assert z['tokens']==sum(map(len,ids));n+=1
 assert n==gate['batch_output_checks']==1856
 return dict(passed=True,sessions=32,batch_output_checks=n,files=files,scope='Independent raw-record reconstruction. Correctness passed; original job failed later in W&B finalization. No heldout results yet.')
if __name__=='__main__':
 r=Path(__file__).resolve().parent;s=audit(r);(r/'correctness_v114_local_audit.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({k:v for k,v in s.items() if k!='files'}))
