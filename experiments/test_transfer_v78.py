"""Synthetic end-to-end check: known winners and audit rejection."""
import tempfile,shutil,json,hashlib,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as td:
 d=Path(td)
 for n in ['analyze_transfer_v78.py','benchmark_transfer_v78.py','transfer_v78_plan.json','transfer_v78_manifest.json']:shutil.copy2(r/n,d/n)
 plan=json.loads((d/'transfer_v78_plan.json').read_text());plan['subsets']=8;(d/'transfer_v78_plan.json').write_text(json.dumps(plan));tasks=json.loads((d/'transfer_v78_manifest.json').read_text());outputs=[[1,2]]*384
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for b in [8,64]:
  (d/f'transfer_v78_b{b}_w0_r-1.json').write_text(json.dumps({'status':'completed','outputs':outputs}))
  for w in range(1,9):
   for order in range(2):
    rows=[]
    for rep in [-2,-1,0,1]:
     for start in range(0,384,b):rows.append(dict(start=start,repeat=rep,tokens=2*b,seconds=b*(1+abs(w-(8 if b==8 else 3))),parity=True,domain=tasks[start]['domain'],split=tasks[start]['split'],output_sha256=hashlib.sha256(json.dumps(outputs[start:start+b]).encode()).hexdigest()))
    x=dict(status='completed',batch=b,width=w,round=order,config=dict(max_num_seqs=b,dtype='bfloat16',enable_prefix_caching=False,speculative_config=dict(num_speculative_tokens=w)),source_sha256=sha(d/'benchmark_transfer_v78.py'),plan_sha256=sha(d/'transfer_v78_plan.json'),manifest_sha256=plan['manifest_sha256'],records=rows,engine_init_seconds=1)
    (d/f'transfer_v78_b{b}_w{w}_r{order}.json').write_text(json.dumps(x))
 subprocess.run([sys.executable,str(d/'analyze_transfer_v78.py')],check=True,stdout=subprocess.DEVNULL)
 s=json.loads((d/'transfer_v78_summary.json').read_text());assert s['passed'] and s['primary_pass'];assert all(z['loss_to_best_percent']==0 for z in s['comparisons'])
 p=d/'transfer_v78_b8_w1_r0.json';x=json.loads(p.read_text());x['records'][0]['parity']=False;p.write_text(json.dumps(x))
 result=subprocess.run([sys.executable,str(d/'analyze_transfer_v78.py')],capture_output=True);assert result.returncode!=0
 print('PASS: known winners; zero oracle loss; corrupted parity rejected.')
