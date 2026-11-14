"""Synthetic end-to-end check: known winners, leakage invariance, audit rejection."""
import tempfile,shutil,json,hashlib,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as td:
 d=Path(td)
 for n in ['analyze_profile_v77.py','benchmark_profile_v77.py','profile_v77_plan.json','profile_v77_manifest.json']:shutil.copy2(r/n,d/n)
 plan=json.loads((d/'profile_v77_plan.json').read_text());plan['subsets']=8;(d/'profile_v77_plan.json').write_text(json.dumps(plan));tasks=json.loads((d/'profile_v77_manifest.json').read_text());outputs=[[1,2]]*1344
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for b in [8,64]:
  (d/f'profile_v77_b{b}_w0_r-1.json').write_text(json.dumps({'status':'completed','outputs':outputs}))
  for w in range(1,9):
   for order in range(2):
    rows=[]
    for rep in [-2,-1,0,1]:
     for start in range(0,1344,b):rows.append(dict(start=start,repeat=rep,tokens=2*b,seconds=b*(1+abs(w-(8 if b==8 else 3))),parity=True,domain=tasks[start]['domain'],split=tasks[start]['split'],output_sha256=hashlib.sha256(json.dumps(outputs[start:start+b]).encode()).hexdigest()))
    x=dict(status='completed',batch=b,width=w,round=order,config=dict(max_num_seqs=b,dtype='bfloat16',enable_prefix_caching=False,speculative_config=dict(num_speculative_tokens=w)),source_sha256=sha(d/'benchmark_profile_v77.py'),plan_sha256=sha(d/'profile_v77_plan.json'),manifest_sha256=plan['manifest_sha256'],records=rows,engine_init_seconds=1)
    (d/f'profile_v77_b{b}_w{w}_r{order}.json').write_text(json.dumps(x))
 subprocess.run([sys.executable,str(d/'analyze_profile_v77.py')],check=True,stdout=subprocess.DEVNULL)
 s=json.loads((d/'profile_v77_summary.json').read_text());assert all(g['within1_percent']==100 and g['mean_loss_percent']==0 for g in s['groups']);assert s['baselines_zero_based']['batch']==[7,2]
 for z in s['pooled']:assert z['profile_generation_seconds']>0 and z['test_loss_to_best_percent']==0
 p=d/'profile_v77_b8_w1_r0.json';x=json.loads(p.read_text());x['records'][0]['parity']=False;p.write_text(json.dumps(x))
 result=subprocess.run([sys.executable,str(d/'analyze_profile_v77.py')],capture_output=True);assert result.returncode!=0
 print('PASS: known winners; zero oracle loss; split/feedback isolation; corrupted parity rejected.')
