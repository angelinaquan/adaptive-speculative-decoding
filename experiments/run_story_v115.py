"""Frozen budget/default experiment. Every method is an actual measured pipeline."""
import json,os,subprocess,time,hashlib
from pathlib import Path
import wandb
from analyze_story_v115 import analyze
r=Path(__file__).resolve().parent;plan=json.loads((r/'story_v115_plan.json').read_text())
assert json.loads((r/'correctness_v112_gate.json').read_text())['passed']
assert json.loads((r/'matched_v112_run.json').read_text())['status']=='completed', 'Existing heldout experiment must complete successfully first'
assert not list(r.glob('story_v115_*_deploy.json'))
run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='profiling-payback-v115',name='8b-profiling-budget-defaults-and-payback',dir=os.environ['SERVING_SCRATCH'],config=plan)
(r/'story_v115_run.json').write_text(json.dumps(dict(status='running',url=run.url),indent=2)+'\n')
status='failed';pipelines=[];references={}
def stage(batch,role,phase,width,order,tag,budget=0,start=0):
 before=time.perf_counter()
 subprocess.run(['/tmp/aquan-forespark-v63/ForeSpark/.venv/bin/python',str(r/'benchmark_story_v115.py'),'--batch',str(batch),'--role',role,'--phase',phase,'--width',str(width),'--order',str(order),'--tag',tag,'--budget',str(budget),'--pipeline-start-ns',str(start)],check=True)
 wall=time.perf_counter()-before;name=f'story_v115_{tag}.json';s=json.loads((r/name).read_text());assert s['status']=='completed' and all(z['parity'] for z in s['records']);return s,wall,name
try:
 for batch in plan['batches']:
  _,wall,_=stage(batch,'ar','reference',0,0,f'b{batch}_reference');references[batch]=wall
  for order,methods in enumerate(plan['method_orders']):
   for method in methods:
    start=time.monotonic_ns();tag=f'b{batch}_{method}_o{order}';stages=[];scores={};budget=0
    if method.startswith('fixed'):selected=int(method[5:])
    else:
     pair=method.startswith('pair');budget=int(method[4:]);widths=[3,7] if pair else list(range(1,8))
     for w in sorted(widths,reverse=bool(order)):
      s,wall,name=stage(batch,'native','profile',w,order,tag+f'_profile{w}',budget)
      scores[w]=sum(z['seconds'] for z in s['records'] if not z['warmup']);stages.append(dict(file=name,width=w,seconds=wall))
     selected=min(scores,key=lambda w:(scores[w],w))
    profile=(time.monotonic_ns()-start)/1e9
    choice=dict(batch=batch,order=order,method=method,budget=budget,selected_width=selected,profiling_wall_seconds=profile,profile_scores=scores,stages=stages,selection_used_heldout=False)
    (r/f'story_v115_{tag}_selection.json').write_text(json.dumps(choice,indent=2)+'\n')
    dep,wall,name=stage(batch,'native','deploy',selected,order,tag+'_deploy',0,start)
    row=dict(**choice,milestones=dep['milestones'],deployment_file=name,deployment_subprocess_seconds=wall);pipelines.append(row)
    (r/'story_v115_progress.json').write_text(json.dumps(dict(status='running',pipelines=pipelines),indent=2)+'\n')
    run.log({'completed_pipelines':len(pipelines),f'b{batch}/{method}/profile_seconds':profile,**{f'b{batch}/{method}/total_seconds_h{h}':m['total_pipeline_seconds'] for h,m in dep['milestones'].items()}})
 analysis=analyze(r,pipelines)
 summary=dict(correctness_passed=True,pipelines=pipelines,common_reference_seconds=references,analysis=analysis,scope=plan['scope'],limitations=plan['limits'])
 (r/'story_v115_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 cols=['batch','order','method','budget','width','profiling_total','engine_initialization','warmup_generation','timed_generation','other_overhead']
 run.log({'profiling_cost_breakdown':wandb.Table(columns=cols,data=[[x[k] for k in cols] for x in analysis['profiling_costs']])})
 for batch in plan['batches']:
  for order in [0,1]:
   rows=[z for z in pipelines if z['batch']==batch and z['order']==order]
   run.log({f'b{batch}/order{order}/observed_total_time':wandb.plot.line_series(xs=[96,384,1536],ys=[[z['milestones'][h]['total_pipeline_seconds'] for h in ['1','4','16']] for z in rows],keys=[z['method'] for z in rows],title=f'Measured total seconds, batch{batch}, order{order}',xname='Requests')})
 run.summary.update({'correctness_passed':True,'completed_pipelines':len(pipelines),'analysis_scope':analysis['limitations']});status='completed'
finally:
 run.summary.update({'status':status,'completed_pipelines':len(pipelines)})
 art=wandb.Artifact('story-v115-profiling-payback','experiment-results')
 for pattern in ['story_v115_*.json','benchmark_story_v115.py','run_story_v115.py','analyze_story_v115.py','model_split_v103_manifest.json','correctness_v112_gate.json']:
  for path in r.glob(pattern):art.add_file(str(path))
 run.log_artifact(art);(r/'story_v115_run.json').write_text(json.dumps(dict(status=status,url=run.url),indent=2)+'\n');run.finish()
