import json,copy,hashlib
from pathlib import Path
from prefix_path_v89 import decode
r=Path(__file__).resolve().parent;s=json.loads((r/'path_v89_checks.json').read_text());plan=json.loads((r/'path_v89_plan.json').read_text());assert s['passed'];assert s['source_sha256']==hashlib.sha256((r/'check_path_v89.py').read_bytes()).hexdigest();rows=[]
for record in s['records']:
 seq=record['input_ids']+record['output_ids']
 for width,entry in record['widths'].items():
  for j,a in enumerate(record['anchors']):
   grid=entry['score_grids'][j];candidates=entry['candidate_ids'][j]
   for objective in ['greedy','raw_sum','geometric','prefix','map']:
    for temp in plan['temperatures'] if objective in ['prefix','map'] else [1.]:
     path=decode(grid,temp,objective);tokens=[candidates[t][c] for t,c in enumerate(path)];accepted=0
     for t,token in enumerate(tokens):
      if token!=seq[a+t+1]:break
      accepted+=1
     if objective=='greedy':assert tokens==entry['proposed_ids'][j] and accepted==entry['accepted'][j]
     rows.append({'prompt':record['task_index'],'domain':record['domain'],'split':record['split'],'anchor':a,'width':int(width),'objective':objective,'temperature':temp,'reward':accepted+1,'tokens':tokens})
def choose(data,w,obj):
 return max(plan['temperatures'],key=lambda t:sum(x['reward'] for x in data if x['split']=='train' and x['width']==w and x['objective']==obj and x['temperature']==t))
poison=copy.deepcopy(rows)
for x in poison:
 if x['split']=='validation':x['reward']=999
results=[]
for w in [3,8]:
 baseline=sum(x['reward'] for x in rows if x['split']=='validation' and x['width']==w and x['objective']=='greedy');models={}
 for obj in ['greedy','raw_sum','geometric','prefix','map']:
  t=choose(rows,w,obj) if obj in ['prefix','map'] else 1.
  if obj in ['prefix','map']:assert t==choose(poison,w,obj)
  val=[x for x in rows if x['split']=='validation' and x['width']==w and x['objective']==obj and x['temperature']==t];total=sum(x['reward'] for x in val)
  models[obj]={'temperature':t,'validation_reward':total,'gain_vs_native_percent':100*(total/baseline-1),'domains':{d:sum(x['reward'] for x in val if x['domain']==d) for d in ['alpaca','gsm8k','mbpp']}}
 results.append({'width':w,'native_reward':baseline,'models':models,'prefix_reward_gate_passed':models['prefix']['validation_reward']>=1.1*baseline})
out={'passed':True,'scope':plan['scope'],'results':results,'selection_poison_passed':True,'prior_art':plan['prior_art'],'limitations':['Model probabilities need not equal acceptance probabilities','Internal split of priortraining workload, not fresh test','Teacher-anchor reward excludes runtime overhead and trajectory changes']};(r/'path_v89_summary.json').write_text(json.dumps(out,indent=2)+'\n');(r/'path_v89_records.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(out,indent=2))
