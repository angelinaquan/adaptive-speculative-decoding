import json,hashlib,copy
from pathlib import Path
r=Path(__file__).resolve().parent;s=json.loads((r/'selector_v88_checks.json').read_text());plan=json.loads((r/'selector_v88_plan.json').read_text());assert s['passed'];scales=[str(float(x)) for x in plan['scales']];records=s['records'];assert len(records)==96
assert s['source_sha256']==hashlib.sha256((r/'check_selector_v88.py').read_bytes()).hexdigest()
for row in records:
 seq=row['input_ids']+row['output_ids']
 for w in [3,8]:
  if not row['anchors']:continue
  entry=row['widths'][str(w)]
  for scale in scales:
   for j,a in enumerate(row['anchors']):
    n=0
    for t,v in enumerate(entry['scales'][scale]['proposed_ids'][j]):
     if v!=seq[a+t+1]:break
     n+=1
    assert n==entry['scales'][scale]['accepted'][j]
    assert n<=entry['candidate_oracle_accepted'][j]
def reward(rows,w,scale):
 return sum(sum(v+1 for v in row['widths'][str(w)]['scales'][scale]['accepted']) for row in rows if row['anchors'])
def choose(rows,w):
 fit=[row for row in rows if row['split']=='train'];return max(scales,key=lambda sc:reward(fit,w,sc))
poison=copy.deepcopy(records)
for row in poison:
 if row['split']=='validation':
  for entry in row['widths'].values():
   for item in entry['scales'].values():item['accepted']=[999]*len(item['accepted'])
results=[]
for w in [3,8]:
 selected=choose(records,w);assert selected==choose(poison,w)
 val=[row for row in records if row['split']=='validation'];base=reward(val,w,'1.0');total=reward(val,w,selected);oracle=sum(sum(x+1 for x in row['widths'][str(w)]['candidate_oracle_accepted']) for row in val if row['anchors'])
 results.append({'width':w,'selected_scale':selected,'validation_native_reward':base,'validation_selected_reward':total,'selected_gain_percent':100*(total/base-1),'validation_candidate_oracle_reward':oracle,'candidate_oracle_gain_percent':100*(oracle/base-1),'all_scales':{sp:{sc:reward([row for row in records if row['split']==sp],w,sc) for sc in scales} for sp in ['train','validation']},'domains':{d:{'native':reward([row for row in val if row['domain']==d],w,'1.0'),'selected':reward([row for row in val if row['domain']==d],w,selected)} for d in ['alpaca','gsm8k','mbpp']}})
out={'passed':True,'scope':plan['scope'],'results':results,'selection_poison_passed':True,'limitations':['Internal split of prior training prompts, not fresh final test','Teacher-anchor token reward, not closed-loop throughput','Oracle candidate membership uses target outcomes','Simple correction scale is a calibration ablation, not new architecture']};(r/'selector_v88_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
