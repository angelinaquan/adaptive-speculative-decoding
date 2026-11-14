"""Causal-feedback replay on exogenous teacher anchors; NOT closed-loop serving."""
import json,hashlib,copy,random
from pathlib import Path
r=Path(__file__).resolve().parent

def choose(states):
 # Fixed Beta(1,1) prior, no tuning. Each successful verified token is a
 # success; only the first rejection is a failure. All-accepted blocks censor failure.
 probs={i:a/(a+b) for i,(a,b) in states.items()};widths={i:1 for i in states}
 for _ in range(2*len(states)):
  eligible=[i for i in states if widths[i]<8]
  i=max(eligible,key=lambda i:(probs[i]**(widths[i]+1),-i));widths[i]+=1
 return widths,probs

def replay(records):
 history={};trace=[]
 for offset in [0,8,16,24,32,40]:
  active={row['task_index']:(row,row['anchors'].index(len(row['input_ids'])+offset)) for row in records if len(row['input_ids'])+offset in row['anchors']}
  states={i:history.get(i,(1,1)) for i in active}
  widths,probs=choose(states) # No current or future acceptance labels passed here.
  for i in sorted(active):
   row,j=active[i];width=widths[i];accepted=row['widths'][str(width)]['accepted'][j]
   remaining=len(row['output_ids'])-offset-1
   trace.append({'task_index':i,'offset':offset,'domain':row['domain'],'alpha_before':states[i][0],'beta_before':states[i][1],'predicted_conditional':probs[i],'width':width,'observed_accepted':accepted,'reward':min(accepted+1,remaining),'fixed3_reward':min(row['widths']['3']['accepted'][j]+1,remaining)})
   a,b=states[i];history[i]=(a+accepted,b+int(accepted<width))
 return trace

if __name__=='__main__':
 from allocation_headroom_v73 import allocate
 raw=(r/'labels_v73_checks.json').read_bytes();data=json.loads(raw);assert data['passed'] and data['status']=='completed'
 records=data['records'];trace=replay(records)
 # Changing unobserved width labels must not change any deployed choice/feedback.
 observed={(x['task_index'],x['offset']):x['width'] for x in trace}
 poison=copy.deepcopy(records);rng=random.Random(18657)
 for row in poison:
  for j,anchor in enumerate(row['anchors']):
   offset=anchor-len(row['input_ids'])
   for w in range(1,9):
    if w!=observed[row['task_index'],offset]:row['widths'][str(w)]['accepted'][j]=rng.randrange(w+1)
 poisoned=replay(poison)
 assert [(x['width'],x['predicted_conditional'],x['observed_accepted']) for x in trace]==[(x['width'],x['predicted_conditional'],x['observed_accepted']) for x in poisoned]
 # Perturb every label from offset24 onward; earlier actions must stay identical.
 future=copy.deepcopy(records)
 for row in future:
  for j,anchor in enumerate(row['anchors']):
   if anchor-len(row['input_ids'])>=24:
    for w in range(1,9):row['widths'][str(w)]['accepted'][j]=rng.randrange(w+1)
 changed=replay(future)
 assert [(x['task_index'],x['offset'],x['width']) for x in trace if x['offset']<=24]==[(x['task_index'],x['offset'],x['width']) for x in changed if x['offset']<=24]
 groups=[]
 for offset in [0,8,16,24,32,40]:
  batch=[x for x in trace if x['offset']==offset];n=len(batch);assert sum(x['width'] for x in batch)==3*n
  rewards=[]
  for row in records:
   if len(row['input_ids'])+offset not in row['anchors']:continue
   j=row['anchors'].index(len(row['input_ids'])+offset);remaining=len(row['output_ids'])-offset-1
   rewards.append([min(row['widths'][str(w)]['accepted'][j]+1,remaining) for w in range(1,9)])
  oracle=allocate(rewards,3*n)
  groups.append({'offset':offset,'states':n,'fixed3_reward':sum(x['fixed3_reward'] for x in batch),'history_reward':sum(x['reward'] for x in batch),'oracle_reward':oracle['reward'],'oracle_budget_used':oracle['draft_tokens']})
 fixed=sum(x['fixed3_reward'] for x in trace);history=sum(x['reward'] for x in trace);oracle=sum(x['oracle_reward'] for x in groups)
 summary={'scope':__doc__,'passed':True,'gates':['Unobserved-label poisoning leaves choices andobservedfeedback identical','Future-label poisoning leaves currentandearlier choices identical','Allactualallocationbudgets exactly3*n','Rewards clippedatEOS/outputboundary'],'input_sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fixed3_reward':fixed,'history_reward':history,'hindsight_oracle_reward':oracle,'history_gain_percent':100*(history/fixed-1),'oracle_gain_percent':100*(oracle/fixed-1),'groups':groups,'limitations':['No heldoutnewprompts; same48reusedprompts. No parameters fittedortuned.','Anchors fixed8tokensapart, notnextstatesfromchosenpolicy; thisisnotvalidclosedlooplatency estimation.','Beta/geometric model shares acceptanceacrosswidths heuristically; noncausalwidtheffects violateexactmodelassumptions.','No throughput ornoveltyclaim.']}
 (r/'history_v74_trace.json').write_text(json.dumps(trace,indent=2)+'\n');(r/'history_v74_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
