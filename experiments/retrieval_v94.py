"""Causal repetition lookup screen, prior-art-style baseline, no novelty claim."""
import json
from pathlib import Path

def lookup(prefix,width,min_match):
 # Longest suffix first, newest previous occurrence first. Continuation must be observed.
 for n in range(min(32,len(prefix)-width),min_match-1,-1):
  suffix=prefix[-n:]
  for end in range(len(prefix)-width,n-1,-1):
   if prefix[end-n:end]==suffix:return prefix[end:end+width],n
 return None,0

def accepted(draft,truth):
 n=0
 for a,b in zip(draft,truth):
  if a!=b:break
  n+=1
 return n+1

def run():
 r=Path(__file__).resolve().parent;raw=json.loads((r/'path_v89_checks.json').read_text());rows={3:[],8:[]}
 for record in raw['records']:
  seq=record['input_ids']+record['output_ids']
  for j,a in enumerate(record['anchors']):
   prefix=seq[:a+1]
   for width in [3,8]:
    truth=seq[a+1:a+1+width];assert len(truth)==width
    native=record['widths'][str(width)]['proposed_ids'][j]
    draft,n=lookup(prefix,width,2)
    rows[width].append({'prompt':record['task_index'],'domain':record['domain'],'native':accepted(native,truth),'retrieval':accepted(draft,truth) if draft else None,'match':n})
 results=[]
 for w,data in rows.items():
  fit=[x for x in data if x['prompt']%2==0];screen=[x for x in data if x['prompt']%2==1];thresholds=[999,2,3,4,8,16]
  def value(x,t):return x['retrieval'] if x['match']>=t else x['native']
  vals=[sum(value(x,t) for x in fit) for t in thresholds];t=thresholds[max(range(len(vals)),key=lambda k:vals[k])]
  base=sum(x['native'] for x in screen);hybrid=sum(value(x,t) for x in screen);oracle=sum(max(x['native'],x['retrieval'] or 0) for x in screen)
  results.append({'width':w,'fit_rewards':dict(zip(map(str,thresholds),vals)),'selected_threshold':t,'screen_states':len(screen),'lookup_available':sum(x['match']>=2 for x in screen),'lookup_used':sum(x['match']>=t for x in screen),'native_reward':base,'hybrid_reward':hybrid,'hybrid_gain_percent':100*(hybrid/base-1),'hindsight_routing_oracle_reward':oracle,'oracle_gain_percent':100*(oracle/base-1)})
 out={'scope':'Previously exposed v89 training-pool even/odd prompt split. Lookup receives only prompt and verified prefix; never future output. Native fallback wins ties. Replay reward only; no runtime or novelty claim.','results':results};(r/'retrieval_v94_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
