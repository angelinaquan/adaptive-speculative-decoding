"""Descriptive analysis of balanced repeated validation streams, without p-values."""
import argparse,json,math
from pathlib import Path
METHODS=['original_identity','measured_identity','measured_frozen','measured_online100']
def analyze(data):
 assert data['status']=='completed'
 rows=data['records'];lookup={(r['method'],r['round'],r['start']):r for r in rows}
 expected={(m,rd,start) for m in METHODS for rd in range(4) for start in range(0,144,32)}
 assert len(rows)==len(lookup)==80 and set(lookup)==expected
 for r in rows:
  assert math.isfinite(r['seconds']) and r['seconds']>0 and r['tokens']>0
  assert r['count']==min(32,144-r['start'])
  assert r['tokens']==lookup['original_identity',r['round'],r['start']]['tokens']
 result=dict(scope='Reused validation data, one engine, four balanced method orders. No confidence interval or confirmatory significance claim.',run_url=data['run_url'],methods={})
 for m in METHODS:
  selected=[r for r in rows if r['method']==m]
  metrics=dict(tokens_per_second=sum(r['tokens'] for r in selected)/sum(r['seconds'] for r in selected),comparisons={})
  for baseline in ['original_identity','measured_frozen']:
   paired=[lookup[baseline,r['round'],r['start']] for r in selected]
   gains=[]
   for rd in range(4):
    base=sum(x['seconds'] for x in paired if x['round']==rd)
    proposed=sum(x['seconds'] for x in selected if x['round']==rd)
    gains.append(100*(base/proposed-1))
   metrics['comparisons'][baseline]=dict(gain_percent=100*(sum(x['seconds'] for x in paired)/sum(x['seconds'] for x in selected)-1),round_gains_percent=gains)
  states=[x for x in data['states'] if x['method']==m];assert len(states)==4
  budgets=[x for state in states for x in state['budgets'] if x['available']>0]
  metrics['budget_diagnostic']=dict(steps=len(budgets),trimmed_steps=sum(x['budget']<x['available'] for x in budgets),admitted_fraction=sum(x['budget'] for x in budgets)/sum(x['available'] for x in budgets))
  result['methods'][m]=metrics
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=analyze(json.loads(a.input.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
