import json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
plan=json.loads((r/'fixed_v71_plan.json').read_text());ref=json.loads((r/'fixed_v71_ar0_round-1.json').read_text());assert ref['status']=='completed' and len(ref['outputs'])==192
summary={'scope':plan['scope'],'checks_passed':True,'throughputs':{},'rounds':{},'domain_throughputs':{}}
source=hashlib.sha256((r/'benchmark_fixed_v71.py').read_bytes()).hexdigest();plan_sha=hashlib.sha256((r/'fixed_v71_plan.json').read_bytes()).hexdigest()
for arm in plan['orders'][0]:
 allrows=[]
 for order in [0,1]:
  res=json.loads((r/f'fixed_v71_{arm}_round{order}.json').read_text());assert res['status']=='completed' and res['all_widths_parity_passed']
  assert res['source_sha256']==source and res['plan_sha256']==plan_sha
  assert len(res['records'])==12 and res['config']['max_num_seqs']==64 and res['config']['dtype']=='bfloat16'
  assert res['config']['speculative_config']['num_speculative_tokens']==(8 if arm=='forced3' else int(arm[-1]))
  for row in res['records']:
   wanted=ref['outputs'][row['start']:row['start']+64]
   assert row['parity'] and row['tokens']==sum(map(len,wanted)) and row['output_sha256']==hashlib.sha256(json.dumps(wanted).encode()).hexdigest()
  if arm=='forced3':
   assert len(res['audits'])==4
   for audit in res['audits']:
    assert audit['width']==3 and audit['forced_steps']>0 and audit['target_unchanged'] and audit['logging_disabled']
    assert min(audit[x] for x in ['backbone_graphs','pre_graphs','tail_graphs'])>0
    assert audit['policy_source_sha256']==hashlib.sha256((r/'forced_policy_v71.py').read_bytes()).hexdigest()
  rows=[x for x in res['records'] if x['phase']=='timed'];assert len(rows)==6
  assert {(x['repeat'],x['start']) for x in rows}=={(t,b) for t in [0,1] for b in [0,64,128]}
  allrows+=rows;summary['rounds'][f'{arm}_{order}']=sum(x['tokens'] for x in rows)/sum(x['seconds'] for x in rows)
 summary['throughputs'][arm]=sum(x['tokens'] for x in allrows)/sum(x['seconds'] for x in allrows)
 summary['domain_throughputs'][arm]={d:sum(x['tokens'] for x in allrows if x['domain']==d)/sum(x['seconds'] for x in allrows if x['domain']==d) for d in {x['domain'] for x in allrows}}
best=max((a for a in summary['throughputs'] if a.startswith('native')),key=summary['throughputs'].get)
summary['best_native']=best;summary['forced3_vs_best_native_percent']=100*(summary['throughputs']['forced3']/summary['throughputs'][best]-1)
summary['round_gains_percent']=[100*(summary['rounds'][f'forced3_{i}']/summary['rounds'][f'{best}_{i}']-1) for i in [0,1]]
(r/'fixed_v71_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
