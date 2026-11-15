"""Amortize the recorded v77 profiling protocol against default width 8.

This allocates measured subset generation and engine initialization costs. It
is not a cold-start benchmark or a separately executed minimal profiling job.
"""
import gzip,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];r=root/'results/v77'
s=json.loads((r/'profile_v77_summary.json').read_text());rates={}
for b in [8,64]:
 for w in [3,8]:
  elapsed=tokens=0
  for order in range(2):
   x=json.loads(gzip.decompress((r/f'profile_v77_b{b}_w{w}_r{order}.json.gz').read_bytes()))
   assert x['status']=='completed'
   for z in x['records']:
    if z['repeat']>=0 and z['split']=='test':elapsed+=z['seconds'];tokens+=z['tokens']
  rates[b,w]=elapsed/tokens
rows=[]
for z in s['pooled']:
 b=z['batch'];k=z['width'];saving=rates[b,8]-rates[b,k];cost=z['profile_generation_seconds'];init=sum(sum(x) for x in s['engine_init_seconds_by_batch_width_round'][[8,64].index(b)])
 rows.append({'batch':b,'profile_prompts':z['profile_total_prompts'],'selected_width':k,'selected_tokens_per_second':1/rates[b,k],'baseline_width8_tokens_per_second':1/rates[b,8],'saved_seconds_per_output_token':saving,'allocated_generation_cost_seconds':cost,'measured_engine_initialization_seconds':init,'generation_only_break_even_tokens':cost/saving if saving>0 else None,'including_measured_initialization_break_even_tokens':(cost+init)/saving if saving>0 else None})
out={'baseline':'Default native width8, not already-tuned batch-specific baseline.','formula':'Break-even tokens = profiling seconds / (baseline seconds per token - selected seconds per token).','limitations':['Generation cost includes selected profiling blocks across all8 widths, two rounds, two warmup and two timed passes per round.','Subset costs are allocated retrospectively, not measured standalone small-profile jobs.','Initialization costs are observed across all16 native sessions for that batch size with compilation caches; not cold-start estimates.','Break-even assumes future throughput matches held-out workload and excludes configuration switching.','At batch8 the selected width equals the default; no amortization from speed gain.'],'rows':rows}
(r/'profiling_cost.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
