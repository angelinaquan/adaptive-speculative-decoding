"""Cost accounting and descriptive break-even analysis; no extrapolated result as measurement."""
import json,math
from pathlib import Path

def analyze(root,pipelines):
 costs=[];crossings=[]
 for row in pipelines:
  init=warm=timed=stage_wall=0.
  for stage in row['stages']:
   s=json.loads((root/stage['file']).read_text())
   init+=s['initialization_seconds'];stage_wall+=stage['seconds']
   warm+=sum(z['seconds'] for z in s['records'] if z['warmup'])
   timed+=sum(z['seconds'] for z in s['records'] if not z['warmup'])
  residual=row['profiling_wall_seconds']-init-warm-timed
  assert residual>=-0.01
  costs.append(dict(batch=row['batch'],order=row['order'],method=row['method'],budget=row['budget'],width=row['selected_width'],profiling_total=row['profiling_wall_seconds'],engine_initialization=init,warmup_generation=warm,timed_generation=timed,other_overhead=residual,stage_wall=stage_wall))
 for batch in [8,64]:
  for order in [0,1]:
   rows={z['method']:z for z in pipelines if z['batch']==batch and z['order']==order}
   for name,a in rows.items():
    for baseline in ['fixed3','fixed7']+(['pair'+str(a['budget'])] if name.startswith('full') else []):
     if name==baseline:continue
     b=rows[baseline]
     def fit(z):
      m=z['milestones'];slope=(m['16']['total_pipeline_seconds']-m['4']['total_pipeline_seconds'])/(1536-384)
      assert slope>0
      return m['4']['total_pipeline_seconds']-384*slope,slope
     ca,ta=fit(a);cb,tb=fit(b);extra=ca-cb;saving=tb-ta
     observed={h:100*(1-a['milestones'][h]['total_pipeline_seconds']/b['milestones'][h]['total_pipeline_seconds']) for h in ['1','4','16']}
     result=dict(batch=batch,order=order,method=name,baseline=baseline,extra_fixed_seconds=extra,seconds_saved_per_request=saving,measured_time_reduction_percent=observed,estimated_break_even_requests=None)
     if a['selected_width']==b['selected_width']:result['interpretation']='same_width_no_structural_inference_payback; apparent slope difference is timing variation'
     elif extra>0 and saving>0:
      result.update(interpretation='estimated_crossover_not_measured_validation',estimated_break_even_requests=math.ceil(extra/saving))
     elif extra<=0 and saving>=0:result['interpretation']='lower_fitted_cost_and_rate_in_this_order'
     elif extra>=0 and saving<=0:result['interpretation']='no_fitted_payback'
     else:result.update(interpretation='initial_savings_eroded_by_slower_inference',estimated_break_even_requests=math.ceil(extra/saving))
     crossings.append(result)
 return dict(profiling_costs=costs,comparisons=crossings,limitations='Descriptive fits from measured384/1536-request endpoints. Beyond measured horizons is extrapolation, not evidence of realized savings. Two orders are not confidence intervals;96 unique prompts are repeated.')
if __name__=='__main__':
 r=Path(__file__).resolve().parent;s=json.loads((r/'story_v115_summary.json').read_text());print(json.dumps(analyze(r,s['pipelines']),indent=2))
