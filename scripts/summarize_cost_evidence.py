"""Generate report tables from completed expanded cost-only experiments."""
import csv,hashlib,json
from pathlib import Path

repo=Path(__file__).resolve().parents[1]
rows=[];sources={}
for version,batch,prompts,mapping in [
 ('v26',32,384,{'fixed':'Tuned global fixed width','original':'Native adaptive'}),
 ('v38',64,768,{'fixed7':'Tuned global fixed width','original':'Native adaptive','domain_fixed':'Prespecified domain-specific fixed widths'}),
]:
 path=repo/'results'/version/f'cost_{version}_summary.json';raw=path.read_bytes();s=json.loads(raw)
 sessions=[json.loads(p.read_text()) for p in path.parent.glob(f'cost_{version}_*-round*.json')]
 assert len(sessions)==16
 reference=json.loads((path.parent/('cost_v26_ar-round0.json' if version=='v26' else 'cost_v38_ar-reference.json')).read_text())['outputs']
 assert len(reference)==prompts
 for session in sessions:
  assert session['status']=='completed' and session['outputs']==reference
  for key,name in [('source_sha256',f'benchmark_cost_{version}.py'),('cost_source_sha256','cost_only_v26.py')]:
   assert session['metadata'][key]==hashlib.sha256((repo/'experiments'/name).read_bytes()).hexdigest()
 sources[version]={'file':str(path.relative_to(repo)),'sha256':hashlib.sha256(raw).hexdigest()}
 for key,label in mapping.items():
  v=s['comparisons'][key];low,high=v['batch_and_round_bootstrap_95']
  rows.append(dict(study=version,batch=batch,prompts=prompts,baseline=label,gain_percent=v['gain_percent'],ci95_low=low,ci95_high=high,all_rounds_positive=min(v['round_gains_percent'])>0,ci_excludes_zero=low>0))
out=repo/'reports';out.mkdir(exist_ok=True)
with (out/'cost_evidence.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report={'claim':'Retaining measured target verification costs yields modest throughput gains over globally tuned fixed widths and native adaptive verification in two expanded studies.','rows':rows,'sources':sources,'limits':['No statistically established improvement over the stronger domain-specific fixed comparator in v38.','Existing vLLM scheduler and confidence head; cost-table ablation only. Online calibration did not produce the gains.','One model pair and H200; FP32 tensor dtype does not establish globally strict IEEE arithmetic.','V26 and v38 used disjoint prompts at their respective launches; MBPP expanded from sanitized to full official test subset in v38.','The exploratory 10–15% chat effects did not reproduce at that magnitude against stronger baselines.','Batch128 follow-up is pending; no extrapolation from smaller batches.']}
(out/'cost_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
lines=[r'\begin{tabular}{rrlrr}',r'\hline',r'Batch & Prompts & Baseline & Gain (\%) & 95\% CI (\%) \\',r'\hline']
for z in rows:lines.append(f"{z['batch']} & {z['prompts']} & {z['baseline']} & {z['gain_percent']:+.2f} & [{z['ci95_low']:+.2f}, {z['ci95_high']:+.2f}] "+r'\\')
lines += [r'\hline',r'\end{tabular}']
(out/'cost_evidence_table.tex').write_text('\n'.join(lines)+'\n')
print('Generated report CSV, JSON, and LaTeX table from five audited comparisons.')
