"""Postprocess saved checkpoints only; no parameter or trajectory interpolation."""
import csv,json,math,zipfile
from pathlib import Path
import numpy as np
from storage import read_json,write_json,resolve_artifact,list_artifacts,write_artifact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def csv_write(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 def writer(target):
  with target.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
 write_artifact(path,writer)

def summarize(root):
 root=Path(root);cfg=read_json(root/'protocol.json')
 metrics=read_json(root/'test_metrics.json')
 families=['lognormal','powerlaw','geometric','two_band'];methods=['sgd','muon']
 colors={'sgd':'#355F99','muon':'#C4473A'}
 out=root/'analysis';out.mkdir(exist_ok=True)
 csv_write(out/'all_test_metrics.csv',metrics)
 final=[r for r in metrics if r['step']==cfg['main_steps']]
 csv_write(out/'final_seed_metrics.csv',final)
 aggregates=[]
 for family in families:
  for method in methods:
   for kind,sigma in [('integrated',None)]+[('fixed_sigma',s) for s in cfg['fixed_sigmas']]:
    rows=[r for r in final if r['family']==family and r['method']==method and r['kind']==kind and (sigma is None or r['sigma']==sigma)]
    if not rows:continue
    rec={'family':family,'method':method,'kind':kind,'sigma':sigma,'n_seeds':len(rows)}
    keys=['weighted_bayes_excess_per_coordinate','bayes_excess_per_coordinate','linear_excess_population_per_coordinate',
     'nonlinear_residual_test_per_coordinate','cross_term_test_per_coordinate','high_half_linear_error_fraction','linear_excess_fit_noise_estimate',
     'high_half_absolute_linear_excess_per_coordinate','low_half_absolute_linear_excess_per_coordinate',
     'high_half_linear_fit_noise_per_coordinate','low_half_linear_fit_noise_per_coordinate','training_seconds']
    for key in keys:
     vals=[r[key] for r in rows if key in r]
     if vals:rec[key+'_mean']=float(np.mean(vals));rec[key+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    aggregates.append(rec)
 csv_write(out/'final_aggregate.csv',aggregates)
 # Complete paired comparisons, descriptive across three independent seeds.
 paired=[]
 for family in families:
  for seed in cfg['seeds']:
   values={m:next((r for r in final if r['family']==family and r['seed']==seed and r['method']==m and r['kind']=='integrated'),None) for m in methods}
   if all(values.values()):
    a=values['sgd']['weighted_bayes_excess_per_coordinate'];b=values['muon']['weighted_bayes_excess_per_coordinate']
    paired.append({'family':family,'seed':seed,'sgd_test_excess':a,'muon_test_excess':b,'muon_over_sgd_excess':b/a,
      'sgd_training_seconds':values['sgd']['training_seconds'],'muon_training_seconds':values['muon']['training_seconds']})
 csv_write(out/'paired_final_comparison.csv',paired)
 # Validation-selected checkpoint pairs. Test discrepancies are reported, never reselected.
 selected=read_json(root/'matched_checkpoint_selection.json')['pairs']
 matchrows=[]
 for p in selected:
  if not p['matched']:matchrows.append(p);continue
  indexed={m:[r for r in metrics if r['family']==p['family'] and r['seed']==p['seed'] and r['method']==m and r['step']==p[m+'_step']] for m in methods}
  integrated={m:next((r for r in indexed[m] if r['kind']=='integrated'),None) for m in methods}
  if not all(integrated.values()):
   matchrows.append({**p,'test_complete':False,'test_failure_reason':'Missing integrated test metrics for validation-selected checkpoint pair'});continue
  av=integrated['sgd']['weighted_bayes_excess_per_coordinate'];bv=integrated['muon']['weighted_bayes_excess_per_coordinate']
  base={**p,'sgd_test_excess':av,'muon_test_excess':bv,'test_relative_gap':2*abs(av-bv)/(av+bv),
   'sgd_test_mc_se':integrated['sgd']['weighted_bayes_excess_mc_se'],'muon_test_mc_se':integrated['muon']['weighted_bayes_excess_mc_se']}
  for sigma in cfg['fixed_sigmas']:
   row={**base,'sigma':sigma,'test_complete':True}
   for method in methods:
    r=next((t for t in indexed[method] if t['kind']=='fixed_sigma' and t['sigma']==sigma),None)
    if r is None:
     row['test_complete']=False;row['test_failure_reason']='Missing fixed-noise test metrics for '+method;continue
    for key in ['bayes_excess_per_coordinate','linear_excess_population_per_coordinate','nonlinear_residual_test_per_coordinate','high_half_linear_error_fraction','cross_term_test_per_coordinate',
     'high_half_absolute_linear_excess_per_coordinate','low_half_absolute_linear_excess_per_coordinate','high_half_linear_fit_noise_per_coordinate','low_half_linear_fit_noise_per_coordinate']:
     row[method+'_'+key]=r[key]
   matchrows.append(row)
 csv_write(out/'matched_validation_selected_test_comparison.csv',matchrows)
 # First observed half-error hits; two-band spectra use complete eigenspaces.
 hits=[];diagnostic_issues=[]
 for family in families:
  for seed in cfg['seeds']:
   for method in methods:
    run=root/'runs'/f'{family}_seed{seed}_{method}'
    files=sorted(list_artifacts(run,'diagnostics_*.npz'))
    if not files:continue
    saved_arrays=[]
    for file in files:
     try:
      with np.load(resolve_artifact(file)) as archive:saved_arrays.append((int(file.stem.split('_')[-1]),{key:archive[key] for key in archive.files}))
     except (ValueError,OSError,EOFError,zipfile.BadZipFile):diagnostic_issues.append({'family':family,'seed':seed,'method':method,'file':file.name,'reason':'unreadable diagnostic archive'})
    for k,sigma in enumerate(cfg['fixed_sigmas']):
     required=[f'sigma{k}_{key}' for key in ['lambda','directional_error','directional_fit_noise']]
     valid=[]
     for step,a in saved_arrays:
      if all(key in a and np.all(np.isfinite(a[key])) for key in required) and len({a[key].shape for key in required})==1:
       valid.append((step,a))
      else:diagnostic_issues.append({'family':family,'seed':seed,'method':method,'step':step,'sigma':sigma,'reason':'missing or nonfinite directional diagnostic arrays'})
     initial=next((a for step,a in valid if step==0),None)
     if initial is None:
      diagnostic_issues.append({'family':family,'seed':seed,'method':method,'sigma':sigma,'reason':'no valid step-zero diagnostic; no initial error or hitting time inferred'})
      continue
     lam=initial[f'sigma{k}_lambda'];d=len(lam)
     groups=[np.arange(d//2),np.arange(d//2,d)] if family=='two_band' else [np.array([j]) for j in range(d)]
     for gindex,group in enumerate(groups):
      e0=float(initial[f'sigma{k}_directional_error'][group].sum());first=None;noise=None
      for step,a in valid:
       if float(a[f'sigma{k}_directional_error'][group].sum())<=.5*e0:
        first=step;noise=float(a[f'sigma{k}_directional_fit_noise'][group].sum());break
      hits.append({'family':family,'seed':seed,'method':method,'sigma':sigma,'group':gindex,
       'group_dimension':len(group),'lambda_mean':float(lam[group].mean()),'initial_directional_error':e0,
       'first_observed_half_error_step':first,'right_censored':first is None,'fit_noise_at_hit':noise,
       'note':'Checkpoint observation only; crossings between checkpoints are unknown'})
 csv_write(out/'directional_first_observed_half_error.csv',hits)
 csv_write(out/'diagnostic_coverage_issues.csv',diagnostic_issues)
 # Learning curves display the three seed traces and mean at stored steps.
 fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
 for col,family in enumerate(families):
  for method in methods:
   rows=[r for r in metrics if r['family']==family and r['method']==method and r['kind']=='integrated']
   for seed in cfg['seeds']:
    s=sorted((r for r in rows if r['seed']==seed),key=lambda r:r['step'])
    if s:
     for row,xkey in [(0,'step'),(1,'training_seconds')]:
      axes[row,col].plot([r[xkey] for r in s],[r['weighted_bayes_excess_per_coordinate'] for r in s],color=colors[method],alpha=.35,lw=.9)
   steps=sorted(set(r['step'] for r in rows))
   means=[np.mean([r['weighted_bayes_excess_per_coordinate'] for r in rows if r['step']==step]) for step in steps]
   axes[0,col].plot(steps,means,label=method.upper(),color=colors[method],lw=2)
  axes[0,col].set_title(family.replace('_',' '));axes[0,col].set_xlabel('Training step')
  axes[1,col].set_xlabel('Measured training seconds')
  for row in range(2):axes[row,col].set_yscale('log');axes[row,col].grid(alpha=.2)
 axes[0,0].set_ylabel('Weighted Bayes excess / dimension');axes[1,0].set_ylabel('Weighted Bayes excess / dimension')
 axes[0,0].legend(frameon=False)
 write_artifact(out/'learning_curves.png',lambda path:fig.savefig(path,dpi=180))
 write_artifact(out/'learning_curves.pdf',lambda path:fig.savefig(path));plt.close(fig)
 # Spectral error allocation and nonlinear residual at fixed noise.
 fig,axes=plt.subplots(3,4,figsize=(16,10),constrained_layout=True)
 for i,sigma in enumerate(cfg['fixed_sigmas']):
  for j,family in enumerate(families):
   ax=axes[i,j]
   for method in methods:
    rows=[r for r in metrics if r['family']==family and r['method']==method and r['kind']=='fixed_sigma' and r['sigma']==sigma and r['step'] in cfg['checkpoint_steps']]
    steps=sorted(step for step in set(r['step'] for r in rows) if len({r['seed'] for r in rows if r['step']==step})==len(cfg['seeds']))
    for key,style,label in [('linear_excess_population_per_coordinate','-',method.upper()+' linear'),('nonlinear_residual_test_per_coordinate','--',method.upper()+' residual')]:
     vals=[np.mean([r[key] for r in rows if r['step']==step]) for step in steps]
     ax.plot(steps,vals,style,color=colors[method],label=label,lw=1.5)
   ax.set_yscale('log');ax.grid(alpha=.2);ax.set_title(f'{family.replace("_"," ")}, sigma={sigma:g}');ax.set_xlabel('Training step')
 axes[0,0].legend(frameon=False,fontsize=8);axes[1,0].set_ylabel('Squared error / dimension')
 write_artifact(out/'linear_and_nonlinear_components.png',lambda path:fig.savefig(path,dpi=180))
 write_artifact(out/'linear_and_nonlinear_components.pdf',lambda path:fig.savefig(path));plt.close(fig)
 summary={'full_budget_integrated_results':len([r for r in final if r['kind']=='integrated']),
  'paired_full_budget_comparisons':len(paired),'matched_validation_pairs':sum(p['matched'] for p in selected),
  'requested_match_targets':len(selected),'maximum_decomposition_identity_error':max((r.get('decomposition_identity_error',0) for r in metrics),default=0),
  'diagnostic_coverage_issue_count':len(diagnostic_issues),
  'outputs':['analysis/final_aggregate.csv','analysis/paired_final_comparison.csv','analysis/matched_validation_selected_test_comparison.csv',
   'analysis/directional_first_observed_half_error.csv','analysis/learning_curves.png','analysis/linear_and_nonlinear_components.png']}
 write_json(out/'summary.json',summary)
 lines=['# Nonlinear Gaussian denoising results','',
  f"Full 5000-step integrated test results: {summary['full_budget_integrated_results']} of 24 planned runs. Paired comparisons: {len(paired)} of 12.",'',
  'The table compares the prespecified final training step. Losses are integrated EDM-weighted Bayes excess per coordinate; means and standard deviations summarize available independent main seeds. Actual seed counts are recorded in final_aggregate.csv.','',
  '| Spectrum | SGD mean (SD) | Muon mean (SD) | Paired Muon/SGD ratios |',
  '|---|---:|---:|---|']
 for family in families:
  items={m:next((r for r in aggregates if r['family']==family and r['method']==m and r['kind']=='integrated'),None) for m in methods}
  vals={m:(f"{items[m]['weighted_bayes_excess_per_coordinate_mean']:.6g} ({items[m]['weighted_bayes_excess_per_coordinate_sd']:.3g})" if items[m] and items[m]['n_seeds']>1 else 'incomplete') for m in methods}
  ratios=', '.join(f"{r['muon_over_sgd_excess']:.4g}" for r in paired if r['family']==family)
  lines.append(f"| {family} | {vals['sgd']} | {vals['muon']} | {ratios} |")
 lines+=['',f"Validation-selected matched checkpoint pairs: {summary['matched_validation_pairs']} of {summary['requested_match_targets']} requested targets. Unmatched targets remain explicitly recorded. Selection used no test results.",'',
  'Matching integrated EDM excess does not match the conditional loss at each fixed noise level. The matched comparison CSV therefore reports conditional Bayes excess, absolute high- and low-spectrum fitted-linear errors, their fit-noise estimates, and nonlinear residuals. A smaller high-spectrum fraction alone is not evidence of a smaller absolute low-spectrum error.','',
  f"Largest held-out decomposition identity discrepancy: {summary['maximum_decomposition_identity_error']:.3g}.",'',
  'Exact-SVD Muon and SGD have different wall-clock costs; both step and measured-training-time curves are provided. This constant-step finite-budget study need not reach the limiting optimum. Search budgets are comparable but adaptively used candidate counts can differ. Three seeds support descriptive comparisons, not a universal optimizer claim.','',
  '![Learning curves](analysis/learning_curves.png)','',
  '![Linear and nonlinear components](analysis/linear_and_nonlinear_components.png)','',
  'Data: [final aggregate](analysis/final_aggregate.csv), [paired final comparison](analysis/paired_final_comparison.csv), [validation-selected matched comparison](analysis/matched_validation_selected_test_comparison.csv), [directional first observed hits](analysis/directional_first_observed_half_error.csv).','']
 write_artifact(root/'RESULTS.md',lambda path:path.write_text('\n'.join(lines),encoding='utf-8'))
 return summary

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
 print(json.dumps(summarize(args.root),indent=2))
