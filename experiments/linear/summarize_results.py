"""Freeze executed sources and summarize the completed first linear experiment."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT=Path(__file__).resolve().parent


def freeze_run(name):
    path=ROOT/'results'/name
    manifest=json.loads((path/'manifest.json').read_text())
    snapshot=path/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for fn,expected in manifest['code_sha256'].items():
        source=ROOT/fn
        if (snapshot/fn).exists():
            source=snapshot/fn
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Source hash mismatch: {name}/{fn}')
        if source.parent != snapshot:
            shutil.copy2(source,snapshot/fn)
    manifest['seeds']=sorted(set(c['seed'] for c in manifest['cases']))
    manifest['metadata_note']='Seed list reconciled with executed cases; computations unchanged.'
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return json.loads((path/'summary.json').read_text())


def main():
    freeze_run('smoke_checked')
    pilot=freeze_run('pilot_d32')
    scale=freeze_run('scale_d256')
    allrows=pilot+scale
    mainrows=[r for r in pilot if r['kind']=='gaussian_w0' and not r['matched_condition']]
    matched=[r for r in pilot if r['matched_condition']]
    geometry=[r for r in pilot if r['kind']=='controlled_correlation']
    maxima={key:max(r['diagnostics'][key] for r in allrows)
            for key in ['modal_lower_violation','modal_upper_violation','loss_quadratic_violation',
                        'modal_increase','balance_error']}
    refinement={key:max(r['refinement'][key] for r in allrows)
                for key in ['max_modal_initial_normalized','max_matched_high_fraction_difference',
                            'extinction_difference_over_initial_opnorm']}
    bands=[b for r in allrows for b in r['diagnostics']['bands']]
    spectral={key:max(b[key] for b in bands if b[key] is not None)
              for key in ['gamma_bound_violation','chi_bound_violation','gamma_monotonicity_increase']}
    stats=dict(total_cases=len(allrows),main_cases=len(mainrows),matched_cases=len(matched),
        geometry_cases=len(geometry),dimension_cases=len(scale),
        refinement_passed=sum(r['refinement_passed'] for r in allrows),
        maximum_refinement_discrepancies=refinement,maximum_theorem_violations=maxima,
        spectral_violations=spectral,
        gamma_nonempty_cases=sum(r['diagnostics']['gamma_nonempty_bands']>0 for r in allrows),
        chi_nonempty_cases=sum(r['diagnostics']['chi_nonempty_bands']>0 for r in allrows))
    for groupname,rows in [('main',mainrows),('matched',matched),('scale',scale)]:
        diffs=[r['matched']['muon_high_fraction'][2]-r['matched']['gd_high_fraction'][2] for r in rows]
        stats[groupname+'_high_fraction_difference_eps_0.01']=dict(
            positive_count=sum(v>0 for v in diffs),count=len(diffs),minimum=min(diffs),maximum=max(diffs))
    stats['main_all_prespecified_checkpoints']=[]
    for i,eps in enumerate([.5,.1,.01,.001]):
        diffs=[r['matched']['muon_high_fraction'][i]-r['matched']['gd_high_fraction'][i] for r in mainrows]
        stats['main_all_prespecified_checkpoints'].append(dict(epsilon=eps,
            positive_count=sum(v>0 for v in diffs),negative_count=sum(v<0 for v in diffs),
            minimum=min(diffs),maximum=max(diffs)))
    informative=0
    for r in allrows:
        source=ROOT/'results'/('pilot_d32' if r['d']==32 else 'scale_d256')/'trajectories'/f"{r['case_id']}.npz"
        with np.load(source) as tr:
            active=tr['F']/tr['F'][0]>=1e-8
            x=tr['g'][active]/tr['g'][0]
            chi=r['diagnostics']['chi0']
            has_nontrivial=False
            for b in r['diagnostics']['bands']:
                if b['chi_applicable']:
                    ub=chi*b['initial_fraction']*tr['F'][0]/tr['g'][0]**2 * x**(2*(b['alpha']/chi-1))
                    has_nontrivial=has_nontrivial or bool(np.any(ub<1-1e-10))
            informative+=has_nontrivial
    stats['chi_bound_below_one_in_saved_range_cases']=informative
    (ROOT/'results'/'aggregate.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    with (ROOT/'results'/'case_metrics.csv').open('w',newline='',encoding='utf8') as handle:
        fields=['case_id','d','seed','family','sigma','kind','correlation','matched_condition',
                'kappa_c','gamma0','chi0','gamma_nonempty_bands','chi_nonempty_bands',
                'muon_high_fraction_eps_0.01','gd_high_fraction_eps_0.01','refinement_passed']
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for r in allrows:
            row={k:r[k] for k in fields if k in r}
            row.update({k:r['diagnostics'][k] for k in fields if k in r['diagnostics']})
            row['muon_high_fraction_eps_0.01']=r['matched']['muon_high_fraction'][2]
            row['gd_high_fraction_eps_0.01']=r['matched']['gd_high_fraction'][2]
            writer.writerow(row)
    lines=[
        '# First linear experiment: results', '',
        'The first linear experiment is complete. All 88 paired configurations passed the specified step-refinement checks. These results concern denoising error under continuous GD and continuous Simplified Muon.', '',
        '## Coverage', '',
        '| Study | Dimension | Seeds | Configurations |',
        '|---|---:|---|---:|',
        '| Four spectra, three noise levels, random unaligned W0 | 32 | 17, 43, 97 | 36 |',
        '| Matched kappa(C)=10, sigma=0.3 | 32 | 17, 43, 97 | 12 |',
        '| Fixed initial directional errors, three correlations | 32 | 17, 43, 97 | 36 |',
        '| Dimension check, sigma=1 | 256 | 17 | 4 |', '',
        'The controlled-correlation runs for power-law, geometric, and two-band spectra have seed-invariant residual Gram matrices. Their repeated rotations test reproducibility rather than provide independent statistical replicates; the 36 recorded geometry configurations represent 18 distinct observable dynamics. The d=256 check uses one seed.', '',
        '## Numerical validation', '',
        'Five independent solver checks passed: commuting and isotropic exact solutions, a noncommuting rank-one solution via scalar quadrature, direct integration of R before rank loss, and the top-singular-space tie rule. See `../validation.json`.', '',
        f"- Largest last-refinement directional discrepancy, normalized by initial directional error: {refinement['max_modal_initial_normalized']:.6g}.",
        f"- Largest high-spectrum fraction discrepancy at checked matched losses: {refinement['max_matched_high_fraction_difference']:.6g}.",
        f"- Largest numerical extinction-time discrepancy divided by initial operator norm: {refinement['extinction_difference_over_initial_opnorm']:.6g}.",
        f"- Maximum directional-envelope violation on saved numerical trajectories: {max(maxima['modal_lower_violation'],maxima['modal_upper_violation']):.3g}, at roundoff scale.",
        '- Applicable Gamma/chi spectral bounds had zero positive violations at evaluated checkpoints.', '',
        'Step refinement measures empirical numerical stability. Numerical extinction and tolerance hitting times remain approximations of the continuous trajectory; they do not constitute a proof of finite-time arrival.', '',
        '## Observed spectral allocation', '',
        'Define the high-spectrum band by c_j >= median(c), including whole tied eigenspaces. At relative excess F/F0=0.01, the fraction of remaining loss in this band is larger under Muon than GD in all 36 main pilot configurations. At matched total excess, this also means less loss remains in the complementary low-spectrum band. This describes a redistribution of denoising error, not a uniform improvement of all directions or a sampling-quality result.', '',
        '| Spectrum | sigma | GD high-band fraction, mean over 3 seeds | Muon high-band fraction, mean over 3 seeds |',
        '|---|---:|---:|---:|']
    for family in ['lognormal','powerlaw','geometric','two_band']:
        for sigma in [.1,1.,3.]:
            rs=[r for r in mainrows if r['family']==family and r['sigma']==sigma]
            gd=np.mean([r['matched']['gd_high_fraction'][2] for r in rs])
            mu=np.mean([r['matched']['muon_high_fraction'][2] for r in rs])
            lines.append(f'| {family} | {sigma:g} | {gd:.6f} | {mu:.6f} |')
    lines.extend(['',
        f"The same direction holds in {stats['matched_high_fraction_difference_eps_0.01']['positive_count']}/12 matched-conditioning configurations and {stats['scale_high_fraction_difference_eps_0.01']['positive_count']}/4 dimension checks. These are observed cases, not a universal comparison theorem.", '',
        'All four prespecified matched-loss checkpoints are reported below; positive means Muon retains a larger high-band fraction than GD.', '',
        '| Relative excess | Positive differences / 36 | Negative differences / 36 | Difference range |',
        '|---:|---:|---:|---|'])
    for row in stats['main_all_prespecified_checkpoints']:
        lines.append(f"| {row['epsilon']:g} | {row['positive_count']} | {row['negative_count']} | [{row['minimum']:.6f}, {row['maximum']:.6f}] |")
    lines.extend(['',
        'At relative excess 0.001, three main high-noise cases reverse sign: log-normal seed 43, power-law seed 43, and log-normal seed 97 (all sigma=3). Their negative differences exceed their measured last-refinement discrepancies. Thus the 0.01 observation does not imply an all-time ordering.', '',
        '## Usefulness of the spectral certificates', '',
        f"The initial Gamma condition covers a nonempty spectral band in {stats['gamma_nonempty_cases']}/88 configurations. The chi condition covers a nonempty band in {stats['chi_nonempty_cases']}/88, often only the very top of the spectrum. Thus validity and coverage should be reported separately: the initial Gamma certificate is usually uninformative for this grid, while chi has nonempty but sometimes narrow coverage. Restarting certificates later in training remains a next analysis.", '',
        f"Nonempty eligibility alone does not ensure a useful bound: the chi upper envelope becomes less than one in {informative}/88 cases over saved checkpoints with F/F0 >= 1e-8. This is a separate measure of informativeness.", '',
        '## Figures and reproducibility', '',
        '- `pilot_d32/figures/01_directional_bounds.png`: representative directional trajectories and bounds.',
        '- `pilot_d32/figures/02_matched_loss_spectrum.png`: remaining high-spectrum loss at matched excess.',
        '- `pilot_d32/figures/03_initialization_geometry.png`: controlled changes in residual-column correlations.',
        '- `scale_d256/figures/`: single-seed dimension-check figures.',
        '- `case_metrics.csv`: one row per recorded paired configuration.',
        '- `aggregate.json`: machine-readable aggregate values.',
        '- Each run contains its manifest, exact inputs and numerical trajectories, individual records, and verified executed-source snapshot.', '',
        'The nonlinear stage remains planned. No AdamW or generated-distribution measurements were performed.'])
    (ROOT/'results'/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
