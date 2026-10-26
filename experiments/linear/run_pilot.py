"""Run paired continuous linear-denoising experiments with refinement checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from cases import make_covariance, make_initial_residual
from flow import solve_muon, diagnostics, curve_discrepancy, at_matched_loss, gd_at_matched_loss

ROOT = Path(__file__).resolve().parent
FAMILIES = {'lognormal':1.0, 'powerlaw':1.0, 'geometric':float(np.log(100)), 'two_band':20.0}
SEEDS = [17,43,97]
EPS = np.array([.5,.1,.01,.001])


def save_json(path, obj):
    # One writer owns this run directory. Direct writes also work on managed
    # Windows filesystems that disallow replacing an existing directory entry.
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf8')


def build_cases(profile):
    if profile == 'smoke':
        return [dict(d=8,seed=17,family=f,parameter=p,sigma=1.0,
                     kind='gaussian_w0',correlation=0.,matched_condition=False)
                for f,p in FAMILIES.items()]
    d = 32 if profile == 'pilot' else 256
    seeds = SEEDS if profile == 'pilot' else [17]
    sigmas = [.1,1.,3.] if profile == 'pilot' else [1.]
    cases=[]
    for seed in seeds:
        for family,parameter in FAMILIES.items():
            for sigma in sigmas:
                cases.append(dict(d=d,seed=seed,family=family,parameter=parameter,
                    sigma=sigma,kind='gaussian_w0',correlation=0.,matched_condition=False))
            if profile == 'pilot':
                cases.append(dict(d=d,seed=seed,family=family,parameter=parameter,
                    sigma=.3,kind='gaussian_w0',correlation=0.,matched_condition=True))
                for rho in [.2,.8,.95]:
                    cases.append(dict(d=d,seed=seed,family=family,parameter=parameter,
                        sigma=.3,kind='controlled_correlation',correlation=rho,matched_condition=False))
    return cases


def case_name(spec):
    return (f"{spec['family']}_d{spec['d']}_s{spec['seed']}_noise{spec['sigma']:g}_"
            f"{spec['kind']}_rho{spec['correlation']:g}_match{int(spec['matched_condition'])}")


def matched_high_fraction(tr, c, mask):
    q = at_matched_loss(tr, EPS)
    return (q[:,mask]@c[mask])/(q@c)


def run_case(spec, out, base_resolution):
    begin = time.perf_counter()
    covariance = make_covariance(spec['d'],spec['seed'],spec['family'],spec['parameter'],
        target_kappa_c=10. if spec['matched_condition'] else None,sigma=spec['sigma'])
    lam, Q = covariance['eigvals'],covariance['Q']
    c = lam + spec['sigma']**2
    init = make_initial_residual(spec['d'],spec['seed'],Q,lam,spec['sigma'],
                                kind=spec['kind'],correlation=spec['correlation'])
    R0 = init['R0_eigen']
    high = c >= np.median(c)
    levels=[]
    previous=None
    last_comparison=None
    resolution=base_resolution
    fine=None
    for level in range(6):
        trajectory=solve_muon(R0,c,resolution=resolution)
        level_info=dict(resolution=resolution,h=float(trajectory['h']),
            numerical_extinction_time=float(trajectory['t'][-1]),steps=int(trajectory['steps']))
        if previous is not None:
            comparison=curve_discrepancy(previous,trajectory)
            comparison['max_matched_high_fraction_difference']=float(np.max(abs(
                matched_high_fraction(previous,c,high)-matched_high_fraction(trajectory,c,high))))
            comparison['extinction_difference_over_initial_opnorm']=comparison['extinction_difference']/np.linalg.norm(R0,2)
            level_info['versus_previous']=comparison
            last_comparison=comparison
        levels.append(level_info)
        fine=trajectory
        passed = (last_comparison is not None and
                  last_comparison['max_modal_initial_normalized'] <= 1e-3 and
                  last_comparison['max_matched_high_fraction_difference'] <= 1e-2 and
                  last_comparison['extinction_difference_over_initial_opnorm'] <= 3e-3)
        # At least h,h/2,h/4 are evaluated even if early agreement looks good.
        if level>=2 and passed:
            break
        previous=trajectory
        resolution *= 2
    diag, env = diagnostics(R0,c,fine)
    q_mu = at_matched_loss(fine,EPS)
    q_gd,t_gd = gd_at_matched_loss(R0,c,EPS)
    frac_mu=(q_mu[:,high]@c[high])/(q_mu@c)
    frac_gd=(q_gd[:,high]@c[high])/(q_gd@c)
    name=case_name(spec)
    np.savez_compressed(out/'trajectories'/f'{name}.npz',
        **{k:fine[k] for k in ['t','q','F','g','S','rank','gamma','chi']},
        eigvals=lam,c=c,R0=R0,Q=Q,W0=init['W0'],
        q_upper=env['q_upper'],q_lower=env['q_lower'],
        matched_epsilons=EPS,matched_muon_q=q_mu,matched_gd_q=q_gd)
    record=dict(**spec,case_id=name,parameter_used=float(covariance['parameter_used']),
        covariance_mean=float(lam.mean()),diagnostics=diag,refinement=last_comparison,
        refinement_passed=bool(passed),refinement_levels=levels,
        matched=dict(epsilons=EPS.tolist(),muon_high_fraction=frac_mu.tolist(),
                     gd_high_fraction=frac_gd.tolist()),
        seconds=time.perf_counter()-begin)
    save_json(out/'records'/f'{name}.json',record)
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',choices=['smoke','pilot','scale'],default='smoke')
    p.add_argument('--run-name',default=None)
    p.add_argument('--base-resolution',type=int,default=512)
    args=p.parse_args()
    name=args.run_name or {'smoke':'smoke','pilot':'pilot_d32','scale':'scale_d256'}[args.profile]
    out=ROOT/'results'/name
    (out/'trajectories').mkdir(parents=True,exist_ok=True)
    (out/'records').mkdir(exist_ok=True)
    specs=build_cases(args.profile)
    code_hash={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'cases.py',ROOT/'flow.py',Path(__file__)]}
    manifest=dict(profile=args.profile,case_count=len(specs),seeds=sorted({s['seed'] for s in specs}),
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        code_sha256=code_hash,base_resolution=args.base_resolution,
        eta_gd=1.0,eta_muon=1.0,
        numerical_method='PSD-resolvent approximation of continuous H flow; at least three step sizes',
        comparison='Spectral allocation at matched relative excess loss; no wall-time advantage claim',
        accuracy_gate=dict(max_modal_initial_normalized=1e-3,max_matched_high_fraction_difference=1e-2,
                           extinction_difference_over_initial_opnorm=3e-3),
        cases=specs)
    if (out/'manifest.json').exists() and list((out/'records').glob('*.json')):
        old_manifest=json.loads((out/'manifest.json').read_text(encoding='utf8'))
        for key in ['code_sha256','cases','base_resolution']:
            if old_manifest.get(key) != manifest[key]:
                raise RuntimeError('Existing run has different code/config; choose a new --run-name')
    save_json(out/'manifest.json',manifest)
    records=[]
    started=time.perf_counter()
    with threadpool_limits(limits=1):
        for i,spec in enumerate(specs):
            record_path=out/'records'/f'{case_name(spec)}.json'
            if record_path.exists():
                # Resume only an unchanged source/config run.
                record=json.loads(record_path.read_text(encoding='utf8'))
                print(f'RESUME {i+1}/{len(specs)} {case_name(spec)}',flush=True)
            else:
                record=run_case(spec,out,args.base_resolution)
                print(json.dumps(dict(case=i+1,total=len(specs),case_id=record['case_id'],
                    seconds=round(record['seconds'],2),refinement_passed=record['refinement_passed'],
                    modal_discrepancy=record['refinement']['max_modal_initial_normalized'],
                    gamma_bands=record['diagnostics']['gamma_nonempty_bands'],
                    chi_bands=record['diagnostics']['chi_nonempty_bands'])),flush=True)
            records.append(record)
            save_json(out/'summary.json',records)
            save_json(out/'status.json',dict(completed=len(records),total=len(specs),
                elapsed_seconds=time.perf_counter()-started,status='running'))
    save_json(out/'status.json',dict(completed=len(records),total=len(specs),
        elapsed_seconds=time.perf_counter()-started,status='completed',
        refinement_passed=sum(r['refinement_passed'] for r in records)))
    print(f'COMPLETED {len(records)} cases -> {out}',flush=True)


if __name__=='__main__':
    main()
