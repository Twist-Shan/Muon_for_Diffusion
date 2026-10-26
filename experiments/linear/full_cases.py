"""Frozen expanded linear grid; original pilot generators remain unchanged."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np

from cases import make_covariance, make_initial_residual

SEEDS = (17, 43, 97)
BASE = {"lognormal": 1.0, "powerlaw": 1.0,
        "geometric": math.log(100), "two_band": 20.0}
WIDTHS = {"lognormal": (.25, .5, 1., 1.5, 2.),
          "powerlaw": (.25, .5, 1., 1.5, 2.),
          "geometric": tuple(math.log(x) for x in (3, 10, 100, 1000)),
          "two_band": (2., 5., 20., 100.)}


def canonical(spec):
    return {key: value for key, value in spec.items() if key != "study_labels"}


def key_for(spec):
    return json.dumps(canonical(spec), sort_keys=True, separators=(",", ":"), allow_nan=False)


def case_name(spec):
    digest = hashlib.sha256(key_for(spec).encode()).hexdigest()[:12]
    return (f"{spec['family']}_d{spec['d']}_s{spec['seed']}_noise{spec['sigma']:g}_"
            f"p{spec['parameter']:.8g}_{spec['kind']}_rho{spec['correlation']:g}_"
            f"a{spec['init_scale']:g}_m{spec['band_size']}_k{spec['target_kappa_c']}_{digest}")


def build_cases():
    cases = {}

    def add(study, family, seed, d=32, sigma=1., parameter=None,
            kind="gaussian_w0", rho=0., init_scale=.5, band_fraction=.5, target=None):
        spec = dict(d=d, seed=seed, family=family,
                    parameter=float(BASE[family] if parameter is None else parameter),
                    sigma=float(sigma), kind=kind, correlation=float(rho),
                    init_scale=float(init_scale), band_size=int(d * band_fraction) if family == "two_band" else 0,
                    target_kappa_c=float(target) if target is not None else None,
                    matched_condition=target is not None)
        key = key_for(spec)
        if key not in cases:
            cases[key] = dict(**spec, study_labels=[])
        if study not in cases[key]["study_labels"]:
            cases[key]["study_labels"].append(study)

    for seed in SEEDS:
        for family, widths in WIDTHS.items():
            for parameter in widths:
                for sigma in (.1, 1., 3.):
                    add("spectrum_width", family, seed, parameter=parameter, sigma=sigma)
            for sigma in (.1, 1., 3.):
                for scale in (.1, .5, 1., 2.):
                    add("initialization_scale", family, seed, sigma=sigma, init_scale=scale)
            for rho in (0., .2, .5, .8, .95, .99, 1.):
                add("initialization_geometry", family, seed, sigma=.3,
                    kind="controlled_correlation", rho=rho)
            for target in (2., 5., 10., 20.):
                add("matched_conditioning", family, seed, sigma=.3, target=target)
            for d in (64, 128, 256):
                add("dimension", family, seed, d=d)
        for fraction in (.125, .25, .5, .75):
            add("two_band_size", "two_band", seed, sigma=.3, band_fraction=fraction)
    return list(cases.values())


def grid_counts(cases):
    return {"unique_cases": len(cases), "study_memberships": dict(Counter(
        label for spec in cases for label in spec["study_labels"]))}


def generate(spec):
    cov = make_covariance(spec["d"], spec["seed"], spec["family"], spec["parameter"],
                          target_kappa_c=spec["target_kappa_c"], sigma=spec["sigma"])
    if spec["family"] == "two_band" and spec["band_size"] != spec["d"] // 2:
        if spec["target_kappa_c"] is not None:
            raise ValueError("Matched conditioning is defined only for the half-size two-band family")
        m, d = spec["band_size"], spec["d"]
        if not 1 <= m < d:
            raise ValueError("two-band size must be between one and d-1")
        weights = np.full(d, 1. / spec["parameter"])
        weights[:m] = 1.
        lam = weights / weights.mean()
        cov["eigvals"] = lam
        cov["Sigma"] = (cov["Q"] * lam[None, :]) @ cov["Q"].T
    lam, Q = cov["eigvals"], cov["Q"]
    init = make_initial_residual(spec["d"], spec["seed"], Q, lam, spec["sigma"],
                                 kind=spec["kind"], correlation=spec["correlation"])
    if spec["kind"] == "gaussian_w0" and spec["init_scale"] != .5:
        init["W0"] = init["W0"] * (spec["init_scale"] / .5)
        init["R0_eigen"] = init["W0"] @ Q - Q * (lam / (lam + spec["sigma"]**2))[None, :]
    c = lam + spec["sigma"]**2
    # The top spectral block remains informative for all two-band multiplicities.
    high = c == c.max() if spec["family"] == "two_band" else c >= np.median(c)
    return cov, init, c, high


def protocol():
    specs = build_cases()
    return dict(version=1, name="full_linear_v1", **grid_counts(specs), seeds=list(SEEDS),
                case_specs=specs,
                estimand="Continuous population GD versus continuous compact-polar flow; matched relative excess loss",
                coefficients={"eta_gd": 1., "eta_muon": 1.},
                matched_epsilons=[.5, .1, .01, .001, .0001],
                base_resolution={"d_below_256": 512, "d_at_least_256": 256},
                maximum_refinement_levels=8,
                gates={"max_modal_initial_normalized": .001,
                       "max_matched_high_fraction_difference": .01,
                       "extinction_difference_over_initial_opnorm": .003},
                high_band="two_band: entire largest-eigenvalue eigenspace; other families: c >= median(c), retaining ties",
                historical_reuse="Exact source, version, specification, eigenvalues, Q, W0 and R0 verified; recompute the penultimate grid and apply the expanded epsilon gate",
                reporting={"rho_endpoints": "rho=0 and rho=1 are analytic/rank-boundary numerical controls, not generic full-rank cases",
                           "geometry_seed_interpretation": "Deterministic spectra with fixed Gram matrix have identical observables across rotation seeds; these are reproducibility checks",
                           "precision": "Observed adjacent-grid discrepancies, not certified continuous-time error bounds; differences below their measured numerical uncertainty are unresolved",
                           "failures": "Retain all failure records and failed refinement gates; do not silently omit cases",
                           "scope": "No discrete-optimizer speed, learning-rate fairness, or nonlinear generalization claim"})
