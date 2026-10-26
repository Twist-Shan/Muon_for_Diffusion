"""Independent analytic checks for the continuous-magnitude flow integrator.

Run with: C:/anaconda3/python.exe -s -B validate_flow.py
The rank-one reference uses scalar quadrature/root finding, not the matrix
integrator or its envelope helper. JSON records every tolerance and error.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from flow import solve_muon


SEED = 1729
ROUND_OFF_TOL = 2e-10


def orthogonal(rng, d):
    return np.linalg.qr(rng.standard_normal((d, d)))[0]


def modal_error(actual, expected, q0):
    active = q0 > 1e-14 * max(float(np.max(q0)), 1.0)
    relative = float(np.max(np.abs(actual[:, active] - expected[:, active]) /
                            q0[None, active])) if np.any(active) else 0.0
    inactive = float(np.max(np.abs(actual[:, ~active] - expected[:, ~active]))) \
        if np.any(~active) else 0.0
    return max(relative, inactive)


def invariant_checks(tr):
    # gamma and chi are intentionally undefined at extinction. Chi is also
    # intentionally masked when g <= 1e-12*g(0), as documented by flow.py.
    finite = all(np.all(np.isfinite(tr[key]))
                 for key in ('t', 'q', 'F', 'S', 'g', 'rank'))
    finite = finite and bool(np.all(np.isfinite(tr['gamma'][tr['F'] > 0])))
    finite = finite and bool(np.all(np.isfinite(
        tr['chi'][tr['g'] > 1e-12 * tr['g'][0]])))
    rank_increase = int(max(0, np.max(np.diff(tr['rank']))))
    loss_increase = float(max(0.0, np.max(np.diff(tr['F'])) / tr['F'][0]))
    result = dict(finite_observables=bool(finite),
                  rank_increase=rank_increase,
                  modal_increase=float(tr['modal_increase']),
                  relative_loss_increase=loss_increase,
                  relative_step_balance_error=float(tr['balance_error']),
                  terminal_q_exactly_zero=bool(np.all(tr['q'][-1] == 0)),
                  h=float(tr['h']), steps=int(tr['steps']))
    result['passed'] = bool(finite and rank_increase == 0 and
        result['modal_increase'] < ROUND_OFF_TOL and
        loss_increase < ROUND_OFF_TOL and
        result['relative_step_balance_error'] < ROUND_OFF_TOL and
        result['terminal_q_exactly_zero'])
    return result


def check_commuting(rng):
    c = np.array([0.4, 0.9, 1.7, 3.1, 7.0])
    amplitudes = np.array([0.0, 0.4, 0.75, 1.1, 1.4])
    eta = 0.8
    R0 = orthogonal(rng, 5) * amplitudes[None, :]
    tr = solve_muon(R0, c, resolution=256, eta=eta)
    exact_q = np.maximum(amplitudes[None, :] - eta * tr['t'][:, None], 0)**2
    error = modal_error(tr['q'], exact_q, amplitudes**2)
    checks = invariant_checks(tr)
    return dict(passed=bool(error < ROUND_OFF_TOL and checks['passed']),
                max_initial_normalized_modal_error=error,
                exact_completion_time=float(amplitudes.max()/eta),
                numerical_completion_time=float(tr['numerical_extinction_time']),
                includes_initially_absorbed_direction=True, checks=checks)


def check_isotropic(rng):
    R0 = rng.standard_normal((5, 5)) / np.sqrt(5)
    c0, eta = 2.3, 1.2
    c = np.full(5, c0)
    _, singular, vh = np.linalg.svd(R0, full_matrices=False)
    tr = solve_muon(R0, c, resolution=256, eta=eta)
    shrunk = np.maximum(singular[None, :] - eta * tr['t'][:, None], 0)
    exact_q = shrunk**2 @ (vh**2)
    error = modal_error(tr['q'], exact_q, np.sum(R0**2, axis=0))
    g_error = float(np.max(np.abs(tr['g'] - c0 * shrunk.max(axis=1))) /
                    (c0 * singular[0]))
    checks = invariant_checks(tr)
    return dict(passed=bool(max(error, g_error) < ROUND_OFF_TOL and checks['passed']),
                max_initial_normalized_modal_error=error,
                max_initial_normalized_g_error=g_error, checks=checks)


def rank_one_reference(amplitudes, c, eta):
    """Return q(t) from independent scalar quadrature and its arrival time."""
    def speed(s):
        return float(np.linalg.norm(c * amplitudes * np.exp(-c*s)))

    integral, integral_error = quad(speed, 0, np.inf, epsabs=2e-13,
                                    epsrel=2e-13, limit=300)
    completion = integral / eta
    cache = {}

    def q_at(t):
        key = round(float(t), 14)
        if key in cache:
            return cache[key]
        if t <= 0:
            value = amplitudes**2
        elif t >= completion:
            value = np.zeros_like(amplitudes)
        else:
            # Tail quadrature avoids subtracting two nearly equal integrals
            # near extinction; neither branch calls the matrix integrator.
            if t <= completion / 2:
                def residual(s):
                    return quad(speed, 0, s, epsabs=2e-13, epsrel=2e-13,
                                limit=200)[0] - eta*t
                high = 1.0 / c.min()
                while residual(high) < 0:
                    high *= 2
            else:
                def residual(s):
                    return eta*(completion-t) - quad(speed, s, np.inf,
                        epsabs=2e-13, epsrel=2e-13, limit=200)[0]
                high = 1.0 / c.min()
                while residual(high) < 0:
                    high *= 2
            s = brentq(residual, 0.0, high, xtol=2e-13, rtol=2e-13)
            value = amplitudes**2 * np.exp(-2*c*s)
        cache[key] = value
        return value

    return q_at, completion, integral_error / eta


def check_rank_one(rng):
    c = np.array([0.4, 0.9, 1.7, 3.1, 7.0])
    amplitudes = np.array([0.5, -0.9, 0.7, -1.1, 0.4])
    u = rng.standard_normal(5)
    u /= np.linalg.norm(u)
    R0 = np.outer(u, amplitudes)
    eta = 0.8
    q_at, completion, quadrature_error = rank_one_reference(amplitudes, c, eta)
    rows = []
    for resolution in (256, 512, 1024):
        tr = solve_muon(R0, c, resolution=resolution, eta=eta, output_per_scale=160)
        exact_q = np.array([q_at(t) for t in tr['t']])
        q_error = modal_error(tr['q'], exact_q, amplitudes**2)
        exact_F = exact_q @ c
        exact_g = np.sqrt(exact_q @ (c**2))
        rows.append(dict(resolution=resolution,
            max_initial_normalized_modal_error=q_error,
            max_initial_normalized_loss_error=float(np.max(np.abs(tr['F']-exact_F))/exact_F[0]),
            max_initial_normalized_g_error=float(np.max(np.abs(tr['g']-exact_g))/exact_g[0]),
            numerical_completion_time=float(tr['numerical_extinction_time']),
            absolute_completion_time_error=float(abs(tr['numerical_extinction_time']-completion)),
            checks=invariant_checks(tr)))
    errors = np.array([r['max_initial_normalized_modal_error'] for r in rows])
    decreasing = bool(np.all(errors[1:] < errors[:-1]))
    ratios = (errors[:-1] / errors[1:]).tolist()
    # This predeclared tolerance verifies a usable trajectory, not the formal
    # order of the method. All raw errors remain in the report on failure.
    finest_tolerance = 0.01
    return dict(passed=bool(decreasing and errors[-1] < finest_tolerance and
                           all(row['checks']['passed'] for row in rows)),
        exact_completion_time=float(completion),
        quadrature_reported_absolute_error=float(quadrature_error),
        modal_error_strictly_decreases=decreasing,
        modal_error_refinement_ratios=ratios,
        finest_modal_error_tolerance=finest_tolerance, resolutions=rows,
        completion_time_note='Grid arrival errors are reported, not required to decrease monotonically.')


def check_direct_residual(rng):
    c = np.array([0.4, 0.9, 1.7, 3.1, 7.0])
    V, U = orthogonal(rng, 5), orthogonal(rng, 5)
    H0 = (V * np.array([1.1, 1.3, 1.6, 2.0, 2.5])) @ V.T
    R0 = (U @ H0) / c[None, :]
    eta = 0.8
    inv_sqrt = 1 / np.sqrt(c)
    first_loss = float(np.linalg.eigvalsh(H0 * inv_sqrt[:, None] * inv_sqrt[None, :])[0] / eta)
    times = np.linspace(0.0, 0.7*first_loss, 61)

    def residual_rhs(t, flattened):
        R = flattened.reshape(5, 5)
        left, _, right = np.linalg.svd(R*c[None, :], full_matrices=False)
        return (-eta*(left@right)).ravel()

    direct = solve_ivp(residual_rhs, (0.0, times[-1]), R0.ravel(),
                       t_eval=times, method='DOP853', rtol=2e-11, atol=2e-13)
    H_errors, q_errors = [], []
    for t, flattened in zip(direct.t, direct.y.T):
        R = flattened.reshape(5, 5)
        _, singular, vh = np.linalg.svd(R*c[None, :], full_matrices=False)
        H = (vh.T*singular)@vh
        exact = H0-eta*t*np.diag(c)
        H_errors.append(np.linalg.norm(H-exact)/np.linalg.norm(H0))
        q_errors.append(np.max(np.abs(np.sum(R**2, axis=0)-np.sum(exact**2, axis=0)/c**2) /
                               np.sum(R0**2, axis=0)))
    tr = solve_muon(R0, c, resolution=256, eta=eta)
    mask = tr['t'] <= times[-1]
    exact_H = H0[None, :, :]-eta*tr['t'][mask, None, None]*np.diag(c)[None, :, :]
    exact_q = np.sum(exact_H**2, axis=1)/c[None, :]**2
    projection_error = modal_error(tr['q'][mask], exact_q, np.sum(R0**2, axis=0))
    tolerance = 2e-9
    max_H, max_q = float(max(H_errors)), float(max(q_errors))
    return dict(passed=bool(direct.success and max(max_H, max_q, projection_error) < tolerance),
        direct_solver_success=bool(direct.success), direct_solver_message=direct.message,
        comparison_tolerance=tolerance,
        exact_first_rank_loss_time=first_loss, tested_until=float(times[-1]),
        direct_max_relative_H_error=max_H,
        direct_max_initial_normalized_modal_error=max_q,
        projected_flow_max_initial_normalized_modal_error=projection_error,
        rhs_evaluations=int(direct.nfev))


def check_top_tie(rng):
    c = np.array([0.4, 0.9, 1.7, 3.1, 7.0])
    # H0=I has a full top singular-value tie. The right derivative selects
    # min eig(C) regardless of the arbitrary eigenbasis returned by eigh.
    R0 = orthogonal(rng, 5) / c[None, :]
    tr = solve_muon(R0, c, resolution=256)
    error = abs(float(tr['chi'][0])-float(c.min()))
    return dict(passed=bool(error < ROUND_OFF_TOL), chi0=float(tr['chi'][0]),
                expected_chi0=float(c.min()), absolute_error=error)


def run_validation():
    rng = np.random.default_rng(SEED)
    checks = dict(commuting_orthogonal_columns=check_commuting(rng),
                  isotropic_svd_solution=check_isotropic(rng),
                  rank_one_scalar_quadrature=check_rank_one(rng),
                  direct_R_before_first_rank_loss=check_direct_residual(rng),
                  top_singular_value_tie=check_top_tie(rng))
    return dict(passed=bool(all(item['passed'] for item in checks.values())),
                seed=SEED, dtype='float64', thread_limit=1,
                roundoff_check_tolerance=ROUND_OFF_TOL,
                reference_methods='Analytic commuting/SVD solutions; independent scalar quad+brentq; direct DOP853 on R.',
                checks=checks)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        report = run_validation()
    destination = Path(__file__).with_name('validation.json')
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], report=str(destination),
                         checks={name: result['passed'] for name, result in report['checks'].items()})))
    raise SystemExit(0 if report['passed'] else 1)
