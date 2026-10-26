"""Rank-preserving approximation of continuous compact-polar denoising flow.

All matrices here are expressed in the eigenbasis of C. The state H=|RC|
suffices for R^T R, but does not reconstruct W. The resolvent step is
H_new = (H - eta*h*C)_+. Step refinement is mandatory after rank loss.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq


def magnitude(R, c):
    _, singular, vh = np.linalg.svd(R * c[None, :], full_matrices=False)
    return (vh.T * singular) @ vh


def top_chi(values, vectors, c):
    top = values[-1]
    if top <= 0:
        return np.nan
    keep = values >= top - 1e-10 * top
    V = vectors[:, keep]
    return float(np.linalg.eigvalsh(V.T @ (c[:, None] * V))[0])


def solve_muon(R0, c, resolution=2048, eta=1.0, output_per_scale=800):
    """Uniform implicit steps; positive eigenvalues are never tolerance-truncated.

    h=||R0||op/(eta*resolution). Reported numerical extinction is an
    approximation, not an observed exact extinction time of the continuous ODE.
    """
    R0 = np.asarray(R0, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if np.any(c <= 0) or not np.all(np.isfinite(c)) or eta <= 0:
        raise ValueError('C must be positive definite and eta positive')
    d = c.size
    if R0.shape != (d, d) or resolution < 8:
        raise ValueError('Invalid residual shape or step resolution')
    scale = float(np.linalg.norm(R0, 2))
    if scale == 0:
        raise ValueError('Use the absorbing zero solution for a zero residual')
    h = scale / (eta * resolution)
    H = magnitude(R0, c)
    eig, V = np.linalg.eigh(H)
    g0 = eig[-1]
    upper = (1 + .5 * np.sqrt(np.log(c.max() / c.min()))) * np.linalg.norm(R0)
    max_steps = int(np.ceil(1.05 * upper / (eta * h))) + 20
    stride = max(1, resolution // output_per_scale)
    rows = []
    largest_modal_increase = 0.0
    prev_q = np.sum(H * H, axis=0) / c**2
    q0 = prev_q.copy()
    sampled_step_balance_error = 0.0

    def record(step, state, eigenvalues, eigenvectors):
        q = np.sum(state * state, axis=0) / c**2
        F = float(q @ c)
        S = float(np.maximum(eigenvalues, 0).sum())
        g = float(max(eigenvalues[-1], 0))
        rank = int(np.count_nonzero(eigenvalues > 1e-10 * g0))
        gamma = S*S/F if F > 0 else np.nan
        chi = top_chi(eigenvalues, eigenvectors, c) if g > 1e-12*g0 else np.nan
        rows.append((step*h, q, F, S, g, rank, gamma, chi))

    record(0, H, eig, V)
    for step in range(1, max_steps + 1):
        old = H
        candidate = old.copy()
        candidate.flat[::d+1] -= eta * h * c
        eig, V = np.linalg.eigh(candidate)
        eig = np.maximum(eig, 0.0)
        positive = eig > 0.0
        H = (V[:, positive] * eig[positive]) @ V[:, positive].T
        q = np.sum(H * H, axis=0) / c**2
        largest_modal_increase = max(largest_modal_increase,
            float(np.max((q - prev_q) / np.maximum(q0, 1e-30))))
        if step % stride == 0 or not np.any(positive):
            record(step, H, eig, V)
            delta = old - H
            lhs = float(prev_q @ c - q @ c)
            rhs = 2*eta*h*eig.sum() + np.sum(delta*delta / c[None, :])
            sampled_step_balance_error = max(sampled_step_balance_error,
                abs(lhs-rhs) / max(float(q0 @ c), 1e-30))
        prev_q = q
        if not np.any(positive):
            break
    else:
        raise RuntimeError(f'No numerical extinction by {max_steps} steps')
    return dict(t=np.array([r[0] for r in rows]), q=np.array([r[1] for r in rows]),
        F=np.array([r[2] for r in rows]), S=np.array([r[3] for r in rows]),
        g=np.array([r[4] for r in rows]), rank=np.array([r[5] for r in rows]),
        gamma=np.array([r[6] for r in rows]), chi=np.array([r[7] for r in rows]),
        h=h, steps=step, numerical_extinction_time=step*h,
        modal_increase=largest_modal_increase, balance_error=sampled_step_balance_error)


def spectral_groups(c):
    groups = []
    for value in np.unique(c)[::-1]:
        # Covariance generators use exactly repeated values for each band.
        groups.append(np.flatnonzero(c == value))
    return groups


def envelope(R0, c, t, eta=1.0):
    groups = spectral_groups(c)
    cb = np.array([c[ix[0]] for ix in groups])
    b = np.array([np.linalg.norm(R0[:, ix], 2) for ix in groups])
    a2 = np.sum(R0*R0, axis=0)
    def G(s):
        return float(np.linalg.norm(cb * b * np.exp(-cb*s)))
    L, m = c.max(), c.min()
    Ainf = quad(G, 0, 1/L, epsabs=1e-11)[0]
    if L > m:
        Ainf += quad(G, 1/L, 1/m, epsabs=1e-11)[0]
    Ainf += quad(G, 1/m, np.inf, epsabs=1e-11, limit=300)[0]
    sgrid = np.r_[0.0, np.geomspace(1e-9/L, 42/m, 18000)]
    Ggrid = np.linalg.norm(cb[None, :] * b[None, :] *
                           np.exp(-sgrid[:, None]*cb[None, :]), axis=1)
    Agrid = np.r_[0, cumulative_trapezoid(Ggrid, sgrid)]
    quadrature_relative_difference = abs(Agrid[-1]-Ainf)/max(Ainf, 1e-30)
    Agrid *= Ainf/Agrid[-1]
    Aunique, indices = np.unique(Agrid, return_index=True)
    sstar = np.interp(eta*np.asarray(t), Aunique, sgrid[indices])
    sstar[eta*np.asarray(t) >= Ainf] = np.inf
    q_upper = a2[None, :] * np.exp(-2*sstar[:, None]*c[None, :])
    q_lower = np.maximum(np.sqrt(a2)[None, :] - eta*np.asarray(t)[:, None], 0)**2
    return dict(q_upper=q_upper, q_lower=q_lower, Ainf=Ainf, B1=float(b.sum()),
        B2=float(np.linalg.norm(b)), block_count=len(groups),
        time_lower=float(np.linalg.norm(R0, 2)/eta),
        quadrature_relative_difference=quadrature_relative_difference)


def at_matched_loss(tr, epsilons):
    x = tr['F']/tr['F'][0]
    return np.column_stack([np.interp(epsilons, x[::-1], col[::-1])
                            for col in tr['q'].T])


def gd_at_matched_loss(R0, c, epsilons, eta=1.0):
    q0 = np.sum(R0*R0, axis=0)
    F0 = q0 @ c
    out, times = [], []
    for eps in epsilons:
        if eps == 1:
            t = 0.0
        else:
            tmax = np.log(1/eps)/(4*eta*c.min()) * (1+1e-8)
            t = brentq(lambda tt: float((q0*np.exp(-4*eta*c*tt))@c/F0-eps), 0,tmax)
        times.append(t)
        out.append(q0*np.exp(-4*eta*c*t))
    return np.asarray(out), np.asarray(times)


def curve_discrepancy(coarse, fine):
    q0 = fine['q'][0]
    errors = []
    for j in range(q0.size):
        if q0[j] > 1e-20:
            interp = np.interp(fine['t'], coarse['t'], coarse['q'][:,j], right=0)
            errors.append(np.max(np.abs(interp-fine['q'][:,j]))/q0[j])
    qfine = at_matched_loss(fine, np.geomspace(.99, 1e-4, 121))
    qcoarse = at_matched_loss(coarse, np.geomspace(.99, 1e-4, 121))
    return dict(max_modal_initial_normalized=float(max(errors, default=0)),
        max_matched_modal_initial_normalized=float(np.max(abs(qfine-qcoarse)/np.maximum(q0,1e-30))),
        extinction_difference=float(abs(coarse['t'][-1]-fine['t'][-1])))


def diagnostics(R0, c, tr):
    env = envelope(R0, c, tr['t'])
    q, q0, F = tr['q'], tr['q'][0], tr['F']
    relF = F/F[0]
    valid = (relF >= 1e-8)
    gamma0, chi0 = tr['gamma'][0], tr['chi'][0]
    lower_violation = float(np.max(np.maximum(env['q_lower']-q,0)/np.maximum(q0,1e-30)))
    upper_violation = float(np.max(np.maximum(q-env['q_upper'],0)/np.maximum(q0,1e-30)))
    band_rows=[]
    for alpha in np.unique(c):
        mask = c >= alpha
        bandF = q[:,mask] @ c[mask]
        frac = bandF[valid]/F[valid]
        gamma_ok = bool(alpha > gamma0 + 1e-10*c.max())
        chi_ok = bool(alpha > chi0 + 1e-10*c.max())
        ub_gamma = bandF[0]/F[0] * relF[valid]**(alpha/gamma0-1)
        x = tr['g'][valid]/tr['g'][0]
        ub_chi = np.minimum(1., chi0*bandF[0]/tr['g'][0]**2 * x**(2*(alpha/chi0-1)))
        band_rows.append(dict(alpha=float(alpha), dimension=int(mask.sum()),
            gamma_applicable=gamma_ok, chi_applicable=chi_ok,
            initial_fraction=float(bandF[0]/F[0]),
            gamma_bound_violation=float(max(0,np.max(frac-ub_gamma))),
            chi_bound_violation=float(max(0,np.max(frac-ub_chi))) if chi_ok else None,
            gamma_monotonicity_increase=float(max(0,np.max(np.diff(frac)))) if gamma_ok else None))
    eps = np.array([.5,.1,.01,.001,.0001])
    hit = np.interp(eps, relF[::-1], tr['t'][::-1])
    result = dict(kappa_c=float(c.max()/c.min()), gamma0=float(gamma0), chi0=float(chi0),
        gamma_nonempty_bands=sum(row['gamma_applicable'] for row in band_rows),
        chi_nonempty_bands=sum(row['chi_applicable'] for row in band_rows),
        number_spectral_bands=len(band_rows),
        modal_lower_violation=lower_violation, modal_upper_violation=upper_violation,
        loss_quadratic_violation=float(np.max(np.maximum(relF-np.maximum(1-tr['t']/env['B1'],0)**2,0))),
        theoretical_completion_lower=env['time_lower'], theoretical_completion_upper=env['Ainf'],
        numerical_extinction_time=float(tr['t'][-1]),
        tolerance_hitting_times={str(e):float(tt) for e,tt in zip(eps,hit)},
        numerical_step=float(tr['h']), steps=int(tr['steps']),
        quadrature_relative_difference=env['quadrature_relative_difference'],
        modal_increase=float(tr['modal_increase']), balance_error=float(tr['balance_error']),
        bands=band_rows)
    return result, env
