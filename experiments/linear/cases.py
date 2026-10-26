"""Reproducible covariance and initialization cases for linear denoising.

All covariance eigenvalues have arithmetic mean one.  The observation
covariance is C = Sigma + sigma**2 I, and W_star = Sigma C^{-1}.
Separate fixed SeedSequence domains keep the eigenbasis, spectrum draw, and
initialization independent without changing NumPy's global random state.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


_Q_DOMAIN = 14741
_SPECTRUM_DOMAIN = 82711
_W0_DOMAIN = 51383
_RESIDUAL_DOMAIN = 29947
_FAMILIES = {"lognormal", "powerlaw", "geometric", "two_band"}


def _validate_dimension_seed(d: int, seed: int) -> tuple[int, int]:
    if isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError("d must be a positive integer")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    return int(d), int(seed)


def _noise_variance(sigma: float) -> float:
    sigma = float(sigma)
    variance = sigma * sigma
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(variance) or variance <= 0:
        raise ValueError("sigma must be positive with a finite, nonzero square")
    return variance


def _rng(seed: int, domain: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, domain]))


def _orthogonal(d: int, seed: int, domain: int) -> np.ndarray:
    """Gaussian QR with the diagonal of R made nonnegative."""
    q, r = np.linalg.qr(_rng(seed, domain).standard_normal((d, d)))
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return q * signs[None, :]


def make_covariance(
    d: int,
    seed: int,
    family: str,
    parameter: float,
    target_kappa_c: float | None = None,
    sigma: float = 1.0,
) -> dict:
    """Generate a rotated, trace-normalized covariance with a chosen spectrum.

    ``lognormal`` uses exp(beta*z_j), with a fixed sorted standard-normal draw;
    ``powerlaw`` uses j**(-a); ``geometric`` uses exp(-h*(j-1)/(d-1));
    ``two_band`` uses r in the first floor(d/2) positions and one elsewhere.
    beta, a, h must be nonnegative; r must be at least one.  At d=1 all
    families return the unit spectrum.  Q and the normal draw are shared
    across family/parameter choices for the same dimension and seed.

    If ``target_kappa_c`` is supplied, solve for the family parameter using
    Brent's method.  The target is the condition number of C, not Sigma.
    Targets at or above the infinite-parameter limit are rejected, except
    the isotropic target one.  The two-band search uses log(r).  Centered
    log scores avoid exponential overflow; extreme tails may underflow to
    zero, which is valid for Sigma because the noise variance is positive.

    Returns ``eigvals`` (descending), ``Q``, ``Sigma``, ``parameter_used``,
    and ``family``.  No global random seed is read or modified.
    """
    d, seed = _validate_dimension_seed(d, seed)
    noise_variance = _noise_variance(sigma)
    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {sorted(_FAMILIES)}")
    parameter = float(parameter)
    lower_parameter = 1.0 if family == "two_band" else 0.0
    if not math.isfinite(parameter) or parameter < lower_parameter:
        raise ValueError(f"{family} parameter must be finite and >= {lower_parameter:g}")

    q = _orthogonal(d, seed, _Q_DOMAIN)
    z = np.sort(_rng(seed, _SPECTRUM_DOMAIN).standard_normal(d))[::-1]
    if family == "lognormal":
        base_scores = z - z[0]
    elif family == "powerlaw":
        base_scores = -np.log(np.arange(1, d + 1, dtype=np.float64))
    elif family == "geometric":
        base_scores = -np.linspace(0.0, 1.0, d) if d > 1 else np.zeros(1)
    else:
        base_scores = np.full(d, -1.0)
        base_scores[: d // 2] = 0.0
        if d == 1:
            base_scores[:] = 0.0

    def spectrum(search_parameter: float) -> np.ndarray:
        # Every base score is nonpositive with a zero maximum.  Overflow in
        # the product can only produce -inf, hence zero after exponentiation.
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp(search_parameter * base_scores)
        return weights * (d / float(weights.sum()))

    search_parameter = math.log(parameter) if family == "two_band" else parameter
    if target_kappa_c is not None:
        target = float(target_kappa_c)
        if not math.isfinite(target) or target < 1:
            raise ValueError("target_kappa_c must be finite and >= 1")
        if target == 1:
            search_parameter = 0.0
        else:
            if d == 1:
                raise ValueError("d=1 permits only target_kappa_c=1")
            top_multiplicity = int(np.count_nonzero(base_scores == 0.0))
            log_noise = math.log(noise_variance)
            log_limit = float(np.logaddexp(math.log(d / top_multiplicity), log_noise)) - log_noise
            log_target = math.log(target)
            if log_target >= log_limit:
                limit = math.exp(log_limit) if log_limit < math.log(np.finfo(float).max) else math.inf
                raise ValueError(
                    f"target_kappa_c={target:g} is unattainable: {family}, d={d}, "
                    f"sigma={sigma:g} requires target < {limit:g} after mean-one normalization"
                )

            def mismatch(value: float) -> float:
                values = spectrum(value)
                return math.log(values[0] + noise_variance) - math.log(values[-1] + noise_variance) - log_target

            max_search = math.log(np.finfo(float).max) if family == "two_band" else np.finfo(float).max
            # Start independently of the supplied shape parameter: a huge
            # default parameter must not create an unnecessarily vast bracket.
            upper = 1.0
            while mismatch(upper) < 0:
                if upper >= max_search:
                    raise ValueError("target requires a family parameter beyond float64 range")
                upper = min(2.0 * upper, max_search)
            search_parameter = brentq(mismatch, 0.0, upper, xtol=5e-15, rtol=1e-13)

    eigvals = spectrum(search_parameter)
    parameter_used = math.exp(search_parameter) if family == "two_band" else search_parameter
    covariance = (q * eigvals[None, :]) @ q.T
    covariance = 0.5 * (covariance + covariance.T)
    return {
        "eigvals": eigvals,
        "Q": q,
        "Sigma": covariance,
        "parameter_used": float(parameter_used),
        "family": family,
    }


def make_initial_residual(
    d: int,
    seed: int,
    Q: np.ndarray,
    eigvals: np.ndarray,
    sigma: float,
    kind: str = "gaussian_w0",
    correlation: float = 0.0,
) -> dict:
    """Return a shared initialization in physical and covariance coordinates.

    ``gaussian_w0`` draws W0_ij independently as N(0, 0.25/d).  Its random
    stream does not depend on the covariance, so the same seed and dimension
    give the same physical W0 for every covariance and noise level.

    ``controlled_correlation`` sets R0_eigen = U sqrt(H), where U is an
    independently drawn Gaussian-QR orthogonal matrix and
    H = (1-rho) I + rho 11^T.  Each residual column has unit norm and every
    pair has inner product rho.  rho=0 gives orthogonal columns, rho=1 gives
    rank one, and intermediate rho varies the correlation continuously.
    W0 = W_star + R0_eigen Q^T, so this controlled initialization necessarily
    depends on the covariance.  U is shared across correlation choices.

    Returns ``R0_eigen`` = (W0-W_star) Q and the physical matrix ``W0``.
    """
    d, seed = _validate_dimension_seed(d, seed)
    noise_variance = _noise_variance(sigma)
    q = np.asarray(Q, dtype=np.float64)
    values = np.asarray(eigvals, dtype=np.float64)
    if q.shape != (d, d) or not np.all(np.isfinite(q)):
        raise ValueError("Q must be a finite d-by-d matrix")
    if not np.allclose(q.T @ q, np.eye(d), atol=1e-10, rtol=1e-10):
        raise ValueError("Q must be orthogonal")
    if values.shape != (d,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("eigvals must be a finite nonnegative vector of length d")
    rho = float(correlation)
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("correlation must lie in [0, 1]")
    scale = np.maximum(values, noise_variance)
    scaled_values = values / scale
    shrinkage = scaled_values / (scaled_values + noise_variance / scale)
    w_star = (q * shrinkage[None, :]) @ q.T

    if kind == "gaussian_w0":
        w0 = 0.5 * _rng(seed, _W0_DOMAIN).standard_normal((d, d)) / math.sqrt(d)
        r0_eigen = (w0 - w_star) @ q
    elif kind == "controlled_correlation":
        u = _orthogonal(d, seed, _RESIDUAL_DOMAIN)
        perpendicular_scale = math.sqrt(1.0 - rho)
        common_scale = math.sqrt(1.0 + (d - 1) * rho)
        sqrt_gram = perpendicular_scale * np.eye(d)
        sqrt_gram += (common_scale - perpendicular_scale) / d
        r0_eigen = u @ sqrt_gram
        w0 = w_star + r0_eigen @ q.T
    else:
        raise ValueError("kind must be 'gaussian_w0' or 'controlled_correlation'")
    return {"R0_eigen": r0_eigen, "W0": w0}
