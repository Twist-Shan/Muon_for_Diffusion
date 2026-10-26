"""Analyze the full continuous linear study without modifying its inputs.

Example: python -s -B analyze_full.py --run-dir results/full_linear_v1
Use --output-dir for a read-only analysis of a historical run.  Every seed
summary fixes all configuration fields; ranges are descriptive, never CIs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from flow import at_matched_loss, gd_at_matched_loss, spectral_groups, envelope

ROOT = Path(__file__).resolve().parent
FAMILIES = ("lognormal", "powerlaw", "geometric", "two_band")
NAMES = dict(zip(FAMILIES, ("Log-normal", "Power law", "Geometric", "Two bands")))
BASE = {"lognormal": 1., "powerlaw": 1., "geometric": float(np.log(100)), "two_band": 20.}
EPS = np.array([.5, .1, .01, .001, .0001])
DELTAS = np.array([.5, .1, .01, .001])
RESTARTS = (1., .5, .1, .01)
TOLERANCES = np.logspace(-1, -10, 10)
CONFIG = ("family", "d", "parameter", "sigma", "kind", "correlation", "init_scale", "band_size", "target_kappa_c", "matched_condition")
COLORS = ("#24789A", "#C06B35", "#6D528E", "#328768")
SIGN_MARGIN_FACTOR = 2.0


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf8")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v)) if isinstance(v, (list, dict, tuple)) else clean(v)
                             for k, v in row.items()})


def normalized_record(record):
    r = dict(record)
    r.setdefault("init_scale", .5)
    r.setdefault("band_size", r["d"] // 2 if r["family"] == "two_band" else 0)
    r.setdefault("target_kappa_c", 10. if r.get("matched_condition") else None)
    if "study_labels" not in r:
        r["study_labels"] = ["historical_geometry" if r["kind"] == "controlled_correlation" else
                             "historical_matched" if r.get("matched_condition") else "historical_main"]
    return r


def meta(record):
    return {k: record[k] for k in ("case_id", "seed") + CONFIG}


def read_run(directory):
    records = [normalized_record(json.loads(p.read_text(encoding="utf8")))
               for p in sorted((directory / "records").glob("*.json"))]
    if not records:
        records = [normalized_record(r) for r in json.loads((directory / "summary.json").read_text(encoding="utf8"))]
    if not records:
        raise ValueError("No completed case records")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    for r in records:
        if not (directory / "trajectories" / f"{r['case_id']}.npz").is_file():
            raise FileNotFoundError(f"Missing trajectory for {r['case_id']}")
    return records, manifest


def load_trajectory(directory, record):
    with np.load(directory / "trajectories" / f"{record['case_id']}.npz", allow_pickle=False) as archive:
        tr = {k: archive[k] for k in archive.files}
    if tr["q"].shape != (tr["t"].size, tr["c"].size) or np.any(tr["c"] <= 0):
        raise ValueError(f"Invalid trajectory: {record['case_id']}")
    return tr


def high_mask(record, tr):
    expected = tr["c"] == tr["c"].max() if record["family"] == "two_band" else tr["c"] >= np.median(tr["c"])
    if "high_band_mask" in tr and not np.array_equal(tr["high_band_mask"], expected):
        raise ValueError(f"Unexpected high-band mask: {record['case_id']}")
    return expected


def crossing(t, curve, target):
    """Interpolated first crossing and the two saved-time bracketing samples."""
    ix = np.flatnonzero(curve <= target)
    if not ix.size:
        return np.nan, np.nan, np.nan
    j = int(ix[0])
    if j == 0:
        return float(t[0]), float(t[0]), float(t[0])
    value = np.interp(target, curve[j - 1:j + 1][::-1], t[j - 1:j + 1][::-1])
    return float(value), float(t[j - 1]), float(t[j])


def envelope_clock(tr, Ainf):
    """Compute A(s), grouping repeated eigenvalues by block operator norm."""
    c, R0 = tr["c"], tr["R0"]
    groups = spectral_groups(c)
    cb = np.array([c[ix[0]] for ix in groups])
    b = np.array([np.linalg.norm(R0[:, ix], 2) for ix in groups])
    s = np.r_[0., np.geomspace(1e-9 / c.max(), 42 / c.min(), 18000)]
    G = np.linalg.norm(cb[None, :] * b[None, :] * np.exp(-s[:, None] * cb[None, :]), axis=1)
    A = np.r_[0., cumulative_trapezoid(G, s)]
    discrepancy = abs(A[-1] - Ainf) / Ainf
    A *= Ainf / A[-1]
    return lambda z: np.interp(z, s, A), discrepancy


def directional_metrics(record, tr):
    c, q, t = tr["c"], tr["q"], tr["t"]
    q0 = np.sum(tr["R0"] ** 2, axis=0)
    norm = float(np.linalg.norm(tr["R0"], 2))
    A, quad_error = envelope_clock(tr, record["diagnostics"]["theoretical_completion_upper"])
    directional, blocks = [], []
    for j in range(c.size):
        if q0[j] <= 1e-24 * max(q0.max(), 1.):
            continue
        for delta in DELTAS:
            hit, before, after = crossing(t, q[:, j] / q0[j], delta)
            lo = np.sqrt(q0[j]) * (1 - np.sqrt(delta))
            hi = float(A(np.log(1 / delta) / (2 * c[j])))
            directional.append(dict(case_id=record["case_id"], column=j, c=c[j], delta=delta,
                observed_time=hit, saved_bracket_lower=before, saved_bracket_upper=after,
                theoretical_lower=lo, theoretical_upper=hi,
                position_in_theoretical_interval=(hit - lo) / (hi - lo) if hi > lo + 1e-12 * norm else None,
                lower_violation_over_R0op=max(lo - after, 0) / norm,
                upper_violation_over_R0op=max(before - hi, 0) / norm))
    for ix in spectral_groups(c):
        initial = float(q0[ix].sum())
        if initial <= 1e-24:
            continue
        curve = q[:, ix].sum(axis=1) / initial
        for delta in DELTAS:
            hit, before, after = crossing(t, curve, delta)
            a = np.sqrt(q0[ix])
            lo = brentq(lambda tt: float(np.maximum(a - tt, 0) @ np.maximum(a - tt, 0) - delta * initial), 0, float(a.max()))
            hi = float(A(np.log(1 / delta) / (2 * c[ix[0]])))
            blocks.append(dict(case_id=record["case_id"], c=c[ix[0]], block_dimension=len(ix), delta=delta,
                observed_time=hit, saved_bracket_lower=before, saved_bracket_upper=after,
                theoretical_lower=lo, theoretical_upper=hi,
                upper_over_observed=hi / hit if hit > 0 else None,
                lower_over_observed=lo / hit if hit > 0 else None,
                position_in_theoretical_interval=(hit - lo) / (hi - lo) if hi > lo + 1e-12 * norm else None))
    # Curves are aggregated inside tied eigenspaces; a block lower bound is
    # the sum of the individual lower bounds, not (||R0 Pi||F - t)^2.
    upper_ratios, lower_ratios, widths = [], [], []
    for ix in spectral_groups(c):
        initial = q0[ix].sum()
        actual = q[:, ix].sum(axis=1) / initial
        upper = tr["q_upper"][:, ix].sum(axis=1) / initial
        lower = tr["q_lower"][:, ix].sum(axis=1) / initial
        active = (actual <= .99) & (actual >= 1e-4)
        upper_ratios.extend((upper[active] / actual[active]).tolist())
        lower_ratios.extend((lower[active] / actual[active]).tolist())
        widths.extend((upper[active] - lower[active]).tolist())
    summary = dict(case_id=record["case_id"], clock_grid_relative_discrepancy=quad_error,
        upper_over_actual_median=float(np.median(upper_ratios)) if upper_ratios else None,
        upper_over_actual_p90=float(np.quantile(upper_ratios, .9)) if upper_ratios else None,
        lower_over_actual_median=float(np.median(lower_ratios)) if lower_ratios else None,
        envelope_width_initial_normalized_median=float(np.median(widths)) if widths else None,
        largest_hitting_lower_violation_over_R0op=max((r["lower_violation_over_R0op"] for r in directional), default=0.),
        largest_hitting_upper_violation_over_R0op=max((r["upper_violation_over_R0op"] for r in directional), default=0.))
    return directional, blocks, summary


def allocation_metrics(record, tr):
    mask, c = high_mask(record, tr), tr["c"]
    qm = at_matched_loss(tr, EPS)
    qg, _ = gd_at_matched_loss(tr["R0"], c, EPS)
    mu, gd = (qm[:, mask] @ c[mask]) / (qm @ c), (qg[:, mask] @ c[mask]) / (qg @ c)
    ref = record["refinement"] or {}
    raw = ref.get("matched_high_fraction_differences_by_epsilon")
    source_eps = record["matched"]["epsilons"]
    uncertainty = []
    for eps in EPS:
        matches = np.flatnonzero(np.isclose(source_eps, eps, rtol=1e-12, atol=0))
        if raw is not None and matches.size:
            uncertainty.append(raw[int(matches[0])])
        elif matches.size:
            uncertainty.append(ref.get("max_matched_high_fraction_difference", np.nan))
        else:
            uncertainty.append(np.nan)
    rows = []
    for eps, m, g, u in zip(EPS, mu, gd, uncertainty):
        difference = float(m - g)
        passes = record["refinement_passed"] and np.isfinite(u)
        sign1 = ("positive" if difference > 0 else "negative") if passes and abs(difference) > max(float(u), 1e-12) else "unresolved"
        resolved = passes and abs(difference) > max(SIGN_MARGIN_FACTOR * float(u), 1e-12)
        # A rank-one residual remains rank one. Its polar trajectory has
        # r_j=r_j(0) exp(-c_j*zeta), so GD has exactly the same allocation
        # at matched loss. Report this analytic control separately instead
        # of assigning physical meaning to a discretization remainder.
        rank_one_control = record["kind"] == "controlled_correlation" and record["correlation"] == 1.
        sign = "analytic_tie" if rank_one_control else ("positive" if difference > 0 else "negative") if resolved else "unresolved"
        rows.append(dict(**meta(record), study_labels=record["study_labels"], epsilon=eps,
            high_band_dimension=int(mask.sum()), muon_fraction=m, gd_fraction=g, difference=difference,
            adjacent_grid_uncertainty=u, empirical_sign_margin=SIGN_MARGIN_FACTOR * u,
            sign=sign, raw_one_discrepancy_sign=sign1, analytic_rank_one_tie=rank_one_control,
            refinement_passed=record["refinement_passed"]))
    return rows


def certificate_metrics(record, tr):
    """Restart at an actual saved sample, never at interpolated Gamma/chi."""
    c, q, F, t, g = (tr[k] for k in ("c", "q", "F", "t", "g"))
    rel = F / F[0]
    allrows, summaries = [], []
    for request in RESTARTS:
        start = int(np.flatnonzero(rel <= request + 1e-13)[0])
        active = np.flatnonzero((np.arange(t.size) >= start) & (rel >= 1e-4) & (F > 0))
        gamma, chi = float(tr["gamma"][start]), float(tr["chi"][start])
        if not active.size or not np.isfinite(chi):
            continue
        progress, x = F[active] / F[start], g[active] / g[start]
        rows = []
        for alpha in np.unique(c):
            mask = c >= alpha
            initial_band = float(q[start, mask] @ c[mask])
            if initial_band <= 1e-25 * F[start]:
                continue
            actual = (q[active][:, mask] @ c[mask]) / F[active]
            base_fraction = initial_band / F[start]
            gamma_ok, chi_ok = alpha > gamma + 1e-10 * c.max(), alpha > chi + 1e-10 * c.max()
            row = dict(case_id=record["case_id"], restart_requested=request,
                restart_actual=float(rel[start]), restart_time=float(t[start]), alpha=float(alpha),
                dimension=int(mask.sum()), gamma=gamma, chi=chi,
                restart_band_fraction=base_fraction, gamma_eligible=gamma_ok, chi_eligible=chi_ok)
            if gamma_ok:
                bound = base_fraction * progress ** (alpha / gamma - 1)
                row.update(gamma_max_violation=float(max(0, np.max(actual - bound))),
                    gamma_max_monotonicity_increase=float(max(0, np.max(np.diff(actual), initial=0))),
                    gamma_terminal_bound=float(bound[-1]), gamma_terminal_actual=float(actual[-1]),
                    gamma_terminal_slack=float(bound[-1] - actual[-1]),
                    gamma_useful=bool(np.any(bound < min(1., base_fraction) - 1e-10)))
            if chi_ok:
                coefficient = chi * initial_band / g[start] ** 2
                bound_g = np.minimum(1., coefficient * x ** (2 * (alpha / chi - 1)))
                bound_f = np.minimum(1., coefficient * (chi * F[active] / g[start] ** 2) ** (alpha / chi - 1))
                row.update(chi_gradient_max_violation=float(max(0, np.max(actual - bound_g))),
                    chi_loss_max_violation=float(max(0, np.max(actual - bound_f))),
                    chi_gradient_useful=bool(np.any(bound_g < 1 - 1e-10)),
                    chi_loss_useful=bool(np.any(bound_f < 1 - 1e-10)),
                    chi_gradient_improves_restart_fraction=bool(np.any(bound_g < base_fraction - 1e-10)),
                    chi_terminal_gradient_bound=float(bound_g[-1]), chi_terminal_loss_bound=float(bound_f[-1]),
                    chi_terminal_actual=float(actual[-1]), chi_terminal_gradient_slack=float(bound_g[-1] - actual[-1]))
            rows.append(row)
        for threshold in ("gamma", "chi"):
            eligible = [r for r in rows if r[f"{threshold}_eligible"]]
            useful_key = "gamma_useful" if threshold == "gamma" else "chi_gradient_useful"
            summaries.append(dict(**meta(record), study_labels=record["study_labels"],
                restart_requested=request, restart_actual=float(rel[start]), threshold=threshold,
                threshold_over_cmax=(gamma if threshold == "gamma" else chi) / float(c.max()),
                eligible_cutoffs=len(eligible), available_cutoffs=len(rows),
                widest_eligible_dimension=max((r["dimension"] for r in eligible), default=0),
                widest_eligible_restart_loss_fraction=max((r["restart_band_fraction"] for r in eligible), default=0),
                useful_cutoffs=sum(bool(r.get(useful_key)) for r in eligible),
                observed_minimum_relative_loss=float(rel[active[-1]])))
        allrows.extend(rows)
    return allrows, summaries


def completion_metrics(record, tr):
    norm = float(np.linalg.norm(tr["R0"], 2))
    rel = tr["F"] / tr["F"][0]
    positive = rel[rel > 0]
    rows = []
    for eps in TOLERANCES:
        hit, before, after = crossing(tr["t"], rel, eps)
        rows.append(dict(case_id=record["case_id"], epsilon=eps, observed_time=hit,
            saved_bracket_lower=before, saved_bracket_upper=after,
            time_over_initial_opnorm=hit / norm,
            extinction_minus_hit_over_initial_opnorm=(tr["t"][-1] - hit) / norm,
            interpolates_final_zero_segment=bool(eps < positive[-1]),
            theoretical_completion_lower=record["diagnostics"]["theoretical_completion_lower"],
            theoretical_completion_upper=record["diagnostics"]["theoretical_completion_upper"]))
    levels = []
    for level in record["refinement_levels"]:
        levels.append(dict(case_id=record["case_id"], resolution=level["resolution"], h=level["h"],
            extinction_time=level["numerical_extinction_time"],
            extinction_over_initial_opnorm=level["numerical_extinction_time"] / norm,
            **level.get("versus_previous", {})))
    return rows, levels


def ordering_metrics(record, tr):
    """Compare distinct eigenspace hitting times with empirical margins.

    Margin windows invert delta +/- the last-refinement modal discrepancy.
    They are sensitivity intervals, not rigorous continuous-flow bounds.
    """
    groups = spectral_groups(tr["c"])
    error = float(record["refinement"]["max_modal_initial_normalized"])
    rows = []
    for delta in DELTAS:
        windows, hits = [], []
        for ix in groups:
            curve = tr["q"][:, ix].sum(axis=1) / tr["q"][0, ix].sum()
            hit = crossing(tr["t"], curve, delta)[0]
            earlier = crossing(tr["t"], curve, min(1., delta + error))[1]
            later = crossing(tr["t"], curve, max(0., delta - error))[2]
            hits.append(hit)
            windows.append((earlier, later))
        windows = np.asarray(windows)
        pairmask = np.triu(np.ones((len(groups), len(groups)), dtype=bool), 1)
        earlier = (windows[:, 1, None] < windows[None, :, 0] - 1e-12) & pairmask
        later = (windows[:, 0, None] > windows[None, :, 1] + 1e-12) & pairmask
        pairs = int(pairmask.sum())
        rows.append(dict(**meta(record), delta=delta, distinct_eigenspaces=len(groups), compared_pairs=pairs,
            higher_c_earlier_pairs=int(earlier.sum()), higher_c_later_pairs=int(later.sum()),
            overlapping_empirical_windows=pairs - int(earlier.sum()) - int(later.sum()),
            higher_c_later_fraction=float(later.sum() / pairs) if pairs else None,
            top_block_time=hits[0], bottom_block_time=hits[-1],
            top_over_bottom_time=hits[0] / hits[-1] if hits[-1] > 0 else None,
            modal_discrepancy_used=error, refinement_passed=record["refinement_passed"]))
    return rows


def group_diagnostics(case_rows, envelope_rows):
    by_id = {r["case_id"]: r for r in envelope_rows}
    groups = defaultdict(list)
    for row in case_rows:
        groups[tuple(row[k] for k in CONFIG)].append(dict(row, **{k: v for k, v in by_id[row["case_id"]].items() if k not in row}))
    fields = ("completion_upper_over_numerical", "numerical_over_completion_lower",
              "upper_over_actual_median", "lower_over_actual_median", "envelope_width_initial_normalized_median",
              "max_modal_initial_normalized", "max_matched_high_fraction_difference")
    result = []
    for key, values in groups.items():
        row = dict(zip(CONFIG, key))
        row["n_seeds"] = len({r["seed"] for r in values})
        for field in fields:
            data = [v[field] for v in values if v[field] is not None]
            for statistic, fun in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                row[f"{field}_{statistic}"] = float(fun(data)) if data else None
        result.append(row)
    return result


def group_allocation(rows):
    groups = defaultdict(list)
    for row in rows:
        for study in row["study_labels"]:
            groups[(study, row["epsilon"]) + tuple(row[k] for k in CONFIG)].append(row)
    out = []
    for key, values in groups.items():
        row = dict(study=key[0], epsilon=key[1], **dict(zip(CONFIG, key[2:])),
                   n_seeds=len({v["seed"] for v in values}), seeds=sorted(v["seed"] for v in values))
        for field in ("muon_fraction", "gd_fraction", "difference", "adjacent_grid_uncertainty"):
            data = np.array([v[field] for v in values], dtype=float)
            finite = data[np.isfinite(data)]
            for statistic, fun in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                row[f"{field}_{statistic}"] = float(fun(finite)) if finite.size else None
        row.update({f"{sign}_seeds": sum(v["sign"] == sign for v in values) for sign in ("positive", "negative", "unresolved", "analytic_tie")})
        row["seed_interpretation"] = ("rotation reproducibility; fixed spectrum and residual Gram" if
            row["kind"] == "controlled_correlation" and row["family"] != "lognormal" else "independent seed realizations")
        out.append(row)
    return out


def is_baseline(r, sigma=1.):
    return (r["kind"] == "gaussian_w0" and r["d"] == 32 and not r["matched_condition"] and
            np.isclose(r["sigma"], sigma) and np.isclose(r["init_scale"], .5) and
            np.isclose(r["parameter"], BASE[r["family"]]) and
            (r["family"] != "two_band" or r["band_size"] == r["d"] // 2))


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})


def savefig(fig, directory, name, title, note):
    fig.suptitle(title, fontsize=14, y=.995)
    fig.text(.035, .012, note, fontsize=8, va="bottom", color="#444444")
    fig.tight_layout(rect=(.005, .075, .995, .96), h_pad=2.)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", facecolor="white")
    plt.close(fig)


def range_line(ax, x, samples, label, color, marker="o", linestyle="-", controls=False):
    data = np.asarray(samples, dtype=float)
    mean, lo, hi = np.nanmean(data, axis=0), np.nanmin(data, axis=0), np.nanmax(data, axis=0)
    ax.plot(x, mean, label=label, color=color, marker=marker, ms=3.5, lw=1.4, ls=linestyle)
    ax.fill_between(x, lo, hi, color=color, alpha=.13)
    if controls and len(x):
        for j in [k for k, value in enumerate(x) if value in (0., 1.)]:
            ax.plot(x[j], mean[j], "o", mfc="white", mec="black", ms=6, zorder=5)
    ax.grid(alpha=.18)


def sweep_plot(ax, rows, xkey, label, color, controls=False):
    xs = sorted({r[xkey] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    if not xs:
        ax.text(.5, .5, "Not available", ha="center", transform=ax.transAxes)
        return
    data = []
    for seed in seeds:
        values = []
        for x in xs:
            match = [r for r in rows if r["seed"] == seed and r[xkey] == x]
            if len(match) > 1:
                raise ValueError("Plot attempted to pool different fixed configurations")
            values.append(match[0]["difference"] if match else np.nan)
        data.append(values)
    range_line(ax, xs, data, label, color, controls=controls)
    ax.axhline(0, lw=.7, color="gray")


def legend_if_present(ax):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)


def make_figures(records, run_dir, out, allocation, certificate_summary, blockhits, completions, levels):
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    style()
    baseline = [r for r in records if is_baseline(r)]
    selected = {family: [r["case_id"] for r in baseline if r["family"] == family] for family in FAMILIES}
    # 1. Directional block envelopes at fixed baseline configurations.
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for col, family in enumerate(FAMILIES):
        cases = [r for r in baseline if r["family"] == family]
        if not cases:
            continue
        grid = np.linspace(0, 1.6, 321)
        trajectories = [load_trajectory(run_dir, r) for r in cases]
        extended_envelopes = [envelope(tr["R0"], tr["c"], grid * np.linalg.norm(tr["R0"], 2)) for tr in trajectories]
        for percentile, color in ((0, COLORS[0]), (1, COLORS[1])):
            curves, lowers, uppers = [], [], []
            for tr, env in zip(trajectories, extended_envelopes):
                groups = spectral_groups(tr["c"])
                ix = groups[0 if percentile == 0 else -1]
                q0 = tr["q"][0, ix].sum()
                tx = tr["t"] / np.linalg.norm(tr["R0"], 2)
                curves.append(np.interp(grid, tx, tr["q"][:, ix].sum(axis=1) / q0))
                lowers.append(env["q_lower"][:, ix].sum(axis=1) / q0)
                uppers.append(env["q_upper"][:, ix].sum(axis=1) / q0)
            range_line(axes[0, col], grid, curves, "Top eigenspace" if percentile == 0 else "Bottom eigenspace", color, marker=None)
            for data in (lowers, uppers):
                axes[0, col].plot(grid, np.mean(data, axis=0), ":", color=color, lw=1.)
        axes[0, col].set(title=NAMES[family], xlabel=r"$t/\|R_0\|_{op}$", ylabel="Block error / initial block error", yscale="log", ylim=(1e-4, 1.1))
        legend_if_present(axes[0, col])
        # Average inside one case first; cross-seed range is then displayed.
        casesids = {r["case_id"] for r in cases}
        for field, label, color in (("upper_over_observed", "Upper / observed", COLORS[0]), ("lower_over_observed", "Lower / observed", COLORS[1])):
            samples = [[np.mean([h[field] for h in blockhits if h["case_id"] == r["case_id"] and h["delta"] == delta]) for delta in DELTAS] for r in cases]
            range_line(axes[1, col], DELTAS, samples, label, color)
        axes[1, col].set(xscale="log", xlim=(.5, .001), xlabel="Relative block accuracy", ylabel="Hitting bound / observed time")
        axes[1, col].axhline(1, color="gray", lw=.7)
        legend_if_present(axes[1, col])
    savefig(fig, figures, "01_envelopes_and_hitting", "Directional envelopes and hitting-time tightness",
            "Baseline: d=32, sigma=1, scale=0.5. Solid: seed mean; shading: seed min-max; dotted: mean bounds.\n"
            "Tied eigenspaces are summed. Bottom: average over eigenspaces within each case, then mean/range across seeds; no confidence intervals.")
    # 2. All prescribed width/noise settings at the fixed primary checkpoint.
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for row, sigma in enumerate((.1, 1., 3.)):
        for col, family in enumerate(FAMILIES):
            rs = [r for r in allocation if r["epsilon"] == .01 and r["family"] == family and
                  r["sigma"] == sigma and "spectrum_width" in r["study_labels"]]
            sweep_plot(axes[row, col], rs, "parameter", "Muon minus GD", COLORS[row])
            axes[row, col].set(title=f"{NAMES[family]}, sigma={sigma:g}", xlabel="Spectral shape parameter", ylabel="High-band fraction difference")
    savefig(fig, figures, "02_width_noise", "Matched-loss allocation across spectral width and noise",
            "F/F0=0.01. Each curve fixes family, sigma, dimension, initialization scale and geometry; means/ranges vary only the seed.\n"
            "Positive means a larger remaining high-spectrum share. Means include numerically unresolved points; per-case sign labels are in the CSV.")
    # 3. Initialization magnitude and geometry; endpoints stay visibly distinct.
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for col, family in enumerate(FAMILIES):
        for sigma, color in zip((.1, 1., 3.), COLORS):
            rs = [r for r in allocation if r["epsilon"] == .01 and r["family"] == family and r["sigma"] == sigma and "initialization_scale" in r["study_labels"]]
            sweep_plot(axes[0, col], rs, "init_scale", f"sigma={sigma:g}", color)
        axes[0, col].set(title=NAMES[family], xlabel="Gaussian W0 scale", ylabel="High-band fraction difference")
        legend_if_present(axes[0, col])
        for eps, color in zip((.1, .01, .001), COLORS):
            rs = [r for r in allocation if r["epsilon"] == eps and r["family"] == family and ("initialization_geometry" in r["study_labels"] or "historical_geometry" in r["study_labels"])]
            sweep_plot(axes[1, col], rs, "correlation", f"F/F0={eps:g}", color, controls=True)
        axes[1, col].set(xlabel="Residual-column correlation rho", ylabel="High-band fraction difference")
        legend_if_present(axes[1, col])
    savefig(fig, figures, "03_initialization", "Initialization magnitude and fixed-column-norm geometry",
            "Mean and min-max across seeds; white markers at rho=0 and rho=1 mark analytic/rank-boundary controls.\n"
            "Fixed-Gram/deterministic-spectrum seed rotations repeat the same dynamics. Plotted means include unresolved points; see CSV sign labels.")
    # 4. Restart thresholds and coverage, evaluated at fixed baselines.
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for col, family in enumerate(FAMILIES):
        ids = selected[family]
        for threshold, color in (("gamma", COLORS[0]), ("chi", COLORS[1])):
            for row, field in ((0, "threshold_over_cmax"), (1, "widest_eligible_restart_loss_fraction")):
                samples = []
                for identity in ids:
                    samples.append([next((r[field] for r in certificate_summary if r["case_id"] == identity and r["threshold"] == threshold and r["restart_requested"] == eps), np.nan) for eps in RESTARTS])
                if samples:
                    range_line(axes[row, col], RESTARTS, samples, threshold.capitalize(), color)
                axes[row, col].set(xscale="log", xlim=(1., .01), xlabel="Requested restart F/F0")
        axes[0, col].set(title=NAMES[family], ylabel="Threshold / largest eigenvalue", yscale="log")
        axes[0, col].axhline(1., color="gray", lw=.7)
        axes[1, col].set(ylabel="Restart loss in widest eligible band", ylim=(-.03, 1.03))
        legend_if_present(axes[0, col])
    savefig(fig, figures, "04_restart_certificates", "Restarted spectral certificates: thresholds and coverage",
            "Fixed baseline configurations, mean/min-max across seeds. Restart uses the first saved point at or below the requested loss.\n"
            "Gamma controls actual fraction monotonicity; chi controls an upper envelope. Eligibility and nontriviality are separately tabulated in CSV.")
    # 5. Tolerance convergence and actual adjacent-grid completion estimates.
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for col, family in enumerate(FAMILIES):
        cases = [r for r in baseline if r["family"] == family]
        samples = [[next(r["time_over_initial_opnorm"] for r in completions if r["case_id"] == case["case_id"] and r["epsilon"] == eps) for eps in TOLERANCES] for case in cases]
        if samples:
            range_line(axes[0, col], TOLERANCES, samples, "Tolerance crossing", COLORS[0])
            tail = [any(r["interpolates_final_zero_segment"] for r in completions if r["case_id"] in {case["case_id"] for case in cases} and r["epsilon"] == eps) for eps in TOLERANCES]
            axes[0, col].plot(TOLERANCES[tail], np.mean(samples, axis=0)[tail], "o", mfc="white", mec=COLORS[0], ms=4, zorder=5)
            upper = [case["diagnostics"]["theoretical_completion_upper"] / case["diagnostics"]["theoretical_completion_lower"] for case in cases]
            axes[0, col].text(.04, .05, f"Completion lower: 1\nUpper range: {min(upper):.3g}-{max(upper):.3g}", transform=axes[0, col].transAxes, fontsize=7)
        for case, color in zip(cases, COLORS):
            rows = [r for r in levels if r["case_id"] == case["case_id"]]
            axes[1, col].plot([r["resolution"] for r in rows], [r["extinction_over_initial_opnorm"] for r in rows], "o-", color=color, ms=3, label=f"seed {case['seed']}")
        axes[0, col].set(title=NAMES[family], xscale="log", xlim=(1e-1, 1e-10), xlabel="Relative total loss tolerance", ylabel=r"Crossing time / $\|R_0\|_{op}$")
        axes[1, col].set(xscale="log", xlabel="Resolution (inverse step size)", ylabel=r"Numerical extinction / $\|R_0\|_{op}$")
        legend_if_present(axes[1, col])
        for ax in axes[:, col]:
            ax.grid(alpha=.18)
    savefig(fig, figures, "05_completion_refinement", "Completion estimates: decreasing loss tolerances and step refinement",
            "Top: seed mean and min-max. Hollow points: at least one seed interpolates the last saved segment to zero; CSV gives per-case flags.\n"
            "Bottom: every executed resolution for the fixed baselines. Numerical extinction is an approximation, not evidence of exact finite-time arrival.")
    # 6. No combining conditioning, dimension, and band size into one mean.
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(3, 4)
    for col, family in enumerate(FAMILIES):
        ax = fig.add_subplot(grid[0, col])
        rs = [r for r in allocation if r["epsilon"] == .01 and r["family"] == family and
              ("dimension" in r["study_labels"] or is_baseline(r))]
        sweep_plot(ax, rs, "d", "F/F0=0.01", COLORS[0])
        ax.set(title=NAMES[family], xlabel="Dimension", ylabel="High-band fraction difference")
        ax = fig.add_subplot(grid[1, col])
        rs = [r for r in allocation if r["epsilon"] == .01 and r["family"] == family and "matched_conditioning" in r["study_labels"]]
        sweep_plot(ax, rs, "target_kappa_c", "sigma=0.3", COLORS[1])
        ax.set(xlabel="Matched condition number of C", ylabel="High-band fraction difference")
    ax = fig.add_subplot(grid[2, :2])
    for eps, color in zip((.1, .01, .001), COLORS):
        rs = [r for r in allocation if r["epsilon"] == eps and "two_band_size" in r["study_labels"]]
        sweep_plot(ax, rs, "band_size", f"F/F0={eps:g}", color)
    ax.set(xlabel="Dimension of largest-eigenvalue eigenspace", ylabel="High-band fraction difference", title="Two bands: prescribed multiplicity sweep")
    legend_if_present(ax)
    ax = fig.add_subplot(grid[2, 2:])
    ax.axis("off")
    ax.text(.03, .85, "Each panel fixes the other experimental factors.\n\nBands: full top eigenspace for two-band spectra;\nmedian cutoff retaining ties for other families.\n\nLines and shaded regions: seed mean and min-max.\nThree seeds provide descriptive variability, not a CI.", va="top", fontsize=11)
    savefig(fig, figures, "06_dimension_conditioning_multiplicity", "Dimension, matched conditioning, and eigenspace multiplicity",
            "F/F0=0.01 in upper two rows. d=32 baseline is shown with the dimension sweep; the remaining dimension settings are independently paired runs.\n"
            "Plotted means include unresolved points. All settings and five checkpoints, with numerical-resolution labels, are in grouped_allocation.csv.")
    write_json(figures / "figure_manifest.json", dict(baseline_cases=selected,
        description="Six PNG/PDF pairs; cross-seed means and min-max ranges at fixed configurations; no confidence intervals",
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = args.output_dir.resolve() if args.output_dir else run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    records, manifest = read_run(run_dir)
    arrays = {key: [] for key in ("allocation", "directional", "blocks", "envelopes", "certificates", "restart", "completion", "levels", "cases", "ordering")}
    with threadpool_limits(limits=1):
        for index, record in enumerate(records):
            tr = load_trajectory(run_dir, record)
            arrays["allocation"].extend(allocation_metrics(record, tr))
            direct, blocks, envelope = directional_metrics(record, tr)
            arrays["directional"].extend(direct)
            arrays["blocks"].extend(blocks)
            arrays["envelopes"].append(dict(**meta(record), **{k: v for k, v in envelope.items() if k != "case_id"}))
            cert, restart = certificate_metrics(record, tr)
            arrays["certificates"].extend(cert)
            arrays["restart"].extend(restart)
            completion, levels = completion_metrics(record, tr)
            arrays["ordering"].extend(ordering_metrics(record, tr))
            arrays["completion"].extend(completion)
            arrays["levels"].extend(levels)
            diag = {k: v for k, v in record["diagnostics"].items() if not isinstance(v, (dict, list))}
            arrays["cases"].append(dict(**meta(record), study_labels=record["study_labels"],
                **diag, **record["refinement"], refinement_passed=record["refinement_passed"],
                completion_upper_over_numerical=diag["theoretical_completion_upper"] / tr["t"][-1],
                numerical_over_completion_lower=tr["t"][-1] / diag["theoretical_completion_lower"],
                initial_column_norm_squared_min=float(tr["q"][0].min()),
                initial_column_norm_squared_max=float(tr["q"][0].max()),
                high_band_dimension=int(high_mask(record, tr).sum())))
            if (index + 1) % 25 == 0:
                print(f"ANALYZED {index + 1}/{len(records)}", flush=True)
    grouped = group_allocation(arrays["allocation"])
    names = dict(allocation="matched_allocation", directional="directional_hitting_times", blocks="block_hitting_times",
                 envelopes="envelope_tightness", certificates="restart_certificates", restart="restart_case_metrics",
                 completion="completion_tolerances", levels="refinement_levels", cases="case_metrics", ordering="eigenspace_learning_order")
    for key, name in names.items():
        write_csv(out / f"{name}.csv", arrays[key])
    write_csv(out / "grouped_allocation.csv", grouped)
    write_csv(out / "grouped_diagnostics.csv", group_diagnostics(arrays["cases"], arrays["envelopes"]))
    write_json(out / "case_metrics.json", arrays["cases"])
    total_expected = len(manifest.get("cases", records))
    configuration_key = lambda r: tuple(r[k] for k in ("seed",) + CONFIG)
    expected_set = {configuration_key(normalized_record(r)) for r in manifest.get("cases", records)}
    observed_set = {configuration_key(r) for r in records}
    study_counts = Counter(label for r in records for label in r["study_labels"])
    sign_counts = []
    for study in sorted(study_counts):
        for eps in EPS:
            rs = [r for r in arrays["allocation"] if study in r["study_labels"] and r["epsilon"] == eps]
            sign_counts.append(dict(study=study, epsilon=eps, cases=len(rs), **Counter(r["sign"] for r in rs)))
    cert_counts = []
    for study in sorted(study_counts):
        for eps in RESTARTS:
            for threshold in ("gamma", "chi"):
                rs = [r for r in arrays["restart"] if study in r["study_labels"] and r["restart_requested"] == eps and r["threshold"] == threshold]
                cert_counts.append(dict(study=study, restart=eps, threshold=threshold, cases=len(rs),
                    any_eligible=sum(r["eligible_cutoffs"] > 0 for r in rs), any_useful=sum(r["useful_cutoffs"] > 0 for r in rs)))
    geometry_groups = defaultdict(list)
    for row in arrays["allocation"]:
        if row["kind"] == "controlled_correlation":
            geometry_groups[(row["family"], row["seed"], row["d"], row["parameter"], row["sigma"], row["epsilon"])].append(row)
    geometry_effects = []
    for key, values in geometry_groups.items():
        geometry_effects.append(dict(zip(("family", "seed", "d", "parameter", "sigma", "epsilon"), key)) |
            dict(correlations=sorted(r["correlation"] for r in values),
                 gd_high_fraction_range=float(np.ptp([r["gd_fraction"] for r in values])),
                 muon_high_fraction_range=float(np.ptp([r["muon_fraction"] for r in values]))))
    write_csv(out / "geometry_effects.csv", geometry_effects)
    extrema = {}
    for key in ("gamma_max_violation", "gamma_max_monotonicity_increase", "chi_gradient_max_violation", "chi_loss_max_violation"):
        extrema[key] = max((r.get(key, 0) for r in arrays["certificates"]), default=0)
    aggregate = dict(completed_cases=len(records), expected_cases=total_expected,
        complete=len(records) == total_expected and observed_set == expected_set and len(observed_set) == len(records) and all(r["refinement_passed"] for r in records),
        missing_configurations=len(expected_set - observed_set), unexpected_configurations=len(observed_set - expected_set),
        refinement_passed=sum(r["refinement_passed"] for r in records), study_memberships=dict(study_counts),
        seeds=sorted({r["seed"] for r in records}), sign_counts=sign_counts, restart_coverage=cert_counts,
        certificate_maxima=extrema,
        maximum_geometry_GD_fraction_range=max((r["gd_high_fraction_range"] for r in geometry_effects), default=None),
        analytic_rank_one_maximum_allocation_deviation=max((abs(r["difference"]) for r in arrays["allocation"] if r["analytic_rank_one_tie"]), default=None),
        empirical_sign_margin_factor=SIGN_MARGIN_FACTOR,
        max_directional_hitting_lower_violation=max(r["largest_hitting_lower_violation_over_R0op"] for r in arrays["envelopes"]),
        max_directional_hitting_upper_violation=max(r["largest_hitting_upper_violation_over_R0op"] for r in arrays["envelopes"]),
        completion_upper_over_numerical_range=[min(r["completion_upper_over_numerical"] for r in arrays["cases"]), max(r["completion_upper_over_numerical"] for r in arrays["cases"])],
        numerical_over_completion_lower_range=[min(r["numerical_over_completion_lower"] for r in arrays["cases"]), max(r["numerical_over_completion_lower"] for r in arrays["cases"])],
        numerical_note="Adjacent-grid differences are empirical uncertainty indicators, not rigorous error bounds; no CIs from three seeds.",
        theory="output/diffusion_wp_general_restart/pipeline_diffusion.tex; modal-envelope and spectral-bias theorems",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out / "aggregate.json", aggregate)
    write_report(out, records, arrays, grouped, aggregate)
    shutil.copyfile(Path(__file__), out / "analyze_full_executed.py")
    if not args.skip_figures:
        with threadpool_limits(limits=1):
            make_figures(records, run_dir, out, arrays["allocation"], arrays["restart"], arrays["blocks"], arrays["completion"], arrays["levels"])
    print(json.dumps(dict(completed_cases=len(records), expected_cases=total_expected, complete=aggregate["complete"], output=str(out)), indent=2), flush=True)


def write_report(out, records, arrays, grouped, aggregate):
    n = len(records)
    passed = aggregate["refinement_passed"]
    lines = ["# Expanded linear denoising experiment", "",
        f"Analyzed {n}/{aggregate['expected_cases']} prescribed unique configurations; {passed}/{n} passed the numerical refinement gate. " +
        ("The recorded grid is complete." if aggregate["complete"] else "This is an incomplete or numerically unresolved run; it must not be reported as complete."), "",
        "These experiments compare exact population GD with a numerical approximation of continuous compact-polar flow. "
        "The estimand is directional denoising error and spectral allocation at matched relative excess loss. "
        "The data do not establish a universal GD-versus-Muon ordering, a discrete optimizer speed advantage, or a nonlinear result.", "",
        "## Coverage and numerical interpretation", "",
        "| Study membership | Recorded configurations |", "|---|---:|"]
    for study, count in aggregate["study_memberships"].items():
        lines.append(f"| {study} | {count} |")
    lines.extend(["", "Memberships overlap; each unique configuration is executed once. Seeds are fixed independently of calendar dates. "
        "All summaries in grouped_allocation.csv hold every configuration field fixed and vary only seed. The figures show seed means and min-max ranges, not confidence intervals. "
        "With deterministic spectra and a fixed residual Gram matrix, controlled-correlation seed rotations reproduce the same dynamics; these repeats are not independent statistical replicates.", "",
        "At rho=0 and rho=1, the orthogonal-column and rank-one cases are analytic/rank-boundary controls. "
        "The intermediate correlations provide the general geometry experiment. geometry_effects.csv quantifies the Muon allocation variation and GD invariance across rho at each fixed spectrum, seed and checkpoint. For two-band spectra, the high band is the entire largest-eigenvalue eigenspace, including unequal multiplicities. "
        "Other spectra use the median cutoff including all ties.", "",
        "The saved refinement columns report observed differences between the final two step sizes. They are empirical numerical uncertainty indicators, not certified errors of the continuous solution. "
        "A sign is called resolved only when its magnitude exceeds twice its own checkpoint-specific adjacent-grid difference (and 1e-12), and the case passes the refinement gate. This terminology does not certify the continuous-flow sign. "
        "The uninflated one-discrepancy classification is retained as raw_one_discrepancy_sign for sensitivity analysis. "
        "Rank-one rho=1 controls are labeled analytic_tie: both flows share the same spectral trajectory after a time reparameterization and therefore have identical allocation at matched loss. Their nonzero measured differences quantify numerical error. "
        "These controls show why a single adjacent-grid difference can slightly underestimate the remaining first-order error, motivating the factor-two empirical margin. "
        "Historical runs lacking that checkpoint's refinement check are marked unresolved there.", "",
        "## Directional bounds and relative-accuracy times", "",
        "The upper hitting bound is A(log(1/delta)/(2c)); the individual lower bound is a_j(1-sqrt(delta)). "
        "A uses operator norms of full eigenspace blocks. For plotted block errors, lower envelopes are summed over columns and their crossing is solved numerically; a Frobenius block norm is not substituted into the single-direction lower formula. "
        "Individual directional rows are basis-specific inside tied eigenspaces. Block actual errors and upper envelopes are basis-invariant; the valid summed lower bound uses the fixed saved eigenbasis. block_hitting_times.csv reports these block-level quantities.", "",
        f"Largest hitting-bound discrepancies outside the saved crossing brackets, divided by ||R0||op: lower {aggregate['max_directional_hitting_lower_violation']:.3g}, upper {aggregate['max_directional_hitting_upper_violation']:.3g}. "
        "The crossing bracket only measures temporal sampling resolution; solver uncertainty is reported separately. "
        "eigenspace_learning_order.csv compares distinct eigenspace hitting times after expanding each crossing window using delta plus/minus the observed last-refinement modal discrepancy. "
        "Higher-c earlier, higher-c later and overlapping-window counts quantify learning order without treating unresolved separation as a strict order. These windows are empirical sensitivity diagnostics, not certified error bounds.", "",
        "Envelope tightness is evaluated where the actual relative block error is between 1e-4 and 0.99. "
        "envelope_tightness.csv reports upper/actual and lower/actual ratios and envelope widths, rather than only checking violations. "
        "block_hitting_times.csv reports the position inside each theoretical interval and both bound/observed-time ratios.", "",
        "| Initialization class | Cases | Median of case median upper/actual ratios | Median of case median lower/actual ratios |", "|---|---:|---:|---:|"])
    for kind in sorted({r["kind"] for r in records}):
        correlations = sorted({r["correlation"] for r in records if r["kind"] == kind}) if kind == "controlled_correlation" else [None]
        for rho in correlations:
            ids = {r["case_id"] for r in records if r["kind"] == kind and (rho is None or r["correlation"] == rho)}
            rs = [r for r in arrays["envelopes"] if r["case_id"] in ids]
            label = kind + (f", rho={rho:g}" if rho is not None else "")
            # These descriptive diagnostics span the class; allocation means
            # remain strictly stratified in the dedicated table.
            lines.append(f"| {label} | {len(rs)} | {np.median([r['upper_over_actual_median'] for r in rs]):.4g} | {np.median([r['lower_over_actual_median'] for r in rs]):.4g} |")
    lines.extend(["", "This table is a descriptive distribution over the executed class, not an effect estimate adjusted for spectrum or noise. "
        "Per-case metrics preserve all factors for fixed-setting comparisons.", "",
        "## Matched-loss spectral allocation", "",
        "Positive differences mean Muon retains a larger fraction of the remaining loss in the specified high band. "
        "At matched total excess loss, the complementary band then retains less loss. Neither statement is a uniform improvement over all directions.", "",
        "The fixed d=32, Gaussian scale=0.5, baseline-spectrum results at F/F0=0.01 are:", "",
        "| Spectrum | Noise sigma | Seeds | GD mean high share | Muon mean high share | Difference min-max |", "|---|---:|---:|---:|---:|---|"])
    for family in FAMILIES:
        for sigma in (.1, 1., 3.):
            rs = [r for r in arrays["allocation"] if r["epsilon"] == .01 and r["family"] == family and is_baseline(r, sigma)]
            if rs:
                lines.append(f"| {family} | {sigma:g} | {len(rs)} | {np.mean([r['gd_fraction'] for r in rs]):.5f} | {np.mean([r['muon_fraction'] for r in rs]):.5f} | [{min(r['difference'] for r in rs):.5f}, {max(r['difference'] for r in rs):.5f}] |")
    lines.extend(["", "All prescribed checkpoints, including initialization-scale changes and controls, are counted separately below:", "",
        "| Study | Relative loss | Resolved positive | Resolved negative | Unresolved | Analytic rank-one ties | Total |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in aggregate["sign_counts"]:
        lines.append(f"| {row['study']} | {row['epsilon']:g} | {row.get('positive', 0)} | {row.get('negative', 0)} | {row.get('unresolved', 0)} | {row.get('analytic_tie', 0)} | {row['cases']} |")
    negative = [r for r in arrays["allocation"] if r["sign"] == "negative"]
    lines.extend(["", f"There are {len(negative)} resolved negative case/checkpoint differences among the unique records. "
        "All positive, negative and unresolved observations are retained in matched_allocation.csv. "
        "The above membership counts describe coverage; they do not pool effect sizes across different settings. "
        "grouped_allocation.csv provides every fixed-setting mean, range, seed count and resolved-sign count.", ""])
    if negative:
        lines.extend(["The most negative resolved observation at each checkpoint is shown below. These are explicitly selected extremes, not representative effect sizes.", "",
            "| Relative loss | Spectrum | Seed | sigma | Shape parameter | W0 scale | Difference | Adjacent-grid difference |",
            "|---|---|---:|---:|---:|---:|---:|---:|"])
        for eps in EPS:
            rs = [r for r in negative if r["epsilon"] == eps]
            if rs:
                r = min(rs, key=lambda row: row["difference"])
                lines.append(f"| {eps:g} | {r['family']} | {r['seed']} | {r['sigma']:g} | {r['parameter']:.5g} | {r['init_scale']:g} | {r['difference']:.6g} | {r['adjacent_grid_uncertainty']:.6g} |")
        scale_negatives = [r for r in negative if "initialization_scale" in r["study_labels"]]
        if scale_negatives:
            lines.extend(["", "Changing Gaussian initialization magnitude can also reverse the allocation comparison. Resolved negative counts within that prescribed study are:", "",
                "| Relative loss | W0 scale | Negative / recorded cases |", "|---:|---:|---:|"])
            for eps in EPS:
                for scale in sorted({r["init_scale"] for r in scale_negatives}):
                    count = sum(r["epsilon"] == eps and r["init_scale"] == scale for r in scale_negatives)
                    if count:
                        denominator = sum(r["epsilon"] == eps and r["init_scale"] == scale and "initialization_scale" in r["study_labels"] for r in arrays["allocation"])
                        lines.append(f"| {eps:g} | {scale:g} | {count} / {denominator} |")
    lines.extend(["", "## Restarted Gamma and chi certificates", "",
        "Each certificate restarts at the first saved point at or below F/F0 in {1, 0.5, 0.1, 0.01}; the actual restart loss is saved. "
        "Threshold eligibility requires a nonzero band and alpha strictly above Gamma or chi. Gamma controls actual high-band fraction monotonicity; chi provides decreasing gradient-axis and loss-axis upper envelopes. "
        "Checks stop at the last saved point with global F/F0 >= 1e-4. Eligibility alone does not imply a numerically useful bound: a chi bound is useful here only if it drops below one. "
        "An additional flag records whether it improves on the restart fraction. Gamma usefulness requires an improvement on its restart-fraction bound.", "",
        "| Study | Restart | Certificate | Nonempty eligible cases | Useful cases | Total |", "|---|---:|---|---:|---:|---:|"])
    for row in aggregate["restart_coverage"]:
        lines.append(f"| {row['study']} | {row['restart']:g} | {row['threshold']} | {row['any_eligible']} | {row['any_useful']} | {row['cases']} |")
    lines.extend(["", "Maximum positive discrepancies over eligible saved bands:"])
    for key, value in aggregate["certificate_maxima"].items():
        lines.append(f"- {key}: {value:.6g}.")
    lines.extend(["", "These are numerical checks of the theoretical implications, not proofs. "
        "restart_certificates.csv includes every cutoff, its dimension, eligibility, terminal slack, informativeness, violations, and actual monotonicity checks. "
        "restart_case_metrics.csv records the widest eligible band's dimension and restart loss share, so a single eligible top direction is distinguishable from broad spectral coverage.", "",
        "## Completion and tolerance refinement", "",
        "completion_tolerances.csv follows decreasing relative total-loss tolerances from 1e-1 to 1e-10 and stores the saved crossing brackets. "
        "A flag identifies thresholds below the final positive saved loss, where the estimate interpolates toward the last zero state. Such apparent stabilization must not be interpreted as resolved continuous-time arrival. "
        "refinement_levels.csv reports all executed grid resolutions, extinction estimates, and their successive discrepancies. "
        "case_metrics.csv records both theoretical completion bounds and their ratios to the numerical estimate. Exact finite-time completion remains a theorem; the experiment checks consistency and numerical convergence.", "",
        f"Across recorded cases, upper-bound/numerical-completion ratios lie in [{aggregate['completion_upper_over_numerical_range'][0]:.6g}, {aggregate['completion_upper_over_numerical_range'][1]:.6g}], "
        f"and numerical-completion/lower-bound ratios lie in [{aggregate['numerical_over_completion_lower_range'][0]:.6g}, {aggregate['numerical_over_completion_lower_range'][1]:.6g}]. "
        "A small ratio below one can reflect discretization; retain each case's refinement discrepancy when interpreting such comparisons.", "",
        "## Artifacts and reproducibility", "",
        "Six figure pairs (PNG/PDF) show envelopes and hitting tightness; spectral width/noise; initialization magnitude/geometry; restarted certificates; completion refinement; and dimension/conditioning/multiplicity. "
        "Each figure fixes the displayed non-seed settings. No representative seed substitutes for the cross-seed summaries. "
        "All machine-readable tables, aggregate.json, case_metrics.json, the executed analysis source and its SHA-256 are saved beside this report. "
        "The runner manifest and source snapshot identify the simulation code and frozen grid.", ""])
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf8")


if __name__ == "__main__":
    main()
