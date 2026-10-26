"""Plot the linear-denoising pilot without inferring optimizer superiority.

Usage: python plot_results.py --run-dir results/pilot

The saved polar trajectory is a numerical approximation of the continuous
flow. GD is evaluated from its closed-form trajectory at matching total
relative loss. Repeated covariance eigenvalues are aggregated in every plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flow import at_matched_loss, gd_at_matched_loss, spectral_groups


FAMILIES = ("lognormal", "powerlaw", "geometric", "two_band")
FAMILY_NAMES = {
    "lognormal": "Log-normal",
    "powerlaw": "Power law",
    "geometric": "Geometric",
    "two_band": "Two bands",
}
EPSILONS = np.geomspace(1.0, 1e-3, 161)
POLAR_COLOR = "#1b6c8e"
GD_COLOR = "#ba5d32"


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_records(run_dir):
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("summary.json must contain a list of case records")
    available = []
    for record in records:
        path = run_dir / "trajectories" / f"{record['case_id']}.npz"
        if path.is_file():
            available.append(record)
        else:
            warnings.warn(f"Skipping missing trajectory: {path}")
    if not available:
        raise ValueError("No saved trajectories were found for the summary records")
    return available


def load_trajectory(run_dir, record):
    path = run_dir / "trajectories" / f"{record['case_id']}.npz"
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {key: archive[key] for key in archive.files}
    for key in ("t", "q", "F", "c", "R0"):
        if key not in trajectory:
            raise ValueError(f"{path.name} is missing {key!r}")
    q, c = trajectory["q"], trajectory["c"]
    if q.ndim != 2 or q.shape != (trajectory["t"].size, c.size):
        raise ValueError(f"{path.name} has inconsistent trajectory dimensions")
    if trajectory["F"][0] <= 0 or np.any(c <= 0):
        raise ValueError(f"{path.name} requires a nonzero initial loss and positive c")
    return trajectory


def representative(records, family):
    candidates = [r for r in records if r["family"] == family
                  and r["kind"] == "gaussian_w0"]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (
        bool(r.get("matched_condition", False)),
        abs(float(r["sigma"]) - 1.0),
        int(r["seed"]) != 17,
        int(r["seed"]),
        r["case_id"],
    ))


def geometry_records(records, family):
    candidates = [r for r in records if r["family"] == family
                  and r["kind"] == "controlled_correlation"]
    groups = {}
    for record in candidates:
        key = (float(record["sigma"]), int(record["seed"]), int(record["d"]),
               float(record["parameter"]), bool(record.get("matched_condition", False)))
        groups.setdefault(key, []).append(record)
    if not groups:
        return []
    key = min(groups, key=lambda k: (
        k[4], abs(k[0] - 0.3), k[1] != 17,
        -len({float(r["correlation"]) for r in groups[k]}), k[1], k[2], k[3],
    ))
    # A repeated run record must not silently draw the same correlation twice.
    unique = {}
    for record in sorted(groups[key], key=lambda r: r["case_id"]):
        unique.setdefault(float(record["correlation"]), record)
    return [unique[rho] for rho in sorted(unique)]


def title_for(record):
    return (f"{FAMILY_NAMES[record['family']]}  |  "
            rf"$\sigma={float(record['sigma']):g}$, seed {record['seed']}")


def empty_panel(ax, family, description):
    ax.set_title(FAMILY_NAMES[family])
    ax.text(0.5, 0.5, description, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def finish_figure(fig, figure_dir, name, title, note):
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.text(0.08, 0.014, note, ha="left", va="bottom", fontsize=8, color="#454545")
    fig.tight_layout(rect=(0.01, 0.075, 0.995, 0.95), h_pad=2.0, w_pad=2.0)
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"{name}.{extension}", facecolor="white")
    plt.close(fig)


def modal_figure(records, run_dir, figure_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.7))
    selected = {}
    for family, ax in zip(FAMILIES, axes.flat):
        record = representative(records, family)
        if record is None:
            empty_panel(ax, family, "No Gaussian-initialization trajectory available")
            continue
        tr = load_trajectory(run_dir, record)
        selected[family] = record["case_id"]
        groups = spectral_groups(tr["c"])
        indices = np.unique(np.rint(np.linspace(0, len(groups) - 1, min(5, len(groups)))).astype(int))
        colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.86, indices.size))
        for index, color in zip(indices, colors):
            group = groups[index]
            initial = float(tr["q"][0, group].sum())
            if initial <= 0:
                continue
            curve = tr["q"][:, group].sum(axis=1) / initial
            label = rf"$c={tr['c'][group[0]]:.3g}$"
            if group.size > 1:
                label += f" (dim {group.size})"
            ax.plot(tr["t"], np.where(curve > 0, curve, np.nan),
                    color=color, lw=1.8, label=label)
            if "q_upper" in tr and "q_lower" in tr:
                upper = tr["q_upper"][:, group].sum(axis=1) / initial
                lower = tr["q_lower"][:, group].sum(axis=1) / initial
                ax.fill_between(tr["t"], np.maximum(lower, 1e-7),
                                np.maximum(upper, 1e-7), color=color, alpha=0.075)
                for bound in (upper, lower):
                    ax.plot(tr["t"], np.where(bound > 0, bound, np.nan),
                            color=color, lw=0.9, ls=":", alpha=0.8)
        ax.set(title=title_for(record), xlabel="Training time $t$",
               ylabel="Relative squared residual", yscale="log", ylim=(1e-6, 1.15))
        ax.grid(axis="y", which="major", alpha=0.18)
        ax.legend(loc="lower left", frameon=False)
    finish_figure(fig, figure_dir, "01_directional_bounds",
                  "Directional error and continuous-flow bounds",
                  "Solid: numerical polar flow. Dotted lines and shading: theoretical bounds.\n"
                  "Repeated eigenvalues are aggregated over their full eigenspace; at most five eigenspaces are shown per panel.")
    return selected


def matched_curves(tr):
    # Whole tied eigenspaces are included; a top-d/2 index slice is not equivalent.
    mask = tr["c"] >= np.median(tr["c"])
    q_polar = at_matched_loss(tr, EPSILONS)
    q_gd, _ = gd_at_matched_loss(tr["R0"], tr["c"], EPSILONS)
    polar = (q_polar[:, mask] @ tr["c"][mask]) / (q_polar @ tr["c"])
    gd = (q_gd[:, mask] @ tr["c"][mask]) / (q_gd @ tr["c"])
    return polar, gd, int(mask.sum())


def format_loss_axis(ax):
    ax.set(xscale="log", xlim=(1, EPSILONS[-1]), ylim=(-0.025, 1.025),
           xlabel=r"Relative total loss $F/F(0)$", ylabel="High-spectrum share of remaining loss")
    ax.grid(alpha=0.18)


def matched_figure(records, run_dir, figure_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.7))
    selected = {}
    for family, ax in zip(FAMILIES, axes.flat):
        record = representative(records, family)
        if record is None:
            empty_panel(ax, family, "No Gaussian-initialization trajectory available")
            continue
        tr = load_trajectory(run_dir, record)
        polar, gd, dimension = matched_curves(tr)
        selected[family] = {"case_id": record["case_id"], "high_band_dimension": dimension}
        ax.plot(EPSILONS, polar, lw=2, color=POLAR_COLOR, label="Polar flow (numerical)")
        ax.plot(EPSILONS, gd, lw=1.8, ls="--", color=GD_COLOR, label="GD flow")
        format_loss_axis(ax)
        ax.set_title(title_for(record))
        ax.legend(frameon=False)
    finish_figure(fig, figure_dir, "02_matched_loss_spectrum",
                  "High-spectrum loss at matched total loss",
                  r"Band: $c\geq\mathrm{median}(c)$, including all ties. Polar curves interpolate the saved numerical trajectory; GD uses its exact flow formula."
                  "\nLoss decreases from left to right. These curves describe error allocation, not a learning-rate or wall-clock advantage.")
    return selected


def geometry_figure(records, run_dir, figure_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.7))
    selected = {}
    for family, ax in zip(FAMILIES, axes.flat):
        group = geometry_records(records, family)
        if not group:
            empty_panel(ax, family, "No controlled-correlation trajectory available")
            continue
        selected[family] = [r["case_id"] for r in group]
        colors = plt.get_cmap("plasma")(np.linspace(0.12, 0.82, len(group)))
        gd_curves = []
        for record, color in zip(group, colors):
            polar, gd, _ = matched_curves(load_trajectory(run_dir, record))
            gd_curves.append(gd)
            ax.plot(EPSILONS, polar, color=color, lw=1.9,
                    label=rf"Polar, $\rho={float(record['correlation']):g}$")
        if all(np.allclose(curve, gd_curves[0], rtol=1e-9, atol=1e-11) for curve in gd_curves):
            ax.plot(EPSILONS, gd_curves[0], color="#333333", lw=1.5,
                    ls="--", label=r"GD (all $\rho$)" if len(group) > 1 else "GD")
        else:
            for record, color, curve in zip(group, colors, gd_curves):
                ax.plot(EPSILONS, curve, color=color, lw=1.0, ls="--",
                        label=rf"GD, $\rho={float(record['correlation']):g}$")
        format_loss_axis(ax)
        ax.set_title(title_for(group[0]))
        ax.legend(frameon=False)
    finish_figure(fig, figure_dir, "03_initialization_geometry",
                  "Initialization geometry at matched total loss",
                  r"Each initial residual column has unit norm; $\rho$ sets its pairwise inner products. Band: $c\geq\mathrm{median}(c)$, including ties."
                  "\nPolar curves approximate continuous flow. Each panel shows one available paired seed and spectrum; no averaging or monotonicity is assumed.")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Directory containing summary.json and trajectories/")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    records = load_records(run_dir)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    manifest = {
        "run_directory": str(run_dir),
        "matched_loss_range": [float(EPSILONS[0]), float(EPSILONS[-1])],
        "high_band_definition": "c >= median(c), including all ties",
        "numerical_note": "Polar trajectories approximate continuous flow; GD uses its closed-form flow with a scalar loss-matching root solve.",
        "directional_cases": modal_figure(records, run_dir, figure_dir),
        "matched_loss_cases": matched_figure(records, run_dir, figure_dir),
        "geometry_cases": geometry_figure(records, run_dir, figure_dir),
    }
    with (figure_dir / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved three PNG/PDF figure pairs and figure_manifest.json to {figure_dir}")


if __name__ == "__main__":
    main()
