"""Execute the frozen full linear grid without modifying completed pilot runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from flow import solve_muon, diagnostics, curve_discrepancy, at_matched_loss, gd_at_matched_loss
from full_cases import build_cases, case_name, generate, grid_counts, protocol

ROOT = Path(__file__).resolve().parent
EPS = np.array([.5, .1, .01, .001, .0001])
SOURCE_NAMES = ("cases.py", "flow.py", "run_pilot.py", "full_cases.py", "run_full.py")
MAX_LEVELS = 8


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def freeze_protocol():
    value = protocol()
    path = ROOT / "full_protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf8")) != value:
        raise RuntimeError("Frozen protocol differs; create a new version instead of overwriting it")
    if not path.exists():
        save_json(path, value)
    return value


def fraction(tr, c, mask):
    q = at_matched_loss(tr, EPS)
    return (q[:, mask] @ c[mask]) / (q @ c)


def compare(coarse, fine, R0, c, high):
    result = curve_discrepancy(coarse, fine)
    differences = np.abs(fraction(coarse, c, high) - fraction(fine, c, high))
    result["max_matched_high_fraction_difference"] = float(differences.max())
    result["matched_high_fraction_differences_by_epsilon"] = differences.tolist()
    result["extinction_difference_over_initial_opnorm"] = result["extinction_difference"] / float(np.linalg.norm(R0, 2))
    return result


def gate(comparison):
    return comparison is not None and all(comparison[key] <= limit for key, limit in
        protocol()["gates"].items())


def old_match(spec, old):
    return (all(spec[key] == old[key] for key in
                ("d", "seed", "family", "parameter", "sigma", "kind", "correlation", "matched_condition"))
            and spec["init_scale"] == .5
            and spec["band_size"] == (spec["d"] // 2 if spec["family"] == "two_band" else 0)
            and spec["target_kappa_c"] == (10. if old["matched_condition"] else None))


def historical_index(source_hash):
    available = []
    for run in ("pilot_d32", "scale_d256"):
        directory = ROOT / "results" / run
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
        verified = all(manifest["code_sha256"].get(name) == source_hash[name]
                       for name in ("cases.py", "flow.py", "run_pilot.py"))
        verified = verified and manifest["numpy"] == np.__version__ and manifest["scipy"] == scipy.__version__
        if not verified:
            print(f"HISTORICAL_REUSE_DISABLED {run}: source/version mismatch", flush=True)
            continue
        for record in json.loads((directory / "summary.json").read_text(encoding="utf8")):
            available.append((directory, record))
    return available


def load_historical(spec, cov, init, c, available):
    for directory, record in available:
        if not old_match(spec, record):
            continue
        with np.load(directory / "trajectories" / f"{record['case_id']}.npz", allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        checks = {"eigvals": cov["eigvals"], "Q": cov["Q"], "W0": init["W0"], "R0": init["R0_eigen"], "c": c}
        if not all(np.array_equal(saved[key], value) for key, value in checks.items()):
            print(f"HISTORICAL_REUSE_DISABLED {record['case_id']}: generated arrays differ", flush=True)
            return None
        diag = record["diagnostics"]
        saved.update(h=diag["numerical_step"], steps=diag["steps"],
                     modal_increase=diag["modal_increase"], balance_error=diag["balance_error"])
        return saved, record, directory
    return None


def run_case(spec, out, historical):
    start = time.perf_counter()
    cov, init, c, high = generate(spec)
    R0 = init["R0_eigen"]
    base = 256 if spec["d"] >= 256 else 512
    old = load_historical(spec, cov, init, c, historical)
    reuse = None
    levels = []
    last = None
    if old is not None:
        fine, previous_record, old_directory = old
        levels = previous_record["refinement_levels"][:]
        resolution = levels[-1]["resolution"]
        coarse = solve_muon(R0, c, resolution=levels[-2]["resolution"])
        last = compare(coarse, fine, R0, c, high)
        levels[-1] = dict(levels[-1], versus_previous=last,
                          expanded_epsilon_gate_rechecked=True)
        reuse = dict(run=str(old_directory.relative_to(ROOT)), case_id=previous_record["case_id"],
                     original_seconds=previous_record["seconds"],
                     exact_arrays_verified=True, penultimate_grid_recomputed=True)
        previous = fine
        resolution *= 2
    else:
        previous = None
        fine = None
        resolution = base
    while len(levels) < MAX_LEVELS and not (len(levels) >= 3 and gate(last)):
        fine = solve_muon(R0, c, resolution=resolution)
        item = dict(resolution=resolution, h=float(fine["h"]),
                    numerical_extinction_time=float(fine["t"][-1]), steps=int(fine["steps"]))
        if previous is not None:
            last = compare(previous, fine, R0, c, high)
            item["versus_previous"] = last
        levels.append(item)
        previous = fine
        resolution *= 2
    diag, env = diagnostics(R0, c, fine)
    q_mu = at_matched_loss(fine, EPS)
    q_gd, t_gd = gd_at_matched_loss(R0, c, EPS)
    name = case_name(spec)
    np.savez_compressed(out / "trajectories" / f"{name}.npz",
        **{key: fine[key] for key in ("t", "q", "F", "g", "S", "rank", "gamma", "chi")},
        eigvals=cov["eigvals"], c=c, R0=R0, Q=cov["Q"], W0=init["W0"],
        q_upper=env["q_upper"], q_lower=env["q_lower"], high_band_mask=high,
        matched_epsilons=EPS, matched_muon_q=q_mu, matched_gd_q=q_gd,
        matched_gd_times=t_gd)
    record = dict(**spec, case_id=name, parameter_used=float(cov["parameter_used"]),
        covariance_mean=float(cov["eigvals"].mean()), diagnostics=diag,
        refinement=last, refinement_passed=bool(gate(last)), refinement_levels=levels,
        high_band_dimension=int(high.sum()),
        high_band_rule="largest_eigenvalue_block" if spec["family"] == "two_band" else "median_with_ties",
        geometry_control=("orthogonal_columns" if spec["correlation"] == 0 else "rank_one_boundary")
            if spec["kind"] == "controlled_correlation" and spec["correlation"] in (0., 1.) else None,
        matched=dict(epsilons=EPS.tolist(), muon_high_fraction=fraction(fine, c, high).tolist(),
                     gd_high_fraction=((q_gd[:, high] @ c[high]) / (q_gd @ c)).tolist()),
        historical_reuse=reuse, seconds=time.perf_counter() - start)
    save_json(out / "records" / f"{name}.json", record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="full_linear_v1")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Development subset; use a distinct run name")
    args = parser.parse_args()
    frozen = freeze_protocol()
    specs = build_cases()
    if args.limit:
        specs = specs[:args.limit]
    source_hash = hashes()
    if args.freeze_only:
        print(json.dumps(grid_counts(specs)), flush=True)
        return
    out = ROOT / "results" / args.run_name
    for subdir in ("trajectories", "records", "errors", "source_snapshot"):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    manifest = dict(profile="full", **grid_counts(specs), cases=specs, seeds=[17, 43, 97],
                    python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    platform=platform.platform(), code_sha256=source_hash,
                    protocol_sha256=hashlib.sha256((ROOT / "full_protocol.json").read_bytes()).hexdigest(),
                    eta_gd=1., eta_muon=1., matched_epsilons=EPS.tolist(),
                    accuracy_gate=frozen["gates"], numerical_method="PSD-resolvent continuous-flow approximation; >=3 step sizes",
                    high_band_rule=frozen["high_band"])
    if (out / "manifest.json").exists():
        previous_manifest = json.loads((out / "manifest.json").read_text(encoding="utf8"))
        for key in ("cases", "code_sha256", "protocol_sha256"):
            if previous_manifest.get(key) != manifest[key]:
                raise RuntimeError("Existing run has different source/configuration; choose a new run name")
    save_json(out / "manifest.json", manifest)
    for name in SOURCE_NAMES + ("full_protocol.json", "full_protocol.md"):
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, out / "source_snapshot" / name)
    historical = historical_index(source_hash)
    records, failures = [], []
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for index, spec in enumerate(specs):
            name = case_name(spec)
            record_path = out / "records" / f"{name}.json"
            try:
                if record_path.exists():
                    record = json.loads(record_path.read_text(encoding="utf8"))
                    if not (out / "trajectories" / f"{name}.npz").exists():
                        raise RuntimeError("Saved record is missing its trajectory")
                    resumed = True
                else:
                    record = run_case(spec, out, historical)
                    resumed = False
                records.append(record)
                print(json.dumps(dict(case=index + 1, total=len(specs), case_id=name,
                    seconds=round(record["seconds"], 2), resumed=resumed,
                    historical_reuse=record["historical_reuse"] is not None,
                    refinement_passed=record["refinement_passed"],
                    fraction_discrepancy=record["refinement"]["max_matched_high_fraction_difference"])), flush=True)
            except Exception as exc:
                failure = dict(case_id=name, spec=spec, error=str(exc), traceback=traceback.format_exc())
                failures.append(failure)
                save_json(out / "errors" / f"{name}.json", failure)
                print(json.dumps(dict(case=index + 1, total=len(specs), case_id=name, error=str(exc))), flush=True)
            save_json(out / "summary.json", records)
            save_json(out / "status.json", dict(completed=len(records), attempted=index + 1, total=len(specs),
                failures=len(failures), refinement_passed=sum(r["refinement_passed"] for r in records),
                elapsed_seconds=time.perf_counter() - started, status="running"))
    status = dict(completed=len(records), attempted=len(specs), total=len(specs), failures=len(failures),
                  refinement_passed=sum(r["refinement_passed"] for r in records),
                  historical_reused=sum(r["historical_reuse"] is not None for r in records),
                  elapsed_seconds=time.perf_counter() - started,
                  status="completed" if not failures and all(r["refinement_passed"] for r in records) else "completed_with_unresolved_cases")
    save_json(out / "status.json", status)
    print(json.dumps(status), flush=True)
    if failures or status["refinement_passed"] != len(specs):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
