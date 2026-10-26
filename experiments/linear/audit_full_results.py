"""Independent artifact and algebra audit of a completed linear experiment grid.

Reads saved arrays without importing the experiment generator or integrator.
No numerical trajectories are recalculated. Use python -s -B.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    status = json.loads((root / "status.json").read_text(encoding="utf8"))
    records = [json.loads(p.read_text(encoding="utf8")) for p in sorted((root / "records").glob("*.json"))]
    failures = []
    expected = manifest["cases"]
    source_valid = {}
    for name, digest in manifest["code_sha256"].items():
        path = root / "source_snapshot" / name
        source_valid[name] = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    if not all(source_valid.values()):
        failures.append("Executed source snapshot hash mismatch")
    def spec_key(obj):
        return json.dumps({k: obj[k] for k in expected[0] if k != "study_labels"}, sort_keys=True)
    if {spec_key(r) for r in records} != {spec_key(s) for s in expected} or len(records) != len(expected):
        failures.append("Expected and recorded case sets differ")
    largest = {name: 0. for name in ["initial_residual_relative", "initial_directional_relative", "loss_identity_relative", "muon_matched_loss_relative", "gd_matched_loss_relative", "gd_curve_relative", "initial_gamma_relative", "mean_eigenvalue_absolute", "covariance_basis_orthogonality"]}
    for record in records:
        path = root / "trajectories" / (record["case_id"] + ".npz")
        if not path.exists():
            failures.append(record["case_id"] + ": missing trajectory")
            continue
        with np.load(path, allow_pickle=False) as tr:
            t, q, c, F = (tr[k] for k in ("t", "q", "c", "F"))
            r0, Q, w0, lam = (tr[k] for k in ("R0", "Q", "W0", "eigvals"))
            f0 = F[0]
            if not all(np.isfinite(tr[k]).all() for k in ["t", "q", "c", "F", "R0", "Q", "W0", "eigvals"]):
                failures.append(record["case_id"] + ": nonfinite required arrays")
                continue
            if not (np.all(c > 0) and np.all(np.diff(t) > 0) and np.all(q >= 0) and F[-1] == 0):
                failures.append(record["case_id"] + ": invalid trajectory shape or endpoint")
            if not record["refinement_passed"]:
                failures.append(record["case_id"] + ": unresolved refinement gate")
            q0 = np.sum(r0*r0, axis=0)
            residual = w0 @ Q - Q * (lam/c)[None, :]
            eps = tr["matched_epsilons"]
            mu, gd = tr["matched_muon_q"], tr["matched_gd_q"]
            gt = tr["matched_gd_times"]
            gd_formula = q0[None, :] * np.exp(-4*gt[:, None]*c[None, :])
            checks = dict(
                initial_residual_relative=np.linalg.norm(residual-r0)/max(np.linalg.norm(r0), 1e-30),
                initial_directional_relative=np.max(np.abs(q[0]-q0)/np.maximum(q0, 1e-30)),
                loss_identity_relative=np.max(np.abs(q@c-F))/f0,
                muon_matched_loss_relative=np.max(np.abs((mu@c)/f0-eps)/eps),
                gd_matched_loss_relative=np.max(np.abs((gd@c)/f0-eps)/eps),
                gd_curve_relative=np.max(np.abs(gd-gd_formula)/np.maximum(q0[None, :], 1e-30)),
                initial_gamma_relative=abs(tr["S"][0]**2/f0-tr["gamma"][0])/tr["gamma"][0],
                mean_eigenvalue_absolute=abs(lam.mean()-1),
                covariance_basis_orthogonality=np.max(np.abs(Q.T@Q-np.eye(len(c)))),
            )
            for key, value in checks.items():
                largest[key] = max(largest[key], float(value))
            if max(checks.values()) > 1e-8:
                failures.append(record["case_id"] + ": algebra tolerance exceeded")
    if status.get("status") != "completed" or status.get("completed") != len(expected):
        failures.append("Runner has not marked the complete planned grid successful")
    result = dict(passed=not failures, planned_cases=len(expected), recorded_cases=len(records),
                  source_snapshot_verified=source_valid, maxima=largest, failures=failures,
                  audit_scope="Artifact completeness and saved-array algebra; numerical flow accuracy is assessed separately by refinement and reference-solver checks.")
    (root / "independent_audit.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
