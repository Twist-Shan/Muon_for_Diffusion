"""Check completeness and numerical consistency of the expanded run artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from full_cases import case_name


def validate(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    records = json.loads((directory / "summary.json").read_text(encoding="utf8"))
    expected = {case_name(spec) for spec in manifest["cases"]}
    expected_specs = {case_name(spec): spec for spec in manifest["cases"]}
    actual = {record["case_id"] for record in records}
    assert len(records) == len(expected) == len(actual), "Summary has missing or duplicate cases"
    assert expected == actual, "Summary case IDs differ from frozen specifications"
    assert {p.stem for p in (directory / "records").glob("*.json")} == expected
    assert {p.stem for p in (directory / "trajectories").glob("*.npz")} == expected
    assert not list((directory / "errors").glob("*.json")), "Run contains error records"
    for name, expected_hash in manifest["code_sha256"].items():
        assert hashlib.sha256((directory / "source_snapshot" / name).read_bytes()).hexdigest() == expected_hash
    assert hashlib.sha256((directory / "source_snapshot/full_protocol.json").read_bytes()).hexdigest() == manifest["protocol_sha256"]
    max_loss_identity_error = 0.
    for record in records:
        identity = record["case_id"]
        assert all(record[key] == value for key, value in expected_specs[identity].items())
        assert record == json.loads((directory / "records" / f"{identity}.json").read_text(encoding="utf8"))
        assert record["refinement_passed"] and len(record["refinement_levels"]) >= 3
        for key, tolerance in manifest["accuracy_gate"].items():
            assert record["refinement"][key] <= tolerance, (identity, key)
        with np.load(directory / "trajectories" / f"{identity}.npz", allow_pickle=False) as tr:
            t, q, F, c = (tr[key] for key in ("t", "q", "F", "c"))
            assert q.shape == (len(t), record["d"])
            assert all(np.all(np.isfinite(tr[key])) for key in ("t", "q", "F", "c", "R0", "W0", "Q"))
            assert np.all(np.diff(t) > 0) and t[0] == 0 and F[0] > 0 and F[-1] == 0
            assert np.all(q >= 0) and np.all(c > 0)
            discrepancy = float(np.max(np.abs(q @ c - F)) / F[0])
            max_loss_identity_error = max(max_loss_identity_error, discrepancy)
            assert discrepancy < 1e-12
            high = tr["high_band_mask"]
            assert high.dtype == np.bool_ and int(high.sum()) == record["high_band_dimension"]
            if record["family"] == "two_band":
                assert np.array_equal(high, c == c.max()) and int(high.sum()) == record["band_size"]
            assert np.array_equal(tr["matched_epsilons"], record["matched"]["epsilons"])
            qm = tr["matched_muon_q"]
            fractions = (qm[:, high] @ c[high]) / (qm @ c)
            assert np.allclose(fractions, record["matched"]["muon_high_fraction"], atol=1e-13, rtol=1e-13)
    return dict(passed=True, completed_cases=len(records), exact_manifest_case_set=True,
                source_snapshot_hashes_verified=True, minimum_refinement_levels=min(len(r["refinement_levels"]) for r in records),
                maximum_refinement_levels=max(len(r["refinement_levels"]) for r in records),
                refinement_level_counts=dict(Counter(len(r["refinement_levels"]) for r in records)),
                maximum_loss_identity_relative_error=max_loss_identity_error,
                maximum_refinement_discrepancies={key: max(r["refinement"][key] for r in records)
                    for key in manifest["accuracy_gate"]},
                historical_reused=sum(r["historical_reuse"] is not None for r in records),
                validation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("results/full_linear_v1"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = validate(args.run_dir)
    (args.run_dir / "artifact_validation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps(result), flush=True)
