# Provenance and release scope

`imported_files.json` records SHA-256 values at import, before sanitizing
machine-specific paths in result metadata. `release_files.json` records the
final published payload, excluding itself and Git internals.

The seven Python files in `experiments/nonlinear/frozen/` are byte-identical
to the executed storage-v3 source snapshot. The fresh-run wrapper is new
packaging code, not the launcher used for the archived experiment.
Original numerical code, protocol, measurements, analysis CSVs and figures
are retained. Result text may replace absolute workstation/runtime paths.

The research manuscript and its LaTeX sources are not included in this repository.

Excluded: checkpoints, dense trajectories, fixed random-input replay banks,
Drive credentials/account metadata, Colab session and authentication records,
monitor logs, obsolete pilots, other Muon projects and temporary files.
The large private artifacts are necessary for bitwise replay of the migrated
nonlinear experiment; protocol-level fresh runs use the same seeds and rules
but may draw different tensors across platforms.

Completion: 450 linear configurations; 24 nonlinear 5000-step runs;
1272 integrated checkpoint tests plus 864 fixed-noise diagnostic rows.
Matched-loss coverage is 4/48 targets and is reported separately from
training/test completion. The report-only recovery changed the plotting
backend environment, not numerical training or evaluation.
