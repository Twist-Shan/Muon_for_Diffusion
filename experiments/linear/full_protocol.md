# Frozen full linear experiment protocol, version 1

This extends Section 4.1 of `pipeline_diffusion.tex` and the completed pilot. Exact grid choices are implementation decisions made before the expanded runs. Seeds are 17, 43 and 97, independent of calendar dates. The full machine-readable specification is `full_protocol.json`.

- Spectrum width: d=32, sigma={0.1,1,3}, all seeds. Log-normal beta and power exponent a each take {0.25,0.5,1,1.5,2}; geometric h takes log({3,10,100,1000}); two-band gap r takes {2,5,20,100}, with half the dimensions in each band. There are 162 pairs.
- Two-band size: d=32, sigma=0.3, r=20, upper-band fractions {0.125,0.25,0.5,0.75}, all seeds: 12 pairs.
- Initialization scale: d=32, baseline spectra, sigma={0.1,1,3}, Gaussian W0 standard deviation a/sqrt(d), a={0.1,0.5,1,2}, all seeds: 144 pairs; 36 overlap the width sweep.
- Initialization geometry: d=32, baseline spectra, sigma=0.3, unit residual-column norms and common correlations rho={0,0.2,0.5,0.8,0.95,0.99,1}, all seeds: 84 pairs. Rho=0 and rho=1 are orthogonal-column and rank-one numerical controls. Deterministic spectra give the same Gram matrix across rotation seeds; these repetitions do not create independent geometry replicates.
- Matched observation conditioning: d=32, sigma=0.3, kappa(C)={2,5,10,20}, all four families and all seeds: 48 pairs. These targets are attainable for the half-size two-band family, whose supremum is 1+2/0.09.
- Dimension: d={64,128,256}, sigma=1, four baseline families and all seeds: 36 pairs.

The union has **450 distinct paired cases**. Each pair uses identical covariance and physical W0 for GD and Muon. The covariance eigenbasis and underlying Gaussian spectral draw stay fixed within each seed/dimension sweep.

GD is exact. Muon is the float64 PSD-resolvent approximation of the continuous compact-polar flow, retaining small positive eigenvalues. At least three step sizes are evaluated, beginning at resolution 512 below d=256 and 256 at d=256; resolution doubles up to eight levels. This extends the pilot's six-level cap to accommodate the additional late checkpoint and rank-boundary controls. Acceptance requires initial-normalized directional discrepancy <=0.001, matched high-band fraction discrepancy <=0.01 at every epsilon in {0.5,0.1,0.01,0.001,0.0001}, and extinction-time discrepancy divided by initial residual operator norm <=0.003. These are empirical refinement checks, not certified continuous-time error bounds. Sampling and interpolation remain part of the measured numerical error.

Historical reuse requires matching source hashes, NumPy/SciPy versions, complete specifications and bitwise identical covariance eigenvalues, Q, W0 and R0. The penultimate grid is recalculated to apply the extended epsilon gate. If it fails, finer levels are calculated. The original runs and source files remain unchanged.

The high band is the entire top eigenspace for two-band cases, and c>=median(c), retaining all ties, for other families. The top-block choice is necessary when varying two-band size: a median cutoff otherwise includes all directions for upper-band fractions below one half. All eigenspaces remain separately available for block-wise analysis.

Report every case, including failures and failed gates. Report observed discrepancies with optimizer differences; signs smaller than the measured numerical uncertainty are unresolved. Distinguish numerical extinction and tolerance hitting from the theorem's exact arrival time. Gamma guarantees actual fraction monotonicity when applicable; chi bounds do not by themselves prove actual monotonicity. No result here establishes discrete-optimizer compute superiority, learning-rate fairness, or nonlinear behavior.

Reproduce from this directory with `C:/anaconda3/python.exe -s -B run_full.py --run-name full_linear_v1`. Run metadata, frozen sources, per-case records, and trajectories are saved under `results/full_linear_v1`.
