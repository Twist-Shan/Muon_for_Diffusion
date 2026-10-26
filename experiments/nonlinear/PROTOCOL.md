# Gaussian nonlinear denoising study

This is a new experiment. It does not reuse earlier pilot results.

The model reproduces the Gaussian-task backbone in Wang and Pehlevan,
[Appendices I.2 and I.4](https://arxiv.org/html/2503.03206v3): four residual
UNetMLPBlock layers and one output linear layer, dimension 128, hidden width
256, Fourier embedding width 256. The author implementation is pinned to
[commit 94965885](https://github.com/Animadversio/DiffusionLearningCurve/blob/94965885b5d94a8fcba0c3b6a31b895d3eb5deee/core/diffusion_nn_lib.py).
`model_validation.json` independently verifies the extracted classes against
that source. The block Linear layers use PyTorch's standard initialization;
the `init_zero` argument is an empty dictionary, not zero initialization.

EDM uses sigma_data=0.5 (thus s=0.25), P_mean=-1.2, P_std=1.2. The training
target is the sampled clean X, never the known Bayes denoiser. Unlike WP's
finite dataset and Adam training, this study uses fresh Gaussian minibatches,
SGD and exact compact-SVD Simplified Muon, and direct denoising diagnostics.
There are no generated-sample metrics.

## Frozen protocol

`protocol.json` is the machine-readable specification. Each of four covariance
families has mean eigenvalue one: lognormal beta=1, power law exponent=1,
geometric range=100, and two bands with ratio=20 and equal multiplicities.
Within each independent seed the eigenbasis is shared across families.
Main seeds are 17, 43, 97; the separate search seed is 137. Seeds have no date
dependence. Every SGD/Muon pair shares its initialization and full minibatch,
corruption-noise, and noise-level sequence.

Each formal run has 5000 updates and batch size 1024. Learning rates are
constant. SGD uses its scalar rate on all trainable parameters. Muon applies
the compact polar factor to all 14 trainable weight matrices without momentum,
weight decay, aspect-ratio scaling, or RMS scaling; all auxiliary vectors and
biases use the corresponding family's selected SGD rate. Both main optimizers
therefore have the identical auxiliary SGD update rule and rate.

Learning-rate search uses 512 steps per candidate, a separate validation sample
of 4096 observations, and final integrated weighted Bayes excess as its sole
selection criterion. Base candidates are SGD [0.001,0.01,0.1,1] and Muon
[0.0001,0.001,0.01,0.1]. A boundary winner triggers expansion by a factor of
10 at both ends, at most twice. Each method has the same base and maximum
search budget (4 and 8 candidates); actual adaptive counts are reported.
SGD selection precedes Muon selection so the shared auxiliary rate is fixed
before comparing Muon candidates. A remaining boundary winner is explicitly
flagged. Search results are not part of the three-seed test statistics.

All matrix SVDs use SciPy LAPACK gesdd on CPU float64, with numerical rank
threshold max(m,n)*eps64*s_max. Updates are cast back to float32. This is a
numerical compact SVD, not Newton--Schulz or a Gram-eigenvalue approximation.
`runner_tests.json` checks rectangular, rank-deficient, and zero matrices.

## Checkpoints and measurements

Actual parameter states are saved at steps 0,25,50 and every 100 updates.
Recovery states are also saved every 100 updates during search. Checkpoints
include the training RNG state; interrupted updates are discarded and replayed
from the last fully saved state. Current immutable executed code is in
`source_snapshot_storage_v3`; the original `source_snapshot` is retained.
The source hashes and complete run settings are checked before resuming.

Integrated held-out test loss uses 8192 observations, independent of search,
training, and validation. Fixed-noise diagnostics use sigma in [0.1,1,3].
At each diagnostic checkpoint, two independent samples of 8192 observations
fit the Bayes-residual linear component by least squares in whitened input
coordinates. The average fit estimates W_lin; the split-fit difference gives
a diagnostic of its Monte Carlo uncertainty. This estimator is consistent
with E[D Y^T] C^-1 and exactly recovers linear functions in exact arithmetic.
Subtracting the known Bayes predictor before fitting avoids an avoidable
Monte Carlo floor for a nearly optimal denoiser.

A separate 8192-sample test set measures total Bayes excess, fitted-linear
error, the remaining nonlinear residual, and their cross term. Their sample
decomposition identity is checked; the cross term is not silently assumed
zero. Directional errors are input-direction norms ||(W_lin-W*)v_j||^2.
Repeated eigenvalues are aggregated over their eigenspace for learning-time
statistics. The estimated linear component and the nonlinear residual are
not treated as theoretical linear-model trajectories.

Matched-loss comparisons select true saved checkpoint pairs using validation
only, before reading test results. Target relative excesses are 0.5,0.25,0.1,
0.05; each optimizer must be within 20% of the target and the pair's relative
loss gap must be at most 2%. Unmatched targets are reported. Independent test
gaps and Monte Carlo standard errors are then reported, with no reselection.
There is no parameter interpolation. All saved checkpoints receive integrated
test evaluation; fixed-noise fits use the 12 prespecified diagnostic steps
and any extra validation-selected checkpoint pairs.

## Execution and recovery

`launch.ps1` starts the immutable runner as a hidden independent process.
The launcher records its PID and paths in `launch.json`; progress appears in
the unique `logs/<launch_id>.stdout.log`, `search_progress.json`, and
`main_progress.json`. Errors appear in that launch's `.stderr.log` and
individual `error.json` files. Log paths are recorded in `launch.json`.
Re-running the launcher resumes completed searches and validated checkpoints.
Do not run two workers simultaneously and do not edit the frozen protocol or
source snapshot while the study is running.

### Storage recovery revision

The previous worker stopped while replacing `recovery.pt` at search step 200.
The committed step-100 state is intact. The uncommitted `recovery.tmp` is
retained for audit and is not used for recovery. The v3 storage-only revision
resumes step 100, preserves all 13 completed search candidates, and leaves
the scientific protocol, source numerical operations, and budgets unchanged.
Original source and failure artifacts are retained in
`storage_revision_v2_archive`; `compatibility_receipt.json` records every
accepted old hash and the new execution hashes. Old checkpoint metadata is
not rewritten to claim a new source version.

Files are now saved as immutable, SHA256-verified versions under
`.versions/<logical_filename>/`, followed by a commit record. Search recovery
filenames include their step. Uncommitted or corrupted versions are ignored.
Canonical filenames are convenience mirrors: use `storage.read_json`,
`storage.resolve_artifact`, and `storage.list_artifacts` for authoritative
reads. A locked or stale mirror cannot invalidate the committed version.
JSON, checkpoint, diagnostic NPZ, and report writes use the same mechanism.
Each launch has separate logs and the runner holds the OS worker lock.

`storage_resume_validation.json` records bitwise-equal parameter, RNG, and
validation metric results for interrupted versus uninterrupted SGD and Muon.
`verification/storage_review_latest.json` records independent I/O fault tests,
including a real Windows read handle, partial commits, corrupt archives, and
the unchanged scientific update loop. `storage_migration_validation.json`
records preservation and compatibility of the actual existing artifacts.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\output\diffusion_nonlinear_learning\launch.ps1
```

The installed runtime uses `C:/anaconda3/python.exe -s`, system NumPy 1.26.4
and SciPy 1.13.1, then appends the existing user-site PyTorch 2.5.1+cu121.
No package installation is required. The GPU is the local RTX 2000 Ada laptop.
The 50-step benchmark measured about 0.44 s for SGD and 9.34 s for exact-SVD
Muon, excluding evaluation/checkpoint overhead. The benchmark used an earlier
cosine smoke schedule and is a timing measurement only; formal runs use the
constant-rate frozen protocol. That early measurement suggested 3–4 hours.
The initial formal runs instead measured approximately 0.33–0.76 seconds per
Muon step as local performance changed, implying roughly 6–13 hours for the
complete budget including search. This is an estimate, not a completion
promise; use `status.py` and actual checkpoint progress for current status.
Reported `training_seconds` excludes validation and checkpoint writes.

On successful execution, `study_complete.json` reports the number of full
budget runs, divergent runs, and evaluated checkpoints. It does not declare
all runs successful if any run diverges. `analysis` contains CSV tables and
figures made from actual saved checkpoints. Three-seed means and standard
deviations are descriptive; they do not establish a universal optimizer
advantage or prove the linear theory applies to learned nonlinear features.

## Guard revision and provenance

The v2 operational revision adds finite validation before checkpoint commits,
history recovery, explicit missing-diagnostic reporting, complete checkpoint
coverage checks, and a failed-worker status on unexpected exceptions. Nine
fault-injection checks passed and an independent code review accepted these
guards. No scientific setting, numerical training operation, or budget changed.

The earlier single-worker snapshot, logs, launch receipt, and unfinished search
candidate are preserved in `guard_revision_v1_archive`. Nine completed search
candidates are reused through `compatibility_receipt.json`, which binds the
old and new source hashes and every retained metadata, history, completion,
and checkpoint file. Their original files and source hashes are unchanged.
The unfinished candidate restarts from initialization. `migration_validation.json`
records the read-only artifact checks and frozen-source hash verification.

The process holds an operating-system file lock for its entire lifetime.
An unprivileged process inventory can omit a worker launched in another Windows
sandbox context; it is not evidence that training stopped. Check actual log and
checkpoint advancement and the held lock before considering any restart.
