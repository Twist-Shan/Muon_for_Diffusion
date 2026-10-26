# Expanded linear denoising experiment

Analyzed 450/450 prescribed unique configurations; 450/450 passed the numerical refinement gate. The recorded grid is complete.

These experiments compare exact population GD with a numerical approximation of continuous compact-polar flow. The estimand is directional denoising error and spectral allocation at matched relative excess loss. The data do not establish a universal GD-versus-Muon ordering, a discrete optimizer speed advantage, or a nonlinear result.

## Coverage and numerical interpretation

| Study membership | Recorded configurations |
|---|---:|
| dimension | 36 |
| spectrum_width | 162 |
| initialization_scale | 144 |
| initialization_geometry | 84 |
| matched_conditioning | 48 |
| two_band_size | 12 |

Memberships overlap; each unique configuration is executed once. Seeds are fixed independently of calendar dates. All summaries in grouped_allocation.csv hold every configuration field fixed and vary only seed. The figures show seed means and min-max ranges, not confidence intervals. With deterministic spectra and a fixed residual Gram matrix, controlled-correlation seed rotations reproduce the same dynamics; these repeats are not independent statistical replicates.

At rho=0 and rho=1, the orthogonal-column and rank-one cases are analytic/rank-boundary controls. The intermediate correlations provide the general geometry experiment. geometry_effects.csv quantifies the Muon allocation variation and GD invariance across rho at each fixed spectrum, seed and checkpoint. For two-band spectra, the high band is the entire largest-eigenvalue eigenspace, including unequal multiplicities. Other spectra use the median cutoff including all ties.

The saved refinement columns report observed differences between the final two step sizes. They are empirical numerical uncertainty indicators, not certified errors of the continuous solution. A sign is called resolved only when its magnitude exceeds twice its own checkpoint-specific adjacent-grid difference (and 1e-12), and the case passes the refinement gate. This terminology does not certify the continuous-flow sign. The uninflated one-discrepancy classification is retained as raw_one_discrepancy_sign for sensitivity analysis. Rank-one rho=1 controls are labeled analytic_tie: both flows share the same spectral trajectory after a time reparameterization and therefore have identical allocation at matched loss. Their nonzero measured differences quantify numerical error. These controls show why a single adjacent-grid difference can slightly underestimate the remaining first-order error, motivating the factor-two empirical margin. Historical runs lacking that checkpoint's refinement check are marked unresolved there.

## Directional bounds and relative-accuracy times

The upper hitting bound is A(log(1/delta)/(2c)); the individual lower bound is a_j(1-sqrt(delta)). A uses operator norms of full eigenspace blocks. For plotted block errors, lower envelopes are summed over columns and their crossing is solved numerically; a Frobenius block norm is not substituted into the single-direction lower formula. Individual directional rows are basis-specific inside tied eigenspaces. Block actual errors and upper envelopes are basis-invariant; the valid summed lower bound uses the fixed saved eigenbasis. block_hitting_times.csv reports these block-level quantities.

Largest hitting-bound discrepancies outside the saved crossing brackets, divided by ||R0||op: lower 0, upper 0. The crossing bracket only measures temporal sampling resolution; solver uncertainty is reported separately. eigenspace_learning_order.csv compares distinct eigenspace hitting times after expanding each crossing window using delta plus/minus the observed last-refinement modal discrepancy. Higher-c earlier, higher-c later and overlapping-window counts quantify learning order without treating unresolved separation as a strict order. These windows are empirical sensitivity diagnostics, not certified error bounds.

Envelope tightness is evaluated where the actual relative block error is between 1e-4 and 0.99. envelope_tightness.csv reports upper/actual and lower/actual ratios and envelope widths, rather than only checking violations. block_hitting_times.csv reports the position inside each theoretical interval and both bound/observed-time ratios.

| Initialization class | Cases | Median of case median upper/actual ratios | Median of case median lower/actual ratios |
|---|---:|---:|---:|
| controlled_correlation, rho=0 | 12 | 3.457 | 1 |
| controlled_correlation, rho=0.2 | 12 | 8.655 | 0 |
| controlled_correlation, rho=0.5 | 12 | 2.947 | 0 |
| controlled_correlation, rho=0.8 | 12 | 1.485 | 0 |
| controlled_correlation, rho=0.95 | 12 | 1.102 | 0 |
| controlled_correlation, rho=0.99 | 12 | 1.02 | 0 |
| controlled_correlation, rho=1 | 12 | 1.001 | 0 |
| gaussian_w0 | 366 | 5.54 | 0.09878 |

This table is a descriptive distribution over the executed class, not an effect estimate adjusted for spectrum or noise. Per-case metrics preserve all factors for fixed-setting comparisons.

## Matched-loss spectral allocation

Positive differences mean Muon retains a larger fraction of the remaining loss in the specified high band. At matched total excess loss, the complementary band then retains less loss. Neither statement is a uniform improvement over all directions.

Unique-case counts below include each configuration once and avoid double-counting overlapping study memberships:

| Relative loss | Resolved positive | Resolved negative | Unresolved | Analytic rank-one ties | Unique cases |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 438 | 0 | 0 | 12 | 450 |
| 0.1 | 438 | 0 | 0 | 12 | 450 |
| 0.01 | 429 | 9 | 0 | 12 | 450 |
| 0.001 | 397 | 27 | 14 | 12 | 450 |
| 0.0001 | 357 | 35 | 46 | 12 | 450 |

The fixed d=32, Gaussian scale=0.5, baseline-spectrum results at F/F0=0.01 are:

| Spectrum | Noise sigma | Seeds | GD mean high share | Muon mean high share | Difference min-max |
|---|---:|---:|---:|---:|---|
| lognormal | 0.1 | 3 | 0.02929 | 0.46249 | [0.38599, 0.51050] |
| lognormal | 1 | 3 | 0.27899 | 0.59928 | [0.23885, 0.41264] |
| lognormal | 3 | 3 | 0.40602 | 0.47760 | [0.05915, 0.08579] |
| powerlaw | 0.1 | 3 | 0.08381 | 0.41363 | [0.25981, 0.37701] |
| powerlaw | 1 | 3 | 0.31110 | 0.61174 | [0.18628, 0.43227] |
| powerlaw | 3 | 3 | 0.41409 | 0.50553 | [0.06283, 0.14521] |
| geometric | 0.1 | 3 | 0.01207 | 0.51497 | [0.44218, 0.55142] |
| geometric | 1 | 3 | 0.21563 | 0.56767 | [0.25633, 0.43552] |
| geometric | 3 | 3 | 0.38078 | 0.46384 | [0.05069, 0.10166] |
| two_band | 0.1 | 3 | 0.00000 | 0.60047 | [0.55402, 0.62985] |
| two_band | 1 | 3 | 0.07989 | 0.52509 | [0.43544, 0.45925] |
| two_band | 3 | 3 | 0.35966 | 0.45214 | [0.05551, 0.12851] |

All prescribed checkpoints, including initialization-scale changes and controls, are counted separately below:

| Study | Relative loss | Resolved positive | Resolved negative | Unresolved | Analytic rank-one ties | Total |
|---|---:|---:|---:|---:|---:|---:|
| dimension | 0.5 | 36 | 0 | 0 | 0 | 36 |
| dimension | 0.1 | 36 | 0 | 0 | 0 | 36 |
| dimension | 0.01 | 36 | 0 | 0 | 0 | 36 |
| dimension | 0.001 | 36 | 0 | 0 | 0 | 36 |
| dimension | 0.0001 | 36 | 0 | 0 | 0 | 36 |
| initialization_geometry | 0.5 | 72 | 0 | 0 | 12 | 84 |
| initialization_geometry | 0.1 | 72 | 0 | 0 | 12 | 84 |
| initialization_geometry | 0.01 | 72 | 0 | 0 | 12 | 84 |
| initialization_geometry | 0.001 | 59 | 0 | 13 | 12 | 84 |
| initialization_geometry | 0.0001 | 49 | 0 | 23 | 12 | 84 |
| initialization_scale | 0.5 | 144 | 0 | 0 | 0 | 144 |
| initialization_scale | 0.1 | 144 | 0 | 0 | 0 | 144 |
| initialization_scale | 0.01 | 140 | 4 | 0 | 0 | 144 |
| initialization_scale | 0.001 | 134 | 10 | 0 | 0 | 144 |
| initialization_scale | 0.0001 | 121 | 14 | 9 | 0 | 144 |
| matched_conditioning | 0.5 | 48 | 0 | 0 | 0 | 48 |
| matched_conditioning | 0.1 | 48 | 0 | 0 | 0 | 48 |
| matched_conditioning | 0.01 | 48 | 0 | 0 | 0 | 48 |
| matched_conditioning | 0.001 | 48 | 0 | 0 | 0 | 48 |
| matched_conditioning | 0.0001 | 44 | 0 | 4 | 0 | 48 |
| spectrum_width | 0.5 | 162 | 0 | 0 | 0 | 162 |
| spectrum_width | 0.1 | 162 | 0 | 0 | 0 | 162 |
| spectrum_width | 0.01 | 157 | 5 | 0 | 0 | 162 |
| spectrum_width | 0.001 | 141 | 20 | 1 | 0 | 162 |
| spectrum_width | 0.0001 | 131 | 25 | 6 | 0 | 162 |
| two_band_size | 0.5 | 12 | 0 | 0 | 0 | 12 |
| two_band_size | 0.1 | 12 | 0 | 0 | 0 | 12 |
| two_band_size | 0.01 | 12 | 0 | 0 | 0 | 12 |
| two_band_size | 0.001 | 12 | 0 | 0 | 0 | 12 |
| two_band_size | 0.0001 | 5 | 0 | 7 | 0 | 12 |

There are 71 resolved negative case/checkpoint differences among the unique records. All positive, negative and unresolved observations are retained in matched_allocation.csv. The above membership counts describe coverage; they do not pool effect sizes across different settings. grouped_allocation.csv provides every fixed-setting mean, range, seed count and resolved-sign count.

The most negative resolved observation at each checkpoint is shown below. These are explicitly selected extremes, not representative effect sizes.

| Relative loss | Spectrum | Seed | sigma | Shape parameter | W0 scale | Difference | Adjacent-grid difference |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.01 | powerlaw | 43 | 3 | 0.25 | 0.5 | -0.0346974 | 1.78359e-05 |
| 0.001 | lognormal | 97 | 3 | 2 | 0.5 | -0.0861234 | 4.53234e-05 |
| 0.0001 | powerlaw | 43 | 3 | 2 | 0.5 | -0.13419 | 0.000332974 |

Changing Gaussian initialization magnitude can also reverse the allocation comparison. Resolved negative counts within that prescribed study are:

| Relative loss | W0 scale | Negative / recorded cases |
|---:|---:|---:|
| 0.01 | 1 | 2 / 36 |
| 0.01 | 2 | 2 / 36 |
| 0.001 | 0.5 | 3 / 36 |
| 0.001 | 1 | 3 / 36 |
| 0.001 | 2 | 4 / 36 |
| 0.0001 | 0.1 | 2 / 36 |
| 0.0001 | 0.5 | 4 / 36 |
| 0.0001 | 1 | 4 / 36 |
| 0.0001 | 2 | 4 / 36 |

## Restarted Gamma and chi certificates

Each certificate restarts at the first saved point at or below F/F0 in {1, 0.5, 0.1, 0.01}; the actual restart loss is saved. Threshold eligibility requires a nonzero band and alpha strictly above Gamma or chi. Gamma controls actual high-band fraction monotonicity; chi provides decreasing gradient-axis and loss-axis upper envelopes. Checks stop at the last saved point with global F/F0 >= 1e-4. Eligibility alone does not imply a numerically useful bound: a chi bound is useful here only if it drops below one. An additional flag records whether it improves on the restart fraction. Gamma usefulness requires an improvement on its restart-fraction bound.

| Study | Restart | Certificate | Nonempty eligible cases | Useful cases | Total |
|---|---:|---|---:|---:|---:|
| dimension | 1 | gamma | 0 | 0 | 36 |
| dimension | 1 | chi | 36 | 27 | 36 |
| dimension | 0.5 | gamma | 0 | 0 | 36 |
| dimension | 0.5 | chi | 36 | 27 | 36 |
| dimension | 0.1 | gamma | 0 | 0 | 36 |
| dimension | 0.1 | chi | 36 | 27 | 36 |
| dimension | 0.01 | gamma | 5 | 5 | 36 |
| dimension | 0.01 | chi | 36 | 27 | 36 |
| initialization_geometry | 1 | gamma | 22 | 22 | 84 |
| initialization_geometry | 1 | chi | 72 | 63 | 84 |
| initialization_geometry | 0.5 | gamma | 48 | 48 | 84 |
| initialization_geometry | 0.5 | chi | 72 | 66 | 84 |
| initialization_geometry | 0.1 | gamma | 60 | 60 | 84 |
| initialization_geometry | 0.1 | chi | 72 | 72 | 84 |
| initialization_geometry | 0.01 | gamma | 72 | 72 | 84 |
| initialization_geometry | 0.01 | chi | 72 | 72 | 84 |
| initialization_scale | 1 | gamma | 0 | 0 | 144 |
| initialization_scale | 1 | chi | 144 | 108 | 144 |
| initialization_scale | 0.5 | gamma | 0 | 0 | 144 |
| initialization_scale | 0.5 | chi | 144 | 108 | 144 |
| initialization_scale | 0.1 | gamma | 3 | 3 | 144 |
| initialization_scale | 0.1 | chi | 144 | 108 | 144 |
| initialization_scale | 0.01 | gamma | 49 | 49 | 144 |
| initialization_scale | 0.01 | chi | 144 | 128 | 144 |
| matched_conditioning | 1 | gamma | 0 | 0 | 48 |
| matched_conditioning | 1 | chi | 48 | 36 | 48 |
| matched_conditioning | 0.5 | gamma | 0 | 0 | 48 |
| matched_conditioning | 0.5 | chi | 48 | 36 | 48 |
| matched_conditioning | 0.1 | gamma | 0 | 0 | 48 |
| matched_conditioning | 0.1 | chi | 48 | 36 | 48 |
| matched_conditioning | 0.01 | gamma | 4 | 4 | 48 |
| matched_conditioning | 0.01 | chi | 48 | 41 | 48 |
| spectrum_width | 1 | gamma | 0 | 0 | 162 |
| spectrum_width | 1 | chi | 162 | 126 | 162 |
| spectrum_width | 0.5 | gamma | 0 | 0 | 162 |
| spectrum_width | 0.5 | chi | 162 | 126 | 162 |
| spectrum_width | 0.1 | gamma | 0 | 0 | 162 |
| spectrum_width | 0.1 | chi | 162 | 126 | 162 |
| spectrum_width | 0.01 | gamma | 43 | 43 | 162 |
| spectrum_width | 0.01 | chi | 162 | 156 | 162 |
| two_band_size | 1 | gamma | 0 | 0 | 12 |
| two_band_size | 1 | chi | 12 | 0 | 12 |
| two_band_size | 0.5 | gamma | 0 | 0 | 12 |
| two_band_size | 0.5 | chi | 12 | 0 | 12 |
| two_band_size | 0.1 | gamma | 0 | 0 | 12 |
| two_band_size | 0.1 | chi | 12 | 0 | 12 |
| two_band_size | 0.01 | gamma | 2 | 2 | 12 |
| two_band_size | 0.01 | chi | 12 | 4 | 12 |

Maximum positive discrepancies over eligible saved bands:
- gamma_max_violation: 1.4808e-07.
- gamma_max_monotonicity_increase: 2.12106e-31.
- chi_gradient_max_violation: 3.27596e-08.
- chi_loss_max_violation: 3.73416e-09.

These are numerical checks of the theoretical implications, not proofs. restart_certificates.csv includes every cutoff, its dimension, eligibility, terminal slack, informativeness, violations, and actual monotonicity checks. restart_case_metrics.csv records the widest eligible band's dimension and restart loss share, so a single eligible top direction is distinguishable from broad spectral coverage.

## Completion and tolerance refinement

completion_tolerances.csv follows decreasing relative total-loss tolerances from 1e-1 to 1e-10 and stores the saved crossing brackets. A flag identifies thresholds below the final positive saved loss, where the estimate interpolates toward the last zero state. Such apparent stabilization must not be interpreted as resolved continuous-time arrival. refinement_levels.csv reports all executed grid resolutions, extinction estimates, and their successive discrepancies. case_metrics.csv records both theoretical completion bounds and their ratios to the numerical estimate. Exact finite-time completion remains a theorem; the experiment checks consistency and numerical convergence.

Across recorded cases, upper-bound/numerical-completion ratios lie in [1.00022, 8.29395], and numerical-completion/lower-bound ratios lie in [1.00049, 1.3125]. A small ratio below one can reflect discretization; retain each case's refinement discrepancy when interpreting such comparisons.

## Artifacts and reproducibility

Six figure pairs (PNG/PDF) show envelopes and hitting tightness; spectral width/noise; initialization magnitude/geometry; restarted certificates; completion refinement; and dimension/conditioning/multiplicity. Each figure fixes the displayed non-seed settings. No representative seed substitutes for the cross-seed summaries. All machine-readable tables, aggregate.json, case_metrics.json, the executed analysis source and its SHA-256 are saved beside this report. The runner manifest and source snapshot identify the simulation code and frozen grid.
