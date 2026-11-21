# PINN4SOH Feature Reconstruction Audit

This report separates descriptive replication from released feature CSVs from reconstruction using raw charging curves. No training source or released CSV is changed. The original-raw candidate extraction and fixed-boundary reconciliation are complete for all 262 XJTU/TJU/HUST cells, with 262 primary and 132 boundary candidate checksums independently rechecked in `RAW_FEATURE_INTEGRITY.json`. Every finite value of 14 non-slope features is reconstructed under the explicitly declared conventions; both slopes remain unresolved. Fig1b/S4 raw-curve illustrations and provenance are complete in `raw_curve_figures/`.

## Completed Original-Raw Candidate Extraction

| Dataset | Cells | Raw Physical Cycles | Candidate Records | Candidate CSV Bytes |
|---|---:|---:|---:|---:|
| XJTU | 55 | 27642 | 236183 | 109129935 |
| TJU | 130 | 59681 | 537084 | 258239188 |
| HUST | 77 | 146122 | 876732 | 411627242 |
| Total | 262 | 233445 | 1649999 | 778996365 |

All required cell identities are unique, none are missing or unexpected, and all saved SHA256 values match the actual candidate files. All five original archives passed their official MD5/SHA256 checks. Candidate records include separate curve-window definitions and are not additional independent cycles or training examples. This verifies the completed 262-cell extraction; it does not establish the original MIT MAT chain or resolve either slope.

HUST scheduling was changed from one worker to three independent cells using the unchanged frozen `process()` implementation. The pool completed its remaining 68 cells in 2112.48 wall seconds, with no failures. Already complete cells were skipped after implementation-hash checks; a stopped next-cell partial was preserved and recomputed. Boundary calculations later became serial with MAT loading gated on available RAM, to protect the active cloud browser. Calculation scripts remain frozen; scheduling and interruption evidence are stored separately. Summed per-cell times are not a whole-job wall-clock benchmark.

## Completed 262-Cell Fixed-Convention Reconciliation

The 132 fixed-boundary supplements (55 XJTU and 77 HUST) have completed without failure. Their real candidate-file SHA256 values, primary implementation hashes and the frozen supplement hash were independently verified along with all 262 primary cells. `RAW_FEATURE_INTEGRITY.json` reports 262 primary and 132 boundary files passed. `RAW_FEATURE_SUMMARY_STATUS.json` reports all 262 primary and fixed-convention cells summarized.

All 225128 released rows in these three datasets map uniquely and in increasing order to physical raw cycles: XJTU 23297, TJU 57465 and HUST 144366. The four current-distribution statistics used for this fingerprint are not independent validation of the mapping. The other features, window duration and accumulated charge provide additional checks.

For **each of the 14 non-slope features**, every finite published value matches its fixed raw candidate within absolute 1e-8, with 100% finite-target coverage in all 262 cells. XJTU voltage entropy contains 151 published non-finite values; its 23146 finite values all match and the 151 non-finite rows are counted separately. The largest non-slope residual is approximately 3.26e-9 seconds in TJU duration, consistent with decimal/float conversion. Statistical/charge values are generally identical or within a few 1e-15. This result uses the stated TJU window, public-reader HUST 3.595 V bound, and the explicitly exploratory XJTU 4.195 V bound fixed after the first pilot across all 55 cells. It is evidence for a fixed reconstruction convention, not proof of the author's unreported exporter implementation.

The fixed non-slope definitions are population SD, biased Fisher excess kurtosis, biased skewness, min-max-to-probability entropy, last-minus-first recorded charge capacity inAh, and last-minus-first physical time inseconds. Exact per-cell and per-feature values, finite coverage and residuals are in `raw_fixed_convention_residuals_per_cell.csv` and `raw_fixed_convention_feature_summary.csv`.

The endpoint slope stated by Note 2 is **not** reconstructed across the released rows:

| Dataset | Released Rows | Voltage Matches Within 1e-8 | Voltage MAE (V/s) | Current Matches Within 1e-8 | Current MAE (A/s) |
|---|---:|---:|---:|---:|---:|
| XJTU | 23297 | 3 | 0.0000211221 | 0 | 0.000554812 |
| TJU | 57465 | 10 | 0.0000198539 | 0 | 0.000150519 |
| HUST | 144366 | 0 | 0.000513978 | 0 | 0.00166284 |

The physical-second endpoint values have **zero matches within 1e-12** in all six dataset/curve combinations. The additional six unit/axis candidates also fail to reconstruct a complete column: the most exact matches in this family are 61/23297 XJTU voltage values from nominal 0.2 V/duration, while all other candidate columns have at most one exact match. No candidate was fitted by multiplying a learned constant. All six frozen endpoint/regression/local-slope definitions across their declared windows likewise fail to match a whole dataset; their residuals are exported separately in `slope_all_frozen_candidates_summary.csv`. A lower MAE or occasional coincidence is not a recovered export formula. Both slopes therefore remain unresolved, and no training input has been replaced.

### Slope Scope and Missing Author Definitions

The explored rules form a declared finite family. They do not prove that no conceivable transform could reproduce the slopes. Endpoint, whole-window OLS, mean/median local difference quotients, first/last local quotients, physical seconds/minutes, sample index, fixed nominal boundary amplitudes, and normalized time have all been compared with aggregate residuals rather than selected per row. A separate original-curve pass has completed all 262 cells and expands this to 32 fixed definitions per curve: raw/min-max curve values × physical seconds/sample index/min-max physical time/min-max sample index × endpoint/OLS/mean difference quotient/mean numerical gradient. All 262 per-cell residual checksums and the common script hash pass in `SLOPE_ADDITIONAL_INTEGRITY.json`; none of the 192 dataset/curve/definition summaries reconstructs an entire slope column. Each definition has at most two exact matches within 1e-12 in any dataset/curve. Four physical-time numerical-gradient candidates omit 140 XJTU current rows with duplicate/nonincreasing timestamps; all other candidates cover every published finite slope. The positive-dt mean-quotient rule is separately retained, so duplicate timestamps are not silently assigned a derivative. Completion and residuals are recorded in `slope_additional_axes/`, `slope_additional_axes_per_cell.csv`, and `slope_additional_axes_summary.csv`.

The author still needs to supply the actual 16-feature exporter and the precise slope conventions: which axis and unit is used, which samples/CC-CV boundary are included and whether endpoints are observed or nominal; whether slope is local or global, which segment length/index is used, and whether smoothing, resampling, duplicate-time handling, or normalization is performed before fitting. These definitions cannot be inferred as established facts from a low residual alone. The public data reader is not the missing exporter. The reproduction preserves released training inputs and reports both slopes as unresolved until a shared fixed rule or author code is verified.

## Completed Descriptive Checks

- Exactly 387 released battery CSVs contain 310705 rows, matching the paper's stated population and sample count.
- 310554 rows remain after replacing infinity with missing and dropping missing rows. The author's union-of-3-sigma exclusion leaves 303952 rows and 303565 adjacent-cycle pairs. These are distinct counts, not interchangeable sample definitions.
- Group counts match: XJTU 55 in six batches (8/15/8/8/8/8), TJU 130 (66/55/9), HUST 77, MIT 125.
- `processed_figures/figure2_released_capacity.{png,pdf}` redraws all 55 XJTU curves with the author plotter's first-row omission. Its horizontal coordinate is the released row index, not recovered physical cycle IDs. For partially discharged batches, interpolated capacity must not be described as independent full-discharge measurements every cycle.
- `figure3a_released_features.{png,pdf}` displays 16-features of the eight XJTU batch-1 cells after the author cleaning and per-cell min-max normalization.
- `figure3b_author_cleaned_per_battery_minmax_pooled.{png,pdf}` reproduces all **176/176** printed Fig.3 b Pearson coefficients to two decimals. The sequence is author cleaning, separate min-max scaling within each battery, then pooling cycles within each of eleven groups. Alternative raw/cleaned unscaled matrices and per-battery coefficient means are exported separately and do not reproduce the same table.
- All 176 printed reference values were manually transcribed from the original paper PDF p 5 and independently checked visually by a second agent without consulting the computed coefficients. This verifies the displayed rounded values, not unrounded unpublished coefficients.
- All six rendered figures passed text-overlap and figure-boundary checks; Fig.2, Fig.3 a and the matching Fig.3 b were visually inspected.

The scalar correlations are descriptive repeated-cycle associations. They do not demonstrate causal chemical effects or statistically independent cycle observations.

## Official Source And Storage Inventory

Preprocessing source is frozen at `wang-fujin/Battery-dataset-preprocessing-code-library` commit `e986a74054fb64c0f5134f2cc97f4643cd7699d2` (2026-07-28). Every downloaded source Git blob was verified against its official SHA1, and SHA256 provenance is recorded in `preprocessing_provenance.json`.

The library provides dataset readers, charge/CC/CV slicing, and capacity accessors. It has **no implementation of the complete 16-feature export**, kurtosis, skewness, or entropy. It therefore does not, on its own, resolve the formula-to-CSV differences.

| Official Raw Archive | Compressed Bytes | Uncompressed Bytes | Battery Records |
|---|---:|---:|---:|
| XJTU `Battery Dataset.zip` | 2438769934 | 2903003649 | 55 battery MATs plus a temperature-compensation MAT |
| TJU `Dataset_1_NCA_battery.zip` | 579629324 | 1923069298 | 66 CSVs |
| TJU `Dataset_2_NCM_battery.zip` | 737047539 | 2405394399 | 55 CSVs |
| TJU `Dataset_3_NCM_NCA_battery.zip` | 356066478 | 1280816591 | 9 CSVs |
| HUST `our_data.zip` | 1188136932 | 4245730747 | 77 PKLs |
| Total For These 262 Cells | 5299650207 | 12758014684 | 262 |

The complete ZIP directories required only about 31 KB of HTTP Range data. Archive sizes plus full expansion total 18.06 GB, so these three datasets fit a 200 GB volume. The candidate extractor instead opens one member at a time, retains compressed archives, and does not expand all raw files at once. A 25 GB local raw-data budget is enforced by the downloader. It records official MD5/SHA256 verification and a separate SHA256 before extraction; each battery then has its own result checksum and completion state.

The original MIT entry points currently return HTML or redirect to the application for the expected MAT names. Their 5095-byteHTML response must not be mistaken for a tiny raw data file. `MIT_raw_headers.json` records that failure. The separate third-party within-cycle mirror audit has now SHA256-verified all 3 H 5 files and established 85577/85577 published cycle mappings, with 381 sampled cycles/762 curves computed across 125 CSVs. Those derived float 32 files must not be called byte-identical to the original MATs, and the sample checks are not full 125-cell extraction of 16-features. The 387-cell original-MAT chain remains unestablished. See `MIT_mirror_audit/` for that independently scoped evidence.

`required_raw_files.csv` maps every released CSV to its required original archive/member. All 262 XJTU/TJU/HUSTmember paths match the official ZIP directory. MIT's 125 CSVphysical-cell/struct-index mappings are now documented in the separate mirror audit, rather than guessed; original-MATbyte identity is still unverified.

Official source links:

- XJTU raw data: <https://zenodo.org/records/10963339>
- TJU raw data: <https://zenodo.org/records/6405084>
- HUST raw data: <https://data.mendeley.com/datasets/nsc7hnsg4s/2>
- MIT original landing page: <https://data.matr.io/1/projects/5c48dd2bc625d700019f3204>
- Preprocessing source: <https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library>

The XJTU reader header incorrectly points to Zenodo `11046967`, which is a 107295050-byte PINN4SOH code archive. The preprocessing README's `10963339` link is the 55-cell raw-data archive.

## Raw Sample Findings

One official XJTU sample, `Batch-1/2C_battery-1.mat`, was fetched by bounded ZIP-member ranges before the complete archive download. ZIP CRC and member length were checked; its SHA256 is recorded in `raw_sample_provenance.json`. The full XJTU archive was subsequently downloaded and passed the official MD5.

The raw sample has 390 physical cycles; the released CSV has 375 rows. It is not a single contiguous truncation. `raw_sample_cycle_alignment.csv` provides a unique, increasing mapping for every published row using capacity and four current-statistic signatures. The first retained raw cycle is 10 and the second is 12. Because four statistics contribute to the alignment, agreement of those four is not an independent test of the alignment.

Across all 375 aligned rows:

| Feature | Explicit Candidate | Result |
|---|---|---|
| Current mean | Mean of charge-stage current in [0.1,0.5] A | 375/375 match |
| Current standard deviation | Population SD, `numpy.std(ddof=0)` | 375/375 match |
| Current kurtosis | Fisher excess kurtosis, `scipy.stats.kurtosis(bias=True)` | 375/375 match |
| Current skewness | `scipy.stats.skew(bias=True)` | 375/375 match |
| Current entropy | Min-max the window values, normalize their sum to probability, then `-sum(p log p)` | 375/375 match |
| CV accumulated charge | Last minus first recorded charge capacity, in Ah | 375/375 match |
| CV duration | Last minus first time, converted from minutes to seconds | 375/375 match |
| Voltage mean/SD/kurtosis/skewness/CC charge/CC duration | CC window [4.0,4.195] V with I>=3.9 A, same statistical conventions | 375/375 match for each of these six |
| Voltage entropy | Same min-max-to-probability convention and [4.0,4.195] V window | All 362 finite values match; remaining 13 published entropy values are non-finite |
| Voltage and current slope | Paper endpoint slope, full-window least-squares slope, mean/median/first/last local slopes | None establishes exact reconstruction |

The 4.195 V upper threshold is an explicitly reported **exploratory diagnostic hypothesis**, examined after the pilot residuals suggested a CC/CV boundary difference; it is absent from Supplementary Note 2. It must not be silently substituted for the paper's 4.2 V or called the proven unpublished extraction algorithm. It is fixed across all later XJTU cells rather than adjusted by row or battery. The original fixed 4.2 V, public reader 4.199 V, 4.19 V and actual-max-voltage windows are retained with all their residuals. Agreement on the pilot is calibration evidence; the remaining cells provide a separate check.

A first TJU cell (`Dataset_2_NCM_battery/CY25-05_1-#1.csv`) was independently parsed from the checksum-verified official archive: 193 raw cycles map to all 192 published rows. Fourteen of sixteen features match over every row using the Note 2 voltage window [4.0,4.2] V, charge-stage [0.1,0.5] A current window, population SD, biased Fisher kurtosis/skewness, min-max probability entropy, capacity differences in Ah, and durations in seconds. Both slopes remain unresolved.

A complete HUST member (`2-5.pkl`, 1386 raw cycles) was retrieved with a bounded official ZIP range, CRC checked, and parsed through a restricted allowlist of data constructors. All 1360 published rows have unique ordered current-statistic matches. The seven current-side features and capacity match; additionally, the fixed public-reader voltage boundary [3.4,3.595] V matches all seven voltage-side features over all 1360 rows. The 3.59/3.599/3.6 alternatives are retained. This is again 14/16 exact feature reconstruction on one cell, with both slopes unresolved. At pilot time the member-only audit did not establish the full archive checksum; the subsequent complete HUST archive passed its official SHA256 before all 77 cells were extracted.

The public XJTU reader uses `current>=3.9A` as a CC gate. Supplementary Note 1 gives 1 C charging for some later batches, so this gate cannot be assumed valid for every XJTU cell. `supplement_voltage_boundaries.py` independently audits all-charge [4.0,4.195] V windows for all 55 XJTU cells and [3.4,3.595] V for all 77 HUST cells, without changing the running primary extractor or replacing any original candidate. Its per-cell outputs are in `boundary_extraction/` and retain the primary current signatures for alignment.

The primary candidate pipeline has completed all 262 cells. A 193-cycle TJU pilot took 16 seconds before caching duplicate windows, and a subsequent 203-cycle cell took about 8 seconds; a 393-cycle XJTU cell took 45 seconds while another source processed concurrently. These are historical pilot measurements, not whole-dataset timing claims. Completed per-cell manifests and the HUST pool wall time above provide the measured scopes. Fixed-boundary completion, mapping coverage, finite-target coverage, and exact feature residuals remain separately reported.

## Formula And Normalization Differences

Supplementary Note 2 states sample SD (`n-1`) and a nonnegative kurtosis formula with no subtraction of three. The released CSV contains 164862 negative voltage-kurtosis values and 286509 negative current-kurtosis values. The exact sample matches above support excess-kurtosis and population-SD conventions for those audited raw cells; a generic formula change must not be applied to all cells without checking the remaining residuals.

Note 2 calls `p_i` a normalized curve value without specifying whether normalization means probability normalization or min-max followed by probability normalization. Both candidates, plus an unscaled min-max entropy sum, are exported. The matching candidate is reported with its scope; a missing definition is not filled in as if it were supplied by the paper.

The raw capacity readers also differ: XJTU reads summary discharge capacity and uses cubic interpolation of labelled capacity tests (linear fallback); TJU uses maximum discharge capacity per cycle in mAh; HUST's public reader uses maximum `Capacity (mAh)` in a cycle. The model inputs use Ah. SOH divisors are XJTU 2.0 Ah, MIT/HUST 1.1 Ah, TJU batches 1/2 3.5 Ah and batch 3 2.5 Ah. Raw adapters record both directly measured and interpolated capacity when available.

There is a source inconsistency in Supplementary FigS4's caption: it groups batches 3-6 as incomplete-discharge protocols, whereas Note 1.3 explicitly describes batch 3 as discharged to 2.5 V. The official `R2.5_battery-1.mat` contains 592 physical cycles but only its initial cycle is labelled `[test capacity]`; the public reader's interpolation therefore returns non-finite values for this cell. The original S4 third row also lacks a red interpolation line. The raw-curve reconstruction displays batch 3's per-cycle discharge capacities without inventing a periodic-test interpolation, and separates measured capacity tests from interpolated trajectories for batches 4-6. Neither incomplete-discharge capacity nor interpolation is described as an independent full-capacity measurement each cycle.

`plot_raw_windows.py` fixes the Fig1b illustration to XJTU batch 1 battery 1, physical cycle 10, before rendering. The plot uses the stated 4.0-4.2 V paper window, plus the public-reader CC gate, rather than silently substituting the diagnostic 4.195 V feature-audit boundary. Its source CSV distinguishes both masks. S4 fixes the first battery in each of the six batches; all cycles are shown with explicit uniform within-cycle display decimation. These are source-grounded curve illustrations, not a claim of pixel identity with an unreported original example cell.

`audit_slope_axes.py` additionally tests observed endpoint differences per physical second, minute, sample index, and normalized time, as well as nominal paper/fixed-window boundary differences per second. No multiplier is fitted to the released slopes. These candidates are reported alongside all six frozen local/regression/endpoint candidates, with exact and 1e-8 residual counts. Partial summaries remain labelled by completed cells and do not establish reconstruction of either slope across a dataset.

The author's model loader inserts cycle index, removes non-finite rows, drops the union of 3-sigma outlier rows across **all columns including capacity**, divides capacity by nominal Ah, and scales all input features including cycle index separately within each entire battery. Validation/test batteries therefore contribute their own full-life extrema to normalization; this protocol is not training-only normalization. It is preserved for reproduction and clearly described, not silently replaced. A prospective deployment sensitivity would need an explicitly separate normalization protocol.

## Reproduction Commands

```powershell
python -X utf8 reproduction/audit_raw_data_inventory.py --direct
python -X utf8 reproduction/audit_xjtu_feature_sample.py --direct
python -X utf8 reproduction/audit_processed_features.py
python -X utf8 reproduction/download_official_raw.py --direct
python -X utf8 reproduction/extract_raw_feature_candidates.py --datasets XJTU TJU HUST --watch
python -X utf8 reproduction/summarize_raw_feature_residuals.py
python -X utf8 reproduction/verify_raw_feature_integrity.py
python -X utf8 reproduction/summarize_raw_feature_audit.py
python -X utf8 reproduction/audit_slope_axes.py
python -X utf8 reproduction/plot_raw_windows.py
```

The downloader and source-specific extraction workers may run independently. Raw extraction completion requires per-cell `status=complete`, successful archive verification, saved candidate checksums, and explicit residual summaries. Successful feature extraction alone does not reproduce training convergence or the paper's predictive results.
