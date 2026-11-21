# MIT Derived-Mirror Audit

The released PINN4SOH MIT CSVs contain two files with mixed physical barcodes. All 85,577 released capacity rows can be mapped to exact entries in the SHA-verified third-party cycle index. Prespecified within-cycle signatures independently support the source identities of both mixed tails. This evidence does not establish that every one of the 16 feature definitions has been exactly reconstructed, and it does not quantify a performance bias.

## Source And Scope

- Derived mirror: [Zenodo record 23984188](https://zenodo.org/records/23984188), DOI `10.5281/zenodo.23984188`, published 2026-11-05, CC BY 4.0.
- Extraction source: [deep-analysis-of-MIT-TOYOTA-dataset](https://github.com/tayebelleuch-ops/deep-analysis-of-MIT-TOYOTA-dataset), observed commit `2d2fae78c499dbc1abcec8ecefe1d49339f69998`.
- The 72,846,386-byte `within_cycle_index.csv` has published SHA-256 `93176963bbaaaea60a46591a3e7d1e5c3f1a04a250107eec1b754a3311b207dc`, verified locally.
- The index contains 140 original struct entries, 135 physical barcodes, and 114,738 cycle entries. Original struct index and source-cycle index are zero-based. HDF5 rows are sorted by physical cell/channel and concatenate continuations; channel number and HDF5 row must not be substituted for the original struct index.
- The original MAT files have not been obtained through a verified current direct download route. The derived HDF5 mirror narrows I/V/Q to float32 while retaining t as float64. It is explicitly an alternative derived source.

The downloaded extraction source enumerates structs with `for i in range(n)`, records `source_struct_index=rec.index`, records within-struct cycle position as `source_cycle_index=j`, and links the native HDF5 row using `h5_row=r0+j`. The adapter verifies HDF5 barcode/cycle metadata against each selected index row before reading its native arrays.

## Released CSV Mapping

There are 125 released MIT CSVs: 46 from May 2017, 43 from June 2017, and 36 from April 2018. Their filename `battery-N` corresponds to original `source_struct_index=N-1`. In 123 files every capacity row follows one physical barcode in order, including correct continuations where present. Two files contain mixed physical identities. Across all 125 files, 85,577 of 85,577 rows have a mapped capacity source with no unresolved rows.

Of the 245 cross-identity tail rows, 244 have a unique global exact capacity match. One repeated capacity at zero-based CSV5 row 1,236 is resolved by the same-segment source indices of its immediately surrounding mapped rows. Its match type is explicitly `exact_capacity_and_neighbor_constrained_segment`; it is not presented as a globally unique capacity match.

All CSV row ranges and raw source cycle indices below are zero-based. Gaps inside a source cycle range reflect the released filtering.

| Released May CSV | CSV Rows | Rows | Source Batch | Source Struct | Channel | Physical Barcode | Raw Source Cycle Range |
|---|---:|---:|---|---:|---:|---|---:|
| battery-4 | 0-1149 | 1150 | 2017-05-12 | 3 | 5 | EL150800464977 | 3-1186 |
| battery-5 | 0-1190 | 1191 | 2017-05-12 | 4 | 6 | EL150800464865 | 6-1225 |
| battery-5 | 1191-1357 | 167 | 2017-06-30 | 15 | 5 | EL150800464977 | 2-177 |
| battery-6 | 0-1009 | 1010 | 2017-05-12 | 5 | 7 | EL150800464883 | 5-1024 |
| battery-6 | 1010-1087 | 78 | 2017-06-30 | 16 | 6 | EL150800464865 | 355-458 |

Thus the continuation of physical channel 5 appears in CSV5, while its earlier portion is in CSV4. A continuation of physical channel 6 appears in CSV6, while its earlier portion is in CSV5. The present evidence describes the released data; it does not establish the exact cause in the upstream processing history.

The verified old110 MIT split assigns CSV5 to test and CSV4/CSV6 to training/validation. Consequently barcode `EL150800464977` spans training CSV4 and test CSV5, and barcode `EL150800464865` spans test CSV5 and training CSV6. Distinct filenames alone therefore do not guarantee physical-cell independence in the original released split.

This is shared physical identity across different parts of a cell's history. It is not evidence that an identical raw cycle or an identical feature vector was duplicated across training and test.

Joining the actual retained-row trace to the physical identity map reveals two cross-physical adjacent pairs: CSV6 original rows1009-1010 are assigned to validation (position10072), and CSV5 rows1190-1191 to test (position7006). There are zero such pairs in training under the verified fixed-420 membership. Author `Valid` and `Test` consume only x1/y1. Consequently these boundaries must not be described as proof that the training physical penalty directly constrains different cells.

Actual endpoint provenance also shows that 1,256 of 14,170 original MIT test x1 predictions (8.8638%) belong to physical identities seen in training. Barcode `EL150800464865` has 1,191 test x1 and 26 train x1 endpoints; `EL150800464977` has 65 test x1 and 867 train x1 endpoints. Counting both PINN training endpoints or only baseline x1 gives the same test identity-exposure total. These are different histories of shared physical cells, not repeated raw samples or an estimate of prediction bias. Original MIT macro metrics use 23 released CSV units; mixed CSV5 contains two physical identities, so the file macro should not be described unconditionally as an independent physical-cell macro.

## Within-Cycle Signatures

The signature sample was prespecified as the first three rows from each of May CSV4/5/6 plus the first three rows in each differently sourced tail: 15 raw cycles, with both CC-voltage and CV-current signatures for each. No cycle was selected by feature residual. Author `MITBatteryClass.py` stage-selection methods are called unchanged through a raw-array adapter.

The two critical sampled tails are CSV5 rows 1191-1193, which match June struct15/channel5/barcode `EL150800464977`, raw cycles2/3/4, and CSV6 rows1010-1012, which match June struct16/channel6/barcode `EL150800464865`, raw cycles355/356/357. HDF5 barcode and cycle metadata agree for every sample. The numerical signatures are consistent with the source mapping and the documented float32 narrowing.

| Signature | CC Voltage Max Absolute Difference | CV Current Max Absolute Difference |
|---|---:|---:|
| Mean | 9.878e-9 | 1.128e-9 |
| Population standard deviation | 1.510e-8 | 2.648e-9 |
| SciPy biased excess kurtosis | 2.127e-6 | 1.847e-7 |
| SciPy biased skewness | 7.016e-7 | 5.995e-8 |
| Min-max probability entropy candidate | 1.224e-6 | 3.011e-8 |
| Window duration, seconds | 2.632e-11 | 4.530e-11 |
| Charge-capacity difference, Ah | 3.075e-8 | 9.666e-8 |

All slope candidates remain hypotheses. No slope definition is chosen by residual minimization. Mirror documentation calls native current I a C-rate, while the author reader labels the same numeric field amperes. Numeric current is preserved without rescaling; these signatures do not settle its physical-unit interpretation.

The 15-cycle sample was rerun locally using complete May and June HDF5 files after both matched their published full-file SHA-256. The signature JSON records `verified_full_sha256` for both sources; the earlier bounded provisional range sample is superseded. May is 797,659,714 bytes, SHA-256 `8a789f4a83409ad0c1d6c214c836f2de94b63ef2e73a50722fe057c3d8a89d77`; June is 581,592,476 bytes, SHA-256 `5f3f752d9d9a0276442fbb853941d9b212e7fffd152995b207ae6fa4a4503fae`. These two files cover every critical mixed-tail sample. Full-mirror retrieval/checksum status is separately machine-readable in `MIT_mirror_mapping_audit.json` under `within_cycle_h5_files`.

All three derived HDF5 files are now complete and published-SHA-verified (2,202,644,831 bytes total); April has 823,392,641 bytes and SHA-256 `bdd195323719986b3d77d4bd610c26e5e4e7c8d8ddde558b03b4696a2bcc6107`. A separate fixed expansion to the first three rows of every released CSV plus the first three rows of both other-source tails computed all 381 cycles/762 curves, with every barcode/cycle assertion passing and no residual-based exclusions. Results are in `signatures_all125/`; this expansion does not replace the original 15-cycle critical sample or change the frozen task plan. Voltage mean, population SD, kurtosis, skewness, entropy, duration, and Q-difference maximum absolute errors are respectively 1.007e-8, 1.510e-8, 3.645e-6, 8.445e-7, 2.214e-6, 2.632e-11 seconds, and 5.352e-8 Ah; current maxima are 2.797e-9, 3.018e-9, 2.981e-7, 8.317e-8, 5.603e-8, 4.530e-11 seconds, and 9.666e-8 Ah.

## Separate Sensitivity Plan

The independent plan under `../../MIT_physical_purity_sensitivity/` contains 40 tasks and has not been launched by the plan generator. It excludes entire May CSV5/CSV6 and preserves every other path's original recorded order. Remaining training/validation and test physical identity sets are disjoint over all released rows: 101 and 22 identities respectively. It retains the original author 20% sample validation and original normalization. Actual unchanged local loader counts are 53,223/13,306/12,914 train/validation/test pairs, with tensor hashes in the plan evidence.

Removing a mixed test file changes the evaluated population. Any resulting metric difference must be reported as a sensitivity comparison and cannot independently estimate how much the original overlap biased accuracy. The original formal runs remain the reference experiment, and the separate adjacent-endpoint validation issue is unaffected by these whole-file exclusions.

## Reusable Evidence

- `MIT_mirror_mapping_audit.json`: complete struct candidates, all-file identity counts, unresolved-row list, physical role overlaps, and source checksums.
- `PINN_CSV_to_mirror_struct.csv`: one row per released CSV.
- `PINN_CSV_to_mirror_cycles.csv`: every released row's exact source batch/struct/cycle/HDF5 row and match type.
- `signatures/MIT_mirror_feature_signatures.{json,csv}`: the prespecified 15-cycle critical sample, all candidate slopes, released values, and residuals.
- `reproduction/audit_mit_mirror.py`: read-only mapping and bounded resumable HDF5 retrieval.
- `reproduction/extract_mit_mirror_features.py`: author stage adapter with identity assertions and source checksum validation.
- `reproduction/make_mit_physical_purity_plan.py`: hash-checked separate plan generation and optional actual-loader verification; no training invocation.
