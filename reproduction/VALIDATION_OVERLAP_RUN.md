# Exact Validation Endpoint Overlap Audit

`audit_validation_overlap.py` replays the author's data loaders in a separate
CPU process. It does not train, change author source, change formal splits,
write into recorded runs, or iterate shuffled DataLoaders.

```bash
python -u /workspace/audit_validation_overlap.py \
  --repo /workspace/PINN4SOH-main \
  --output /workspace/validation_overlap_XJTU_2C \
  --dataset XJTU --batch 2C --sample-sizes all 1 --seed 20261109
```

For an existing run, use its exact configuration, seed and battery path order:

```bash
python -u /workspace/audit_validation_overlap.py \
  --repo /workspace/PINN4SOH-main \
  --run /workspace/full_suite/tasks/path/to/run \
  --output /workspace/validation_overlap_recorded
```

Output must be new or empty. For explicit relocation, repeat
`--path-map OLD_PREFIX=NEW_PREFIX`. No path is silently guessed. Recorded
`config.json` and `splits.json` are required; `data_identity.json`, if present,
must match every replayed train/valid/test tensor hash. Recorded input and
source-inventory hashes are checked when available. Without recorded tensor
hashes, the report explicitly leaves the original-run tensor match as null.
The native local diagnostic is not labelled a verified replay of a cloud run.

## Identity And Trace

Each endpoint identity is the canonical battery CSV path plus the original
zero-based CSV data-row number. This is exactly the author's generated
`cycle index` before normalization. The audit captures it after the author's
own dropna/three-sigma filtering and before feature normalization. A retained
adjacent pair may skip original rows removed by preprocessing; its two original
indices are retained separately.

The actual `train_test_split` call is augmented with one integer identity
array. Sklearn applies its single permutation to that array and the four
original tensors; the extra identity outputs are removed before returning to
the loader. Every split tensor is checked exactly against its traced indices.
A separate replay with no instrumentation must produce identical tensor
hashes. No float-feature approximation is used to infer battery or cycle IDs.
Repeated endpoint labels are independently checked for bitwise equality.

The author loader makes two splits per battery-group load and fixes
`random_state=420` for both. It uses the second split for `train_2/valid_2`.
The run seed is replayed, but model initialization is not required for these
memberships: all global Python/NumPy/Torch RNG states are unchanged by loading.
The audit additionally consumes all three RNGs and initializes the author's
`Solution_u` before a separate replay, then requires unchanged identities and
tensor hashes. Shuffle order during training is outside this membership audit.

## Outputs And Interpretation

- `summary.json`: exact train/valid/test counts, overlap counts/fractions,
  per-battery statistics, RNG checks and replay verification status.
- `<case>/cycle_provenance.csv`: every selected dataset pair, its role,
  dataset position, battery ID and both original row indices.
- `<case>/trace.json`: retained battery rows, actual sklearn split indices,
  parameters, tensor hashes and label consistency checks.
- `artifact_sha256.json`: hashes of all diagnostic artifacts.

The primary quantity is the fraction of validation `x1/y1` cycle identities
also present in training `x1` or `x2`. The author PINN data loss supervises both
`y1` and `y2`; its overlapping validation labels have already participated in
training. The author MLP/CNN Trainer supervises `y1` only, so overlap solely
with dataset `x2/y2` does not imply the same direct label reuse for those
baselines. Both quantities are reported separately.

The diagnostic quantifies checkpoint-selection validation reuse. It does not
measure the resulting bias in validation/test error, demonstrate leakage
between separate held-out test batteries, or establish that the published
prediction results are invalid. The original whole-battery preprocessing is
retained; temporal forecasting and normalization scope are separate questions.

Local XJTU/2C results with seed 20261109 and the native local battery order:

| Case | Training Pairs | Validation Pairs | Valid x1 In Train x1 | Valid x1 In Train x2 |
| --- | ---: | ---: | ---: | ---: |
| All six non-test batteries | 1736 | 435 | 0 | 352 / 435 = 80.9195% |
| One training battery | 283 | 71 | 0 | 57 / 71 = 80.2817% |

All 732 held-out test `x1` identities have zero overlap with either training
endpoint in these two checks. The all-battery PINN training endpoint union has
2094 unique identities; the one-battery union has 341. These are observed
local diagnostic counts, not an extrapolation to every dataset or protocol.

Focused checks can be run with `python reproduction/test_audit_validation_overlap.py`.
They cover battery-aware identity, unchanged sklearn split/RNG behavior, and
an actual XJTU one-battery recorded-metadata/tensor-hash replay fixture.

The separate `outputs/full_reproduction/PDE_identifiability_analysis.md`
contains the structural analysis of G's direct `u_t` input, with an explicit
bounded-range sine-network construction and qualifications for training dropout.

## All 43 Configurations

```bash
python -u /workspace/audit_validation_suite.py \
  --repo /workspace/PINN4SOH-main \
  --evidence /workspace/pinn4soh_final_evidence.json \
  --output /workspace/validation_overlap_43
```

Place `audit_validation_suite.py` and `audit_validation_overlap.py` together.
The evidence package must contain its verified embedded `source_inventory.json`
and common manifest. This entry point audits all 11 full dataset/batch groups,
XJTU's six batches and HUST for n=1..4 (28 cases), and MIT plus TJU's third
batch/index 2 for n=1,2 (4 cases): 43 configurations in total. It uses one
membership computation per case because sklearn fixes the split at 420.

The suite reconstructs `os.listdir` results from the cloud inventory's recorded
directory order, verifies every directory's actual members, rejects ambiguous
mixed file/directory order, and verifies each loaded CSV and relevant author
source against the cloud hashes. Paths are explicitly mapped by their
repository-relative names onto `--repo`. All 11 full case sample counts must
match the previous cloud manifest. This permits faithful TJU batch/index
selection without relying on local Windows directory order.

The original 110-run evidence package includes source hashes and directory
order but lacks per-run tensor hashes and split JSON content. The report calls
this a recorded directory-order replay and leaves original tensor verification
unclaimed. For a current formal run whose data identity file exists, use the
single-run `--run` command above to require those tensor hashes as well.

Each case is checked against the uninstrumented loader. Each full group also
gets the independent global-RNG/model-initialization perturbation check. The
suite writes all 43 provenance CSVs and trace files, plus a combined summary
JSON/CSV and artifact hashes. `--group XJTU/2C` can select a small diagnostic
subset; the reported case count then states that smaller scope.
