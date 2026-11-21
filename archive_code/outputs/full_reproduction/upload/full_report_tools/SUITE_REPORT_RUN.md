# Independent full-paper reporting

These tools read completed tasks and never train or alter the author repository.
Keep them in a new tools directory; do not change a running coordinator's frozen
runner directory.

## Required Files

- `build_suite_report.py`
- `export_suite_evidence.py`
- `suite_evidence_rules.json`
- `paper_protocol.json` from `outputs/full_reproduction`
- `source_data/paper_table_s7.json` next to that protocol file for Table S7 comparisons
- Both frozen stage plan JSON files

Python plus NumPy is sufficient. The training environment already has NumPy.
All statistical calculations are separate from the author's analysis scripts.

## Report A Partial Or Complete Suite

```bash
python /workspace/full_report_tools/build_suite_report.py \
  --suite /workspace/pinn4soh_full_paper_stage1 \
  --plan /workspace/full_report_tools/stage1_plan.json \
  --plan /workspace/full_report_tools/stage2_reproduced_transfer_plan.json \
  --protocol /workspace/full_report_tools/paper_protocol.json \
  --output /workspace/pinn4soh_full_report
```

Once the second suite directory exists, also pass
`--suite /workspace/pinn4soh_full_paper_stage2` (use its actual configured path).
Both plans may be supplied earlier: unlaunched tasks remain explicitly pending.
The report does not create a claimed result for a missing, smoke, failed or
hash-inconsistent task. Completed task hashes, checkpoint reload evidence,
configuration, source identity, exact battery boundaries and stored metrics
must pass independent checks before inclusion.

Primary conclusions use exact-battery equal-weight macro metrics with measured
SOH in the MAPE denominator. Every output also carries pooled metrics and
predicted-denominator MAPE separately. Means, population SD (ddof=0) and sample
SD (ddof=1) are all labeled. Configuration groups separate loss, validation,
parameters, code, data and numerical environments. Shared MLP/CNN baselines are
explicitly matched to each PINN lane for descriptive comparisons.

Additional monotonicity diagnostics count prediction rises, total positive
prediction increments, measured rises and opposite signs of increments within
each held-out battery. They refer to adjacent retained test points, which need
not be consecutive original cycles. The pooled fractions weight by the number
of within-battery adjacent pairs; pooled positive-increment totals sum across
batteries. Macro diagnostics average the per-battery values. They are extra
diagnostics, not replacements for the paper's MAE/MAPE/RMSE.

`suite_evidence_rules.json` covers 25 main/supplement figure and table entries
and gives an explicit evidence/decision rule for every one of the 21 audited
claims. Available numbers do not automatically settle raw-feature, physical
identifiability, causality, timing, leakage or deployment claims. The report
preserves those limits and provides comparison effects for manual scope review.

## Export Browser-Downloadable Evidence

```bash
python /workspace/full_report_tools/export_suite_evidence.py \
  --report /workspace/pinn4soh_full_report \
  --output /workspace/pinn4soh_full_evidence
```

Start with `suite_evidence_index.json`. It lists every standard JSON/CSV file,
SHA256, byte count and table schema. Each JSON file is capped at 18 MiB by
default; large collections are split without dropping tasks. CSV files retain
the complete report tables. The JSON tables use positional rows and explicit
column lists to reduce repeated metadata.

Task evidence includes exact parsed manifest/config/splits/metrics documents,
original file hashes and paths, full logged training/validation/test histories,
validation-selected best epochs, checkpoint reload evidence and all known
artifact hashes. Content-addressed JSON nodes deduplicate shared structures;
`export_suite_evidence.resolve(value, nodes)` restores the original parsed JSON
value. The canonical-value hash can verify the restored value.

Display curves use only preselected repeat1 for all available regular groups
and both PINN losses plus MLP/CNN. Small-sample and transfer curves also use
repeat1, including paired source-only predictions. They are never selected by
test error. Display arrays uniformly sample retained-point indices including
both endpoints, up to 256 points per regular test battery and 128 for other
families. The index and records label this sampling. All numerical metrics and
monotonicity diagnostics still use the complete arrays, which remain on the
cloud with their hashes along with the full checkpoints and raw text logs.

## Verification

```powershell
python -X utf8 -m unittest discover -s reproduction -p 'test_suite*.py' -v
```

The tests use synthetic arrays and evidence only. They cover unequal battery
lengths, MAPE conventions, boundary errors, both runner formats, source-only
metrics, smoke exclusion, duplicate seeds, protocol separation, hash and metric
corruption, fixed-G evidence, target-battery matching, monotonicity diagnostics,
full report/export round trips, best-epoch checks and JSON size partitioning.
