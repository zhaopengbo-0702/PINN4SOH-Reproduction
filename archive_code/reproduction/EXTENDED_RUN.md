# Extended PINN4SOH Experiments

`run_extended.py` runs one task in one independent output directory. It does not edit the author repository or the earlier 110-run results. The suite coordinator owns scheduling and aggregate progress; this runner owns only its task directory.

## Interface

```bash
python -u /workspace/run_extended.py \
  --task-json /workspace/task.json \
  --repo /workspace/PINN4SOH-main \
  --output /workspace/full_suite/tasks/full/XJTU/2C/CNN/01 \
  --helpers /workspace/run_reproduction.py --resume
```

The helper file supplies the existing CSV loader instrumentation and metric utilities. `--smoke` explicitly limits a task to two epochs and clamps warmup to one; smoke output must use a different directory from formal output. `--allow-cpu` is only for an intentional CPU check. Existing `CUDA_VISIBLE_DEVICES` is preserved, and logical CUDA device 0 maps to the assigned visible GPU.

Minimal task:

```json
{
  "task_id": "full/XJTU/2C/CNN/01",
  "family": "full",
  "dataset": "XJTU",
  "batch": "2C",
  "model": "CNN",
  "repeat": 1,
  "seed": 20261109,
  "epochs": 200,
  "threads": 2,
  "protocol_id": "author_current",
  "validation_protocol": "author_sample_split",
  "reference_run": "/workspace/pinn4soh_main_20261109/XJTU/2C/Experiment01",
  "protocol_evidence": ["main_comparision.py Trainer and Model/Compare_Models.py"]
}
```

`model` accepts `PINN`, `MLP`, `CNN`. `family` accepts `full`, `small_sample`, `ablation`. A small-sample task adds `sample_size` in 1..4. The training paths are truncated in the recorded author/reference order, and the regular test paths remain unchanged. `reference_run` is recommended for pairing with the existing run's actual paths. `split_json` can supply a standalone recorded split instead.

## Protocols

`protocol_id=author_current` uses the current source defaults, including its explicit PINN small-sample settings. `protocol_id=paper_note3` applies Supplementary Note 3's general settings with the separately selected `loss_variant`:

| Dataset | PINN alpha | PINN beta | Batch Size |
|---|---:|---:|---:|
| XJTU | 0.7 | 20 | 256 |
| MIT | 1 | 50 | 512 |
| TJU | 1 | 50 | 512 |
| HUST | 0.5 | 80 | 512 |

The paper gives these general settings and does not separately identify the small-sample hyperparameters. The current source differs: XJTU PINN small-sample uses batch 128, alpha 0.5, beta 10; the other PINN small-sample entries use batch 128. The comparison entry uses batch 512 even for XJTU. These protocols must not be silently pooled.

PINN tasks accept three physical-loss variants. The default remains `author_current`; CNN and MLP accept only that default because they have no physical loss.

| `loss_variant` | Physical Penalty | Historical Model Commit |
|---|---|---|
| `published_label_monotonicity` | `relu(y2-y1).sum()` | `d26bc1f40dc0cdf34405cd5e599a6f6af09742e9` (2024-04-23) |
| `paper_prediction_monotonicity` | `relu(u2-u1).sum()` | `1e94ec8ed9af35134c8afbd6c576b80f3228b7a0` (2024-07-11) |
| `author_current` | `relu((u2-u1)*(y1-y2)).sum()` | `cc3cc5053caf38f04e0665f7f88cb109144d035e` (current reference) |

The publication-era expression uses labels only, so its derivative with respect to every model parameter is zero even when its numerical value is nonzero. The prediction expression matches the paper's monotonicity equation. These are controlled reconstruction lanes, not interchangeable results. Each PINN task saves first-batch gradient-connectivity and penalty diagnostics. The task config records exact historical Model blob hashes, actual weights, Note 3 weights, and publication-era defaults. Historical MIT used `early_stop=10`; the current MIT entry uses 20. A historical MIT lane therefore needs an explicit `overrides: {"early_stop": 10}` with protocol evidence. Selecting Note 3 weights alone does not select historical patience or prove exact recovery of the paper's numbers.

Non-current variants reuse the audited loss helper in `run_transfer.py`; its SHA256 becomes an additional implementation dependency. The author repository is unchanged. The historical recursive Git tree also verifies all 387 current CSVs against their publication-era Git blob identities: all match, with no missing or extra CSVs. The evidence is saved in `outputs/full_reproduction/data_csv_history_identity.json`.

The author comparison Trainer is reused, including Adam, the warmup/cosine scheduler, equal-batch validation-MSE averaging, and its hardcoded `early_stop > 10` criterion. Its `--early_stop=20` argument is unused in the source. PINN uses its original sample-weighted validation MSE and configurable early stop. Baselines consume x1/y1 exactly as the comparison source, while PINN consumes both adjacent-cycle samples. Best weights are copied at the validation-selected epoch and independently reloaded to verify saved predictions.

`validation_protocol=author_sample_split` follows the published code's 20% sample validation among non-test batteries. The public peer-review response, pp29-30, explicitly explains that split. The main manuscript's approximate 6:2:2 wording must not be used to invent original validation battery IDs.

`validation_protocol=explicit_battery_split` is an additional sensitivity protocol. It requires `split_json` with exactly three records:

```json
[
  {"role": "train", "paths": ["data/XJTU data/2C_battery-1.csv"], "nominal_capacity": 2.0},
  {"role": "valid", "paths": ["data/XJTU data/2C_battery-2.csv"], "nominal_capacity": 2.0},
  {"role": "test", "paths": ["data/XJTU data/2C_battery-4.csv"], "nominal_capacity": 2.0}
]
```

All three battery sets must be nonempty and disjoint. This is not presented as the author's unique original split. The shared `load_explicit_battery_split(args, split_records, sample_size=None)` returns train/valid/test loaders; callers that instrument the DF loader should label the resulting three calls in that order.

An ablation requires PINN, an explicit alpha/beta entry in `overrides`, and nonempty `protocol_evidence`. Only alpha, beta, batch_size and early_stop overrides are accepted. No published ablation grid is generated: Note 3 mentions Grid Search but gives neither its ranges nor an ablation table. Structural sensitivity tasks must be labelled accordingly.

## Counts And Evidence

- Full PINN/CNN/MLP: 11 dataset/batch groups x 3 models x 10 repeats = 330. CNN+MLP alone add 220 tasks. A different protocol cannot reuse the existing 110 as numerically equivalent.
- Main Table 3/Fig. 5 small-sample experiments: XJTU first batch and HUST x 4 training-battery counts x 3 models x 10 repeats = 240.
- Table S7/Fig. S5 extends XJTU to all six batches. All XJTU plus HUST small-sample experiments total 840 tasks.
- Source-native MIT/TJU small-sample entries and transfer-specific tasks are separate from these 840.

Every task records raw task SHA256, effective parameters, implementation and source hashes, actual battery paths, exact per-battery sample boundaries, input-file and tensor identities, paired seed, best epoch, and checkpoint verification. Two models with the same paths/seed are paired controls; no claim is made that their architectures consume identical RNG sequences.

`metrics.json` contains pooled standard metrics, equal-battery macro standard metrics, and equal-battery metrics with the original reversed-argument MAPE convention. `metrics_per_battery.csv` exposes every battery's values. The original analysis scripts infer battery boundaries with SOH jumps and drop a boundary sample; `legacy_author_analysis` replays that behavior as a diagnostic and identifies invalid/empty segments. It is not substituted for the exact recorded battery boundaries.

Across repeats, report both mean and sample standard deviation (`ddof=1`), explicitly noting the paper does not specify ddof. A claim that PINN is better or more stable must be checked against completed baseline errors and variation; successful execution alone is insufficient.

## Verification

```bash
python -u reproduction/test_extended.py
python -m unittest discover -s reproduction -p test_extended_loss_variants.py -v
```

The focused test uses the real three author models, tiny temporary synthetic CSVs, and two CPU epochs. It checks exact battery macro aggregation, both MAPE conventions, legacy boundary behavior, disjoint battery validation, all three best-checkpoint reloads, identical paired data tensors, unchanged author Python files, and immutable completed-task resume. It is not a convergence experiment.

The loss-variant test independently implements all three formulas, performs two epochs with the real PINN model, and compares every best-state tensor and test prediction exactly. It also checks explicit formula gradients, a nonzero but gradient-free label penalty, the first-batch diagnostics, baseline guards, and preservation of the default variant.
