# Transfer Protocol And Implementation Audit

This audit distinguishes the published paper, publication-era code, and current corrected code. The existing 110 completed main experiments are preserved. A successful smoke test is evidence of executable code, not evidence that the paper's transfer results have been recovered.

## Evidence And Version History

- Paper Table 4, page 8: all 12 directed source/target dataset pairs; XJTU batch 1 represents XJTU, and TJU batch 3 represents TJU. Fine tuning uses the first target training battery, with the regular target test batteries. The reported comparison includes source-only, source-to-target fine tuning, and target-only training with one or two batteries.
- Supplementary Note 3, PDF page 14: regular-training alpha/beta are XJTU `(0.7, 20)`, TJU and MIT `(1, 50)`, HUST `(0.5, 80)`; batch sizes are 256 for XJTU and 512 otherwise. It does not disclose transfer-specific learning rates, source-checkpoint selection, checkpoint training seeds, or a distinct fine-tuning normalization rule.
- The author's public peer-review response clarifies independent held-out test batteries followed by shuffled cycle-level validation within the remaining batteries. The ambiguous main-text 6:2:2 wording alone must not be used to invent separate validation batteries. The runner also accepts an explicitly requested separate-battery validation protocol, which is recorded as a different protocol.
- Latest publication-era repository snapshot: [`d26bc1f40dc0cdf34405cd5e599a6f6af09742e9`](https://github.com/wang-fujin/PINN4SOH/commit/d26bc1f40dc0cdf34405cd5e599a6f6af09742e9), 2024-04-23. Its regular and adaptation code uses `relu(y2-y1).sum()` as the physical penalty. Labels do not require gradients, so this term has **identically zero gradient with respect to model parameters**. It can change the displayed total loss but not an Adam update with the same remaining losses. This diagnosis follows the expression's data dependence, not the quality of a fitted curve.
- [`1e94ec8ed9af35134c8afbd6c576b80f3228b7a0`](https://github.com/wang-fujin/PINN4SOH/commit/1e94ec8ed9af35134c8afbd6c576b80f3228b7a0), 2024-07-11: regular code changes that term to `relu(u2-u1).sum()`, matching the prediction-monotonicity form of paper Equation 8.
- [`7d6fe13ce80affb404157d836e8c2f079d97ee20`](https://github.com/wang-fujin/PINN4SOH/commit/7d6fe13ce80affb404157d836e8c2f079d97ee20), 2024-10-16: regular code changes it to `relu((u2-u1)*(y1-y2)).sum()`.
- [`bf7a93148de6a7e249c1b053bd60fe3c9a3dc1f0`](https://github.com/wang-fujin/PINN4SOH/commit/bf7a93148de6a7e249c1b053bd60fe3c9a3dc1f0), 2024-10-16: adaptation code changes directly from the label-only penalty to the current capacity-regeneration expression, and changes adaptation beta from 50 to 0.2. Alpha remains 0.7.
- Current reference [`cc3cc5053caf38f04e0665f7f88cb109144d035e`](https://github.com/wang-fujin/PINN4SOH/commit/cc3cc5053caf38f04e0665f7f88cb109144d035e), 2026-09-02: `fix.md` records bug fixes and parameter differences from the supplement. It does not remove the need to distinguish the earlier training-expression changes.

The two audited core-source snapshots are saved under `outputs/full_reproduction/upstream_versions/<commit>/`, with `snapshot_metadata.json` recording dates, official URLs, Git blob SHA1 and SHA256. They contain the model, entry scripts, loader, utility and README/fix files, not complete repository/data archives. All downloaded bytes match the official Git blob hash. `fix.md` is explicitly recorded as absent from the April 2024 snapshot. Historical source files were downloaded for inspection and were not executed.

## Native Transfer Configurations

`transfer_task_map.json` contains exact machine-readable mappings and all task specifications. Native script coverage is 19 configurations times 10 repetitions, or 190 fine-tuning tasks, covering all 12 directions:

| Direction | Zero-Based Source Batches | Zero-Based Target Batches | Adaptation Learning Rates |
|---|---|---|---|
| TJU to XJTU | 2, 2, 2, 1, 0, 1 | 0, 1, 2, 3, 4, 5 | .0004, .01, .0005, .002, .003, .0006 |
| XJTU to TJU | 3, 3, 2 | 0, 1, 2 | .003, .002, .002 |
| HUST to MIT | -1 | -1 | .005 |
| MIT to HUST | -1 | -1 | .0002 |
| XJTU to HUST / MIT | 0 | -1 | .001 |
| TJU to HUST / MIT | 2 | -1 | .001 |
| HUST / MIT to XJTU | -1 | 0 | .001 |
| HUST / MIT to TJU | -1 | 2 | .001 |

The separate `paper_fixed12` design contains 12 configurations and 120 tasks. It enforces XJTU index 0 and TJU index 2 for both source and target. In particular, XJTU-to-TJU changes the native script's XJTU source index 2 to index 0. This conflict is explicit in each configuration. Learning rates retain the script's target-specific values; Note 3 does not independently establish those choices. `transfer_design` describes these batch mappings, while `protocol_id` and `loss_variant` separately describe the training implementation. A fixed-batch task is not automatically an exact-paper-equations task.

## Fine-Tuning Behavior

The current author adaptation defaults are batch size 128, one target training battery, at most 200 adaptation epochs, early-stop patience 10 with the author's strict `counter > patience` stopping condition, alpha 0.7 and beta 0.2. The adaptation Adam optimizer has the task's constant learning rate. The inherited pretraining scheduler is constructed but never stepped during adaptation.

Both the complete `solution_u` and complete `dynamical_F` states are loaded from the source checkpoint. `dynamical_F` is **not reinitialized**. Its parameters are frozen, and all `solution_u` parameters remain trainable. The source-only `Test` call first places the entire model in evaluation mode. Subsequent adaptation epochs train `solution_u` only, leaving frozen `dynamical_F` in evaluation mode, including its dropout. The runner preserves this ordering, verifies the frozen state hash before/after, and verifies that the saved best checkpoint reproduces the saved validation-selected test predictions.

Calling a learned `dynamical_F` transferable does not establish a uniquely identifiable electrochemical law or recovery of battery internal chemistry. The authors' response limits the physical constraints and does not supply electrochemical boundary conditions. Freezing a network is a computational action; physical interpretation requires separate evidence.

## Source Checkpoint Lineage

Eleven supplied checkpoints exist. Each has finite tensors and the same compatible architecture: `solution_u` 7,781 parameters and `dynamical_F` 5,881 parameters. The exact SHA256, tensor-shape table and per-module state hashes are in `transfer_task_map.json`. The GitHub file history for `model_XJTU_0.pth` contains one upload, [`4d8f143f332ba2dcc7f769ab8d024487d71e9b53`](https://github.com/wang-fujin/PINN4SOH/commit/4d8f143f332ba2dcc7f769ab8d024487d71e9b53), dated 2023-09-20. Training seeds, source split manifests, checkpoint-selection histories and training-version provenance are not supplied; these fields remain `null` or unverified. Architecture compatibility is not evidence that the weights were trained with the current corrected losses.

Two explicit source modes are supported:

- `supplied_author_checkpoint`: loads the matching dataset/batch checkpoint from the author repository, records its SHA256, and records the missing training provenance. The 190 native tasks use this mode by default.
- `reproduced_checkpoint`: requires an explicit source-run directory, its complete manifest, config, split record and validation log. It rejects smoke/short runs, mismatched dataset/batch or source inventory, and checkpoints without the best-state snapshot fix. `select_reproduced_source` can choose among completed source repetitions using **source validation MSE only**, with deterministic tie breaking; it never uses target test performance. The existing 110 current-implementation source runs may be reused when their recorded inputs match. A separate source pretraining task family is available when a different loss or split protocol is required; it writes a new output directory.

## Runner Contract And Data Boundaries

`run_transfer.py --task-json <single-task.json> --repo <author-repo> --output <one-task-folder> [--resume] [--smoke] [--allow-cpu]` uses the same manifest contract as the full-suite scheduler. It respects the externally assigned `CUDA_VISIBLE_DEVICES`, initializes logical device 0 in that allocation, and restores the allocation after imports of author modules that overwrite the variable.

Output contains config and actual parameter records, environment/source inventories, exact battery sample boundaries, source checkpoint lineage, source-only true/prediction arrays, fine-tuned true/prediction arrays, `model.pth`, logs, per-battery metrics, and top-level task status/artifact hashes. `task_sha256` hashes the unmodified input task. Hardware device-slot numbering is recorded but excluded from the semantic resume hash so the same task can be scheduled on another equivalent allocated slot. Implementation hashes are included in the resume configuration.

Default loading retains the author's first-target-battery and cycle-level 80:20 validation behavior. An explicit reference `split_json` can replace OS-order-dependent target test IDs while retaining cycle-level validation. The optional `explicit_battery_split` mode instead requires three explicitly supplied, disjoint train/valid/test battery lists. No validation battery IDs are guessed. Overlap between target adaptation and held-out target tests is rejected. Unknown supplied-checkpoint training history remains an explicit limit to source-leakage verification.

Adjacent cycle pairs are formed before the random train/validation split. Training supervises both endpoints, whereas validation predicts the first endpoint. A cycle in a validation pair can therefore also appear as the second endpoint of a neighboring training pair. This is a validation-dependence limitation, even when held-out test batteries are independent. Faithfully retaining this procedure does not establish strict independence of validation cycles.

The runner records pooled standard metrics and exact-battery macro metrics. Standard MAPE divides by measured SOH; an additional author-reversed MAPE retains the published code's prediction denominator for diagnosis. Source-only and adapted predictions use the same target test order. Target-only one-/two-battery comparisons are separate small-sample tasks run by the extended runner.

Three explicitly selected physical-loss variants are available without editing the author source: `author_current`, `paper_prediction_monotonicity`, and `published_label_monotonicity`. The last is recorded as having zero model-parameter gradient. Changing the loss expression or original weights is a distinct task/configuration, not a silent correction.

## Verification

`python -X utf8 -m unittest discover -s reproduction -p test_run_transfer.py -v` passed eight tests. These include actual two-epoch author-data adaptation, frozen/source-state hashes, `solution_u` updates, best-checkpoint prediction replay, identical resume, output-tamper rejection, explicit disjoint battery splits, paper-equation and original-label loss execution in the independent wrapper, source pretraining, all-direction mapping coverage, raw-task hashes, and source selection independent of target test scores. The watched author model, adaptation source and provided checkpoint hashes were unchanged. These local CPU tests do not substitute for cloud CUDA tests or full convergence experiments.
