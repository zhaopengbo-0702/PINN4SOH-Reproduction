# C13/C14 环境记录格式补充诊断

本补充只读核对已恢复的实验产物；冻结的 2120 项报告保持不变。

- 核对 640 个任务，96 个候选比较，960 个逐重复配对。
- 任务审计失败：0；逐重复配对通过：960。
- 原始环境 SHA 不同源于 sklearn/CUDA 键名及 SciPy/threads 记录范围差异；补充仅在原始值与各自已完成 coordinator 探测一致时规范化。
- 实测版本：Python 3.10.13、PyTorch 1.13.1+cu116、NumPy 1.24.4、pandas 1.5.3、SciPy 1.10.1、scikit-learn 1.2.2、CUDA 11.6、RTX 3090；线程配置 2。

方向均指 fine-tune 的 RMSE 相对目标独训；按电池等权宏平均后跨 10 次重复汇总。

| 范围 | fine-tune 较低 | 较高 | 持平 |
| --- | ---: | ---: | ---: |
| C13 | 20 | 28 | 0 |
| C14 | 7 | 17 | 0 |
| additional_n2_outside_C14 | 0 | 24 | 0 |

| 范围 | 损失 | 来源 | 较低 | 较高 |
| --- | --- | --- | ---: | ---: |
| C13 | paper_prediction_monotonicity | reproduced_checkpoint | 3 | 9 |
| C13 | paper_prediction_monotonicity | supplied_author_checkpoint | 2 | 10 |
| C13 | published_label_monotonicity | reproduced_checkpoint | 7 | 5 |
| C13 | published_label_monotonicity | supplied_author_checkpoint | 8 | 4 |
| C14 | paper_prediction_monotonicity | reproduced_checkpoint | 0 | 6 |
| C14 | paper_prediction_monotonicity | supplied_author_checkpoint | 1 | 5 |
| C14 | published_label_monotonicity | reproduced_checkpoint | 2 | 4 |
| C14 | published_label_monotonicity | supplied_author_checkpoint | 4 | 2 |
| additional_n2_outside_C14 | paper_prediction_monotonicity | reproduced_checkpoint | 0 | 6 |
| additional_n2_outside_C14 | paper_prediction_monotonicity | supplied_author_checkpoint | 0 | 6 |
| additional_n2_outside_C14 | published_label_monotonicity | reproduced_checkpoint | 0 | 6 |
| additional_n2_outside_C14 | published_label_monotonicity | supplied_author_checkpoint | 0 | 6 |

C13 包含 12 方向 × 2 损失 × 2 来源，共 48 比较；C14 仅 XJTU/1 和 TJU/3 两个目标，共 24 比较。另 24 个 n2 比较单独标为额外范围。来源和损失均分开保留，不能合并为独立重复。

风险与证据边界：

- The frozen strict report is preserved. This supplement changes only the environment matching interpretation after independent corroboration.
- SciPy is not recorded per extended worker; 1.10.1 is corroborated from its completed suite coordinator probe, not inferred from the transfer worker.
- Thread evidence is task/frozen configuration plus the verified runner set_num_threads call; no per-worker get_num_threads measurement was saved.
- GPU model/CUDA runtime match and assigned visibility is checked. Physical GPU UUID, CUDA driver, cuDNN, deterministic-algorithm flags, and BLAS build were not recorded, so bitwise hardware equivalence is not asserted.
- Both methods reuse per-CSV 3-sigma filtering and min-max feature scaling, followed by author 20% cycle-sample validation with random_state=420. Test features are scaled using their entire CSV as in the author loader; this remains an author-protocol limitation.
- For n1 both methods use the same ordered battery/sample split. For n2 the target baseline trains on two batteries while transfer adapts on the first, so sample partitions are intentionally different and per-repeat common seeds do not mean identical training randomness.
- Transfer and target-only loss weights, patience, learning-rate schedules and other recorded hyperparameters differ. Effects describe those methods; they do not isolate pretraining as the only cause.
- Supplied-author checkpoints lack original training seed/split/history provenance; reproduced-source checkpoints have separate lineage and are never merged with supplied weights.
- No retraining, model unpickling, inference replay, significance test, equivalence margin, or physical-parameter identifiability assertion is performed here.

详见 supplement.json、comparisons.csv、repeat_pair_checks.csv 和 environment_evidence.csv。
