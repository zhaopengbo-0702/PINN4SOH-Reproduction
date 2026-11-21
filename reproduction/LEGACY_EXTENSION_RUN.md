# 筛查后按明确组别扩展至十次独立重复

本工具只用于完整 40 项原软件栈筛查后的入选敏感性分析。它保留原 40 项计划、现代训练脚本和历史结果。当前没有真实入选清单，也没有启动扩展训练。

## 选择范围与训练数量

选择清单必须明确列出 regular 的 dataset/batch 组，或冻结 40 项中的 transfer repeat1 task ID；不接受“全部 2120”等宽泛选择器。regular 每组包含两个 PINN 损失版本和共享 MLP、CNN。transfer 精确选择方向和损失；依赖闭包只补同损失、同 repeat 的源 PINN。目标与源数据集的 seed 偏移可能不同，分别对齐现代任务，不强求源 seed 等于目标 seed。

原 repeat1 继续从 `/workspace/pinn4soh_legacy_formal_control_20261110` 的原路径读取，不复制、不重写 metadata、不重训。新 suite 仅训练 repeats2–10，分析清单组合原 repeat1 和新的九次。最大为 360 项新训练、400 项分析任务，保持 GPU0、一个 worker、两条线程。

选择文件结构如下。示例中的 `regular_groups` 和 `transfer_repeat1_task_ids` 留空，因此不能直接生成训练计划；待完整筛查后填写真实条目和证据。

```json
{
  "format": "pinn4soh-legacy-extension-selection-v1",
  "reason": "依据完整筛查的明确观察，说明为何选择这些组",
  "regular_groups": [],
  "transfer_repeat1_task_ids": [],
  "evidence_refs": ["筛查报告及具体CSV行或图表位置"]
}
```

## 准备与复核

`prepare_legacy_extension.py` 使用现代审计环境，无训练动作。必须有完整 `matched:40` 且无分析错误的筛查报告，以及未改变的三份冻结计划。`--output` 为全新目录，`--cloud-suite` 为全新独立 Linux 绝对路径。

```bash
python3 /workspace/legacy_extension_tools_v1/tools/prepare_legacy_extension.py \
  --selection-path /workspace/selected_legacy_extension.json \
  --control-plan /workspace/legacy_single_gpu_v3/control/legacy_control_plan.json \
  --stage1-plan /workspace/legacy_extension_tools_v1/plans/stage1_plan.json \
  --stage2-plan /workspace/legacy_extension_tools_v1/plans/stage2_reproduced_transfer_plan.json \
  --screening-report /workspace/pinn4soh_legacy_control_comparison_20261110/summary.json \
  --legacy-control-suite /workspace/pinn4soh_legacy_formal_control_20261110 \
  --output /workspace/legacy_selected_support_v1 \
  --cloud-suite /workspace/pinn4soh_legacy_selected_extension_v1
```

复核三个输出：`extension_plan.json` 的新任务和依赖、`extension_analysis_manifest.json` 的原位复用路径及现代逐次对应、`extension_design.json` 的选择理由/输入SHA/任务上界。这里的选择是筛查后的条件性选择，不能解释成事先随机抽样。

## 单卡执行

先加 `--dry-run` 检查命令，不运行 GPU 探针或训练。真实启动验证原 repeat1 全部完成时文件 SHA、旧兼容脚本 SHA、同一 legacy runtime，然后调用原冻结兼容 coordinator。每次 preflight 留在新的带时间戳文件中，resume 不覆盖前次观察。

```bash
CUDA_VISIBLE_DEVICES=0 python3 /workspace/legacy_extension_tools_v1/tools/launch_legacy_extension.py \
  --prefix /workspace/legacy37 \
  --support /workspace/legacy_offline_recovered_v1/payload/support \
  --extension /workspace/legacy_selected_support_v1 \
  --legacy-control-suite /workspace/pinn4soh_legacy_formal_control_20261110 \
  --repo /workspace/PINN4SOH-main --gpus 0 --workers-per-gpu 1
```

只有完全相同输入继续时使用 `--resume`。不要同时启动第二个训练进程。源训练先于同次目标迁移，目标只读取清单中的同次源 checkpoint。

## 独立数组统计

使用现代 Python+NumPy，不载入模型 pickle。下面的旧/new suite 可替换为本地逐字节恢复目录；原云端逻辑路径仍保存在清单中，文件证据不因恢复到本地而重写。

```bash
PYTHONPATH=/workspace/site-packages python3 /workspace/legacy_extension_tools_v1/tools/compare_legacy_extension.py \
  --extension /workspace/legacy_selected_support_v1 \
  --legacy-control-suite /workspace/pinn4soh_legacy_formal_control_20261110 \
  --legacy-extension-suite /workspace/pinn4soh_legacy_selected_extension_v1 \
  --current-suites /workspace/pinn4soh_full_paper_stage1 /workspace/pinn4soh_full_paper_stage2 \
  --output /workspace/legacy_selected_comparison_v1
```

分析重新计算每块电池指标再等权平均，保留逐次、逐电池、预测漂移、方法效应与迁移前后效应。只有 exact repeats1–10、十个不同 seed 且全部匹配才输出十次 mean、总体/样本 SD 和配对差异分布。缺失、分割/超参不匹配或审核失败都不会补零或进入完整十次统计。检查 raw/effective task、全部完成时 SHA、代码、source/data、legacy cohort，迁移还验证源 model/config/splits/manifest SHA 与训练 seed、验证选择和 split。

这是入选组的软件栈敏感性分析，不作未预设显著性、等效性或全部 2120 项环境独立结论。MSE 与 RMSE 为同一误差平方关系，不计为两项独立证据。当前运行探针的 TF32 等标志也不证明历史现代 worker 的标志。

## 独立收集与恢复

使用 `collect_legacy_extension_results.py`，不改变冻结 12+40 收集器。它要求完整匹配，归档所有入选 legacy 十次和现代十次、源闭包、suite manifests、扩展支持、screen报告/CSV和扩展分析；不复制运行时二进制和全2120结果。

```bash
python3 /workspace/legacy_extension_tools_v1/tools/collect_legacy_extension_results.py \
  --workspace /workspace --extension /workspace/legacy_selected_support_v1 \
  --legacy-control-suite /workspace/pinn4soh_legacy_formal_control_20261110 \
  --legacy-extension-suite /workspace/pinn4soh_legacy_selected_extension_v1 \
  --current-suites /workspace/pinn4soh_full_paper_stage1 /workspace/pinn4soh_full_paper_stage2 \
  --comparison /workspace/legacy_selected_comparison_v1 \
  --screening-report /workspace/pinn4soh_legacy_control_comparison_20261110/summary.json \
  --output-prefix /workspace/legacy_selected_backup_v1
```

输出 tar.gz 和每片最多64MiB的 JSON/base64 wrappers，逐成员 SHA 和逐片 roundtrip 后写 `EXTENSION_COLLECTION_VERIFIED.json` 与 `JSON_TRANSFER_MANIFEST.json`。下载所有 wrappers 和 transfer manifest 后，可用已有 `restore_backup_json.py --help` 查看标准恢复参数，再将恢复的 tar.gz 安全解到全新目录。
