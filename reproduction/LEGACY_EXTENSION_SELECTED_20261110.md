# 本次筛查后的明确扩展：99新训练、110分析

依据已完成40项配对筛查与独立overlay，选择MIT/all、HUST/all两个regular组，每组两种PINN损失及MLP/CNN；迁移只选择paper_equation的HUST→MIT与XJTU→MIT。闭包补XJTU/2C的paper_equation源PINN，HUST同损失源已在常规组内。共11逻辑配置，repeat1原位复用11个；仅新增repeats2–10的99项。不能把MSE与RMSE翻转当作独立证据。

使用v2包解到全新目录 `/workspace/legacy_extension_tools_v2`。本包包括明确选择文件。它新增“两栈环境必须不同”审核：同环境结果标unmatched，不进入十次完整统计。原v1、40计划和训练runner不覆盖。

先用Python核包内全部成员SHA（只读取）：

```bash
python3 - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('/workspace/legacy_extension_tools_v2')
for name,record in json.loads((p/'TOOLS_SHA256.json').read_text()).items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==record['sha256'],name
print('ALL_EXTENSION_V2_MEMBERS_SHA_PASSED')
PY
```

准备真实计划会检查40matched、完整screen artifact/config证据、原repeat1完成时SHA、冻结三计划SHA。此步无训练。

```bash
python3 /workspace/legacy_extension_tools_v2/tools/prepare_legacy_extension.py \
  --selection-path /workspace/legacy_extension_tools_v2/selection/legacy_extension_selection_20261110.json \
  --control-plan /workspace/legacy_extension_tools_v2/plans/legacy_control_plan.json \
  --stage1-plan /workspace/legacy_extension_tools_v2/plans/stage1_plan.json \
  --stage2-plan /workspace/legacy_extension_tools_v2/plans/stage2_reproduced_transfer_plan.json \
  --screening-report /workspace/pinn4soh_legacy_control_comparison_20261110/summary.json \
  --legacy-control-suite /workspace/pinn4soh_legacy_formal_control_20261110 \
  --output /workspace/legacy_selected_support_20261110 \
  --cloud-suite /workspace/pinn4soh_legacy_selected_extension_20261110
```

确认输出 `new_task_count:99`、`analysis_task_count:110`、`reused_repeat1_count:11`。等待独立overlay及原CPU效率任务结束后启动，防止与效率测量重叠。单卡GPU0、一个worker、两条线程：

```bash
CUDA_VISIBLE_DEVICES=0 python3 /workspace/legacy_extension_tools_v2/tools/launch_legacy_extension.py \
  --prefix /workspace/legacy37 \
  --support /workspace/legacy_offline_recovered_v1/payload/support \
  --extension /workspace/legacy_selected_support_20261110 \
  --legacy-control-suite /workspace/pinn4soh_legacy_formal_control_20261110 \
  --repo /workspace/PINN4SOH-main --gpus 0 --workers-per-gpu 1
```

可先加 `--dry-run` 查看命令；它不运行探针或训练。真实执行前会重新验证同legacy环境、同兼容runner、源数据和目录order。只在输入完全相同的中断恢复时加 `--resume`。

完成后在现代审计环境运行：

```bash
PYTHONPATH=/workspace/site-packages python3 /workspace/legacy_extension_tools_v2/tools/compare_legacy_extension.py \
  --extension /workspace/legacy_selected_support_20261110 \
  --legacy-control-suite /workspace/pinn4soh_legacy_formal_control_20261110 \
  --legacy-extension-suite /workspace/pinn4soh_legacy_selected_extension_20261110 \
  --current-suites /workspace/pinn4soh_full_paper_stage1 /workspace/pinn4soh_full_paper_stage2 \
  --output /workspace/legacy_selected_comparison_20261110
```

完整匹配应为 `matched:110`。统计只在每个配置exact repeats1–10全部匹配时输出分布，不进行未预设显著性/等效性检验，不推广至全部2120项。

随后独立收集本次扩展证据与全部110个现代对应任务（包括迁移源闭包）：

```bash
python3 /workspace/legacy_extension_tools_v2/tools/collect_legacy_extension_results.py \
  --workspace /workspace --extension /workspace/legacy_selected_support_20261110 \
  --legacy-control-suite /workspace/pinn4soh_legacy_formal_control_20261110 \
  --legacy-extension-suite /workspace/pinn4soh_legacy_selected_extension_20261110 \
  --current-suites /workspace/pinn4soh_full_paper_stage1 /workspace/pinn4soh_full_paper_stage2 \
  --comparison /workspace/legacy_selected_comparison_20261110 \
  --screening-report /workspace/pinn4soh_legacy_control_comparison_20261110/summary.json \
  --output-prefix /workspace/legacy_selected_backup_20261110
```

收集完成后保留 `EXTENSION_COLLECTION_VERIFIED.json`、`JSON_TRANSFER_MANIFEST.json` 和全部 `results.extension.*.json` 包装分片，使用现有restore工具恢复并检查整包SHA。原40完整证据仍需原40collector另行归档，本工具只收本次入选任务。
