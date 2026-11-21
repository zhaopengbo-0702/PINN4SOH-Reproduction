# PINN4SOH 结果重绘

`plot_full_paper_results.py` 只读取独立审计报告及固定 repeat 1 的曲线证据，不训练、不选择测试误差最小的重复、不重新计算评估指标。依赖 Python 3.9+、NumPy、Matplotlib；已在 Python 3.9.12 / Matplotlib 3.9.1 验证。

## 正式输入

```powershell
python reproduction/plot_full_paper_results.py `
  --report outputs/full_reproduction/final_report `
  --evidence outputs/full_reproduction/final_export `
  --protocol outputs/full_reproduction/paper_protocol.json `
  --references outputs/full_reproduction/source_data `
  --efficiency outputs/full_reproduction/efficiency_cpu/summary.json `
  --output outputs/full_reproduction/final_figures
```

以上路径表示最终审计报告、证据导出和正式 CPU 测速目录，应替换为当前实际产物。输出目录必须新建或为空，不能位于报告或证据目录内。报告与证据均只读。

`--report` 和 `--evidence` 至少指定一个；报告 CSV 缺失时，可从 `suite_report.json` 的 `summaries`、`published_comparisons`、`comparison_effects` 回退读取已有统计。此时没有逐任务预测/逐电池统计的项目仍显示 pending。证据单独输入也受支持：使用导出中复制的 CSV 或位置式 JSON 表。输入全部以 SHA256 记录，证据分片须与索引 SHA256 一致；报告 CSV 与证据中的同名复制表必须一致，报告 JSON 与选用汇总的关键字段亦必须一致。

默认：`--protocol-id author_current --validation-protocol author_sample_split --transfer-design paper_fixed12 --lanes published paperloss`。当前计划把论文 Note 3 参数作为 overrides 承载，不能因参数来自论文就把 protocol_id 改成不存在的 `paper_note3`。不同验证协议需另开输出目录，不能合并。

## 统计口径

- `published` 对应 `published_label_monotonicity`，即发表时期代码的标签单调性项；`paperloss` 对应 `paper_prediction_monotonicity`，即论文 Eq.8 预测单调性项。
- MLP/CNN 统一使用 `author_current` 基线，且每条通道必须与对应 PINN 具有相同 `test_split_sha256`。测试电池不匹配、PINN 锚点缺失或锚点歧义时，相关基线保留 pending，不作为可比结果显示。
- 只采用 `battery_macro` / `standard`：真实测试电池等权宏平均；MAPE 分母为实测 SOH。pooled 和反向 MAPE 不进入主图。
- 表格保留 SOH 小数，数值单元格为 `mean (sample SD)`；图中的 MAE/RMSE 为 SOH 百分点，MAPE 为百分数，统一乘 100。
- 完整统计要求同一科学配置的 expected_repeats=10、n_repeats=10、complete_repeat_set=true，且该指标实际 n=10。样本标准差明确 ddof=1，绝不跨配置合并。
- 不足十次、缺失、多个不相容 comparison_group、缺少指标或测试电池不匹配均保留 pending。CSV 的 observed 保持空值，不以零补齐，partial_mean 单列保留未齐重复的原报告统计。
- 多次作者权重 source-only 可能复用相同检查点，SD=0 不代表十次独立预训练。

## 产物

- Fig. 4a：11 组小面板。事前固定 repeat 1；每个测试电池最多 256 个均匀抽样点。所有已导出的测试电池按路径稳定排序拼接，电池间画分界线；横轴为保留点顺序，不能解释为原始 cycle ID 或完整原始曲线。同组不同方法须具有完全一致的电池集合与实测抽样。
- Fig. 4b：11 组、三模型、MAE/MAPE/RMSE，均值与样本标准差。
- Fig. 5、Fig. S5：小样本 n=1..4；缺失点不连为零。
- Table 2/3/S7：逐配置均值、样本标准差、论文值、复现减论文差值。Table S3-S6 按准确电池 ID 匹配，MIT 同时匹配日期目录。
- Table 4：作者权重与独立复训权重分表，微调与 source-only 分表，同时保留 target-only n=1/2。行表示 target，列表示 source。
- Table S8：参数量与 CPU 速度。只有非 smoke、状态 complete、CPU、batch=1000、forwards=1000、eval/no_grad 的正式测量才填速度；线程数分行，独立进程重复 n 保留在 CSV。不同 CPU/PyTorch 下的差值仅作描述。
- 所有数值图表导出 300-dpi PNG、TrueType PDF、CSV。`repeat1_display_curves.csv` 保存实际绘制的点。
- `schematics/`：Fig. 1/Fig. 6 可编辑 Mermaid 源图和代码核对说明。它们是计算流程示意，不是独立实体电池实验或已完成所有迁移方向的证明。
- `figure_table_render_status.json/csv`：论文 25 项图表的状态。Fig. S1-S3 的硬件及实验协议为原文材料审计，不标为数值复现。原始数据类 Fig. 2/3/S4 及库存/协议类表格需合并独立数据审计产物。
- `comparison_effects_audited.csv`、`published_comparisons_audited.csv`：复制独立分析器的差值/效果结果，不在绘图阶段另选统计方法。
- `figure_qa.json`：实际渲染文字边界、碰撞与表格单元格溢出检查；未处于视域的 Matplotlib 备用 tick 不计为可见文字。`manifest.json`：输入/输出哈希、脚本哈希、完整性数量；`结果图表简报.md`：中文口径说明。

## 验证

```powershell
python reproduction/test_plot_full_paper_results.py
python reproduction/test_plot_full_paper_results.py `
  --fixture outputs/full_reproduction/plot_layout_fixture_new --render
```

测试包含通道分离、基线共享、十次重复、缺失指标、歧义配置、测试电池哈希、源权重模式、固定 repeat 1、证据单独读取、SHA256 篡改拒绝、精确电池/日期、排除 pooled/反向 MAPE、JSON 回退及 smoke 测速不作为正式结果。

合成夹具的报告与索引含 `synthetic_layout_fixture=true`，所有 PNG/PDF 带醒目标记；其 CSV、数值和图不能作为复现实验结果。布局验证输出与正式结果必须分目录。运行后检查 `all_text_geometry_passed=true`，并人工查看主图、密集小面板和表格；几何通过不替代数据口径审计。
