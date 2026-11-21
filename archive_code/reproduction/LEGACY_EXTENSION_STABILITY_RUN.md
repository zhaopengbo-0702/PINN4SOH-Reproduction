# 入选十次扩展：方向分布与均值报告

`summarize_legacy_extension_stability.py` 是独立只读后处理，不改冻结比较器或主表，不训练、不载入模型。等待原比较器完成真实110项配对审核后再运行。数据未齐时拒绝生成完整报告。

```bash
PYTHONPATH=/workspace/site-packages python3 /workspace/legacy_extension_stability_tools_v1/summarize_legacy_extension_stability.py \
  --comparison /workspace/legacy_selected_comparison_20261110 \
  --extension /workspace/legacy_selected_support_20261110 \
  --output /workspace/legacy_selected_direction_stability_20261110
```

也可在本地恢复目录运行；传入真实comparison和extension支持目录。输出必须全新，且与输入目录互不包含。依赖标准库和Matplotlib；`--no-plots` 可只用标准库生成CSV/JSON/Markdown。

工具核对支持SHA、完整matched任务、exact repeats1–10和seed、全部指标行、比较器效应行与十次均值/样本SD。它独立从逐次指标重建PINN−MLP、PINN−CNN、fine-tune−source-only效应，再与比较器输出交叉验证。不重复原数组审核，输入comparison必须是原独立数组比较器的完整产物。全部输入/输出和脚本SHA记录在新报告中。

产物包括：

- `paired_seed_effect_directions.csv`：每个seed的两栈效应、正负/零方向、是否改变方向，repeat1明确标记。
- `effect_direction_stability.csv` / `direction_stability_summary.json`：每栈十次mean/median/样本SD/min/max，lower/zero/higher计数、联合方向表、配对效应差、均值方向、严格多数方向；原筛查repeat1与新repeats2–10分别描述。
- `seedwise_method_ranks.csv` / `ten_repeat_method_ranks.csv`：受选常规组每栈每种子四方法的rank，及十次平均误差的rank与平均seedwise rank，精确数值相等时用midrank。
- `effect_seed_directions_*.png/.pdf`：每个比较一张四指标图；每个seed两栈配对线，星号表示参与筛查的repeat1，菱形表示十次mean。300dpi PNG、TrueType PDF，主图相同字体与颜色规范。
- `种子方向与十次均值说明.md`：方向计数与mean并列的阅读表，完整精度留在CSV。

负效应表示PINN误差低于基线，或微调误差低于source-only。只看mean方向可能掩盖少数大效应；9/10种子获益也可能mean劣化。联合方向计数和10mean分别回答观察样本中的方向一致性与平均误差差异，均不等于未来运行获益概率。

repeat1参与筛查触发选择，也包含在十次统计；新2–10单列只是探索性新增切片，不改称独立事先设计的确认样本。所有内容明确post-screen selected，不推广至全2120，不作p值、置信区间或未预设等效判断。exact_zero仅表示数字恰为零，近零仍保留原符号及完整精度，不能据此宣称有实际意义。

MSE与RMSE共享平方误差信息，不能当两个独立证据；battery-macro MSE平均各电池MSE，battery-macro RMSE平均各电池RMSE，平方根在聚合前后不同，因此二者排序也可能不同。电池、指标、两种损失和两栈都不能当额外独立训练重复。

测试：

```bash
python -m unittest discover -s reproduction -p test_legacy_extension_stability.py
```

合成夹具渲染仅用于审核图形布局，所有图和输出标记 `SYNTHETIC FIXTURE — NOT EXPERIMENT RESULTS`。真实extension结果到达前不生成实际结论。
