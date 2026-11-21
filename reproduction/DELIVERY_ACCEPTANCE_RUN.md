# 已有结果与替代实现的统一只读验收

```powershell
python reproduction/verify_reproduction_delivery.py --feature-validation outputs/full_reproduction/replacement_feature_validation_20261111/final_v2 --search-validation outputs/full_reproduction/replacement_search_validation_20261111 --output <全新验收目录>
```

扩展110项完成本地独立重算后，加`--extension-comparison <本地比较目录>`。没有这个参数时会明确记录扩展未在此次验收中包含，不能称整篇所有交付已验收。

此入口读取冻结主2120报告、21结论和25图表CSV，核固定SHA；再读取特征`VALIDATION.json`指定的真实样本manifest，核当前代码依赖和产物SHA，独立重算端点/秒slope及逐列发布残差统计。两个slope保留replacement状态和原发布列不吻合事实；MIT派生H5局部精度残差保留。

替代搜索验收核`ACCEPTANCE.json`所列代码、日志、计划与实际小验证全部SHA，重新从validation数组算MSE、从history核best epoch、从候选均值重选配置，并核候选worker只接收train/valid、test在选定后绑定。也重算选定test数组MSE/MAE。完整192项示例grid只建立计划，未执行；2epoch小验证只证明路径可运行，不当性能结论。

输出`ACCEPTANCE.json`及`README.md`。输出必须全新；不训练、不联网、不修改冻结证据。此前`delivery_acceptance_20261111_tools/`使用开发中feature final，保留为历史；当前已执行通过的工具验收为`outputs/full_reproduction/delivery_acceptance_20261111_tools_v3/`，采用feature final_v2，并核解析测试日志及验收器本身所引代码SHA。

真实110扩展应先以`rerun_legacy_extension_windows_io.py`独立重算，核110matched、四个逐次CSV与云逐字节一致、其余JSON解析相同及两张十次统计表仅允许命名SD字段至多1个binary64 ULP的运行时数值差，再进行`summarize_legacy_extension_stability.py`方向和十次均值分析。均值、方向、计数及逐次指标不使用容差。统一验收只接纳该独立重算的`LOCAL_RERUN_IO_VERIFIED.json`，不替代collector外inventory、安全解包和成员SHA审计。

已完成包含真实110扩展的最新统一验收：`outputs/full_reproduction/delivery_acceptance_20261111_final/`。扩展恢复和本地重算详情见`outputs/full_reproduction/legacy_extension_recovered_20261111/扩展110_本地独立复核.md`。

验收通过不恢复作者未知历史seed/grid/checkpoint来源、MIT原MAT单位、卫星46周期映射、未测历史backend设置或物理老化装置。原论文结论成立与否仍按真实实验裁定，不能以新代码存在替代数值证据。
