# 16特征独立导出器与验收

`export_replacement_features.py`补齐一个可以实际运行的16特征导出入口。它产生新CSV、逐窗口诊断和来源/代码/产物SHA，不覆盖作者CSV、训练输入或冻结报告。它没有宣称两个发布slope列已恢复。

补充Note2的式(7)明确给出端点差除以时间差。现在固定使用观察端点和物理秒：voltage slope为V/s，current slope为A/s；MIT派生镜像的电流单位仍有C-rate/A歧义，必须显式标`native_unit_unresolved`，斜率为原生数值/s，不静默换算。两个slope标为`replacement`，定义在比较发布值之前固定，不拟合缩放常数、不按残差选择候选。先前32种候选的未恢复结论不变。

两种计算profile均可用：

- 默认`released_audit`：14项沿用已审计发布约定，population SD、biased Fisher excess峰度、biased skewness、min-max后概率归一化熵、记录Q末减初和物理秒。XJTU的4.195V为此前明确披露的探索性固定边界，HUST3.595V为公开reader边界；不能把前者说成论文明确写出的阈值。
- `paper_note2`：按补充公式使用sample SD、论文峰度/偏度公式、电流梯形积分Ah和原文电压边界。熵归一化仍是明确假设；原文没唯一指定。MIT因电流物理单位未解决，拒绝这个积分profile。此profile没有按整篇全部电池独立验证，不与默认的14项发布约定混用。

输入支持官方XJTU MAT、checksum核验后的官方XJTU/TJU/HUST ZIP精确member、MIT本地派生HDF5及已有物理映射、规范CSV。规范CSV必须包含`raw_cycle,voltage_V,current_A,t_seconds,charge_Ah,cc_stage,cv_stage`；stage标记为0/1，由调用方明确给定，不声称该接口自动恢复未知充电阶段。MIT HDF5入口复用未改动的作者阶段选择，SHA必须核验，仍不是原MAT字节来源链。

时间不严格递增、非有限输入或不足4点的窗口保留空值和原因，不排序/插值/填补。常量窗口的峰度/偏度/熵未定义，写空值并保存诊断。斜率保留符号，不取绝对值。导出记录16项特征的单位和各自状态；比较只在导出后执行。

执行本次全部有界验收：

```powershell
python -X utf8 reproduction/validate_replacement_features.py --output outputs/full_reproduction/replacement_feature_validation_20261111/new_acceptance
```

输出路径必须是新目录。上述入口顺序执行5项解析/异常测试，再从四个数据集固定一节电池各取3个预先对应周期，不训练模型，不全量重提特征。官方ZIP在本地完整重算SHA，MIT H5同样完整重算SHA。每次产物manifest记录runtime、调用参数、代码和源文件SHA。既有行映射用了四项电流统计，不能拿这四项匹配当作映射正确的独立证据。

本次实际验收结果见`outputs/full_reproduction/replacement_feature_validation_20261111/final_v2/VALIDATION.json`，详细报告在上级README。初期试运行的上级`xjtu_real3/tju_real3/hust_real3/mit_real3`及`final/`保留为开发历史；最终验收只读`final_v2/`，避免引用旧代码SHA。
