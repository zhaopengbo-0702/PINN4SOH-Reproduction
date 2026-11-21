# 16特征替代实现：实际执行结果

最终验收入口：`final_v2/VALIDATION.json`。5项解析和异常处理测试通过；四个数据集固定一节电池、各3个预先对应周期均实际导出16特征。每个源文件完整重算SHA通过，全部导出artifact SHA复核通过。代码与最终manifest记录相符，输出不是合成训练结果。

| 样本 | 周期 | 14项既有约定，误差≤1e-8的有限配对 | 两个替代slope，误差≤1e-8配对 |
|---|---|---:|---:|
| XJTU 2C battery1 | 10、12、13 | 42/42 | 0/6 |
| TJU Dataset1 CY25-025_1-#1 | 1、2、4 | 42/42 | 0/6 |
| HUST 1-1 | 12、14、15 | 42/42 | 0/6 |
| MIT May battery1 | 8、15、16 | 21/42 | 0/6 |

阈值仅是描述误差的审计阈值。MIT14项最大绝对残差1.0040634070840326e-6，仍保留派生float32镜像和电流单位限制；不能声称全部14项逐值相等。两个slope使用预先固定的观察端点差/物理秒，确实与发布列不同；该差异明文保留，不调参消除。因此此次完成的是可运行的替代实现，不是作者未公开提取流程的精确恢复。

最终四组manifest含`feature_schema`，14项标`prior_audited_convention`，2项标`replacement`；`window_diagnostics.json`保存点数、端点、物理秒和异常，非MIT还保存所选点索引；MIT保存原物理cell/cycle/barcode映射。`features_16.csv`只在全新目录写入，不改训练输入。来源、代码和产物SHA可逐项重核。

解析fixture验证线性电压V/s、线性电流负斜率、积分、统计量和概率熵；异常fixture验证短窗口、重复/倒序时间和常量未定义值。真实数据与解析fixture在VALIDATION中分开记录，没有将测试通过解释为论文预测结论复现。

代码入口及参数说明：`reproduction/REPLACEMENT_FEATURE_EXPORTER.md`。本次未重新训练替代特征下的所有模型，也未全量重复计算已有原始特征审计。
