# 卫星批次：Table S1与Fig S2–S3独立补充

本补充读取已有官方XJTU压缩包中的`Batch-6/Sim_satellite_battery-1.mat`，按Supplementary Information p4指定的首电池重画容量曲线，并逐原始周期保存放电时长。没有训练、改输入、调整数据划分或重新开展电池老化；原2120报告和25图表/21结论CSV不改。

## 已完成的交付

|文件|内容与范围|
|---|---|
|`TableS1_published_schedule_46cycles.csv`|从补充p4逐项转录并视觉核对46个计划周期时长；原表将1↔46至23↔24配对|
|`TableS1_published_schedule.png/.pdf`|同一数表的可读渲染，计划值不是新测量|
|`FigureS2_published_schedule.png/.pdf`|46周期计划的独立曲线；与原FigS2相同数值概念，不宣称像素复制|
|`FigureS3_first_cell_raw_capacity.png/.pdf`|首电池949物理周期的原始记录容量；165个标记full-capacity test用空心点区分，部分放电容量不称完整容量|
|`satellite_all_raw_cycle_capacity_and_stage3_duration.csv`|全部949周期、容量/描述/标记、原stage3时长、固定负电流连续段、记录顺序和完整有符号差值|
|`satellite_duration_order_audit.png/.pdf`|784个非test记录与从首记录开始重复的46周期计划比较；下图保留全部差值|
|`first_43_non_test_raw_records_descriptive.csv`|首43个非test记录的原始时间及打印整数，用于说明观察到的模式，不替换原46周期计划|
|`SATELLITE_INDEPENDENT_CHECKS.json`|原始范围、与已有S4逐点容量核对、time-reset例外和计划对照汇总|
|`SATELLITE_SUPPLEMENT.json`|输入/member/源码指纹、提取/渲染状态与几何QA|
|`supplement_satellite_protocol_source.py`|本轮执行时脚本字节，来源为`reproduction/supplement_satellite_protocol.py`|

TableS1前23值为`5,20,34,41,46,50,54,56,58,60,62,64,68,69,70,71,72,72,72,72,72,72,72`分钟，后23值为其逆序。该数表是论文所述设计，不能自动当作每条原始记录的实际指令。

## 实际发现与解释边界

首电池949个物理周期中，165条含`test capacity`标记，784条为其余记录。全部949个容量与原已保存S4源CSV逐点完全相同，没有以新补充改写既有结果。

作者reader用`relative_time_min==0`划分stage。原始周期17、310、328、687存在额外零点，直接取stage3会得到0时长。本补充保留这一原值，同时固定从完整cycle取`I<-0.1A`的连续负电流段，并在索引不连续或时间倒退处断开，导出所有段；784个非test周期均得到连续放电时长，段电流中位数均为−1.334A。该补充只检查放电时长，不用于重新导出16特征或替换训练输入。

没有原文给定的计划周期→原始cycle映射。本文事先采用“排除带test capacity标记的记录后，保持原始顺序，从首条非test开始重复46周期计划”的直接约定，不尝试调整相位、跳过异常或对齐到误差最小。此约定下102/784条时长按整数分钟四舍五入等于对应计划值；差值最小−67.000、最大+67.001分钟。差值不能单独归为实验偏离，因为计划与原始记录的对应关系尚未建立，记录端点也不包含未记录的边界时间。

首43条非test记录的整数时长呈上升至72再下降至5的模式，72分钟出现11次；TableS1的46值计划中72分钟出现14次。第43条返回5分钟，第44条又为5分钟。此为原记录的描述性观察，不是拟合得到的“修正版论文协议”，也不证明作者原实验有错。原46周期计划和原始顺序均保留，未用43替代46后再宣称完全匹配。

原FigS3展示的容量周期性变化在独立原始数据图中可见。没有把部分放电点插值成新的完整容量，也没有把定期容量测试当成每个周期的独立完整测量。本云平台不能复现实验室装置FigS1或重新开展55节电池物理老化。

## 核验与复跑

官方XJTU归档已有checksum验证，本轮重新读取该MAT成员并核ZIP CRC/成员SHA。成员102808911字节，SHA256为`4c68a1b0aa80502c438258dff7137fe788bc355aedbe977fbf3420d4f2247f3f`；没有把本次成员核对称为重新计算2.44GB整包SHA。Supplement PDF SHA为`f6f0ee6ea624c959487ea680fefc97142cdd88135976201b6e498f780dbe0753`。

图表采用figure-style规范，PNG为300dpi、PDF矢量文字；四张图/表的可见文字碰撞、图外文字、文字与spine碰撞均为0，并已逐图视觉检查。完整949/784点均保留，没有显示抽样。

复跑必须使用新空目录：

```powershell
python -X utf8 reproduction/supplement_satellite_protocol.py --output outputs/full_reproduction/satellite_protocol_supplement_new
```

仅从本补充已保存CSV重新渲染可加`--render-only`，不重新读MAT。第一版`..._20261110/`保留stage3和几何检查失败证据；`..._v2/`为开发修订过程，本目录`..._final/`为交付入口。
